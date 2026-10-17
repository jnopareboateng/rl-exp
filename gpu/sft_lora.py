#!/usr/bin/env python3
"""Completion-only LoRA SFT for the preformatted A1/A2 Train and Val files.

CPU tests cannot prove GPU, kernel or memory behaviour. GPU training, numerical
loss parity, kernel execution and full optimizer/RNG resume are UNTESTED here.
Dry runs and the pure data, batch and checkpoint functions need only stdlib.
See gpu/SFT.md for the install pins, clipping rule and first-run checks.
"""

import argparse
import ast
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

END = "<|im_end|>"
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
EPOCH_FILES = (*ADAPTER_FILES, "revision.json")
# The code that decides what is trained and how; a resume under different code is refused (see objective_sha256).
OBJECTIVE_FUNCTIONS = ("fit_lengths", "prepare_rows", "token_batches", "epoch_batches", "collate", "completion_loss",
                       "accumulate_update", "step_optimizer")
PROJECTIONS = {
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
}


class PackError(ValueError):
    """A diagnostic authored by this pack. It contains no private data."""


def print_failure(action, error):
    message = str(error) if isinstance(error, PackError) else "Data values are omitted."
    print(f"{action} failed ({type(error).__name__}): {message}", file=sys.stderr)
    for frame in traceback.extract_tb(error.__traceback__):
        print(f"  {Path(frame.filename).name}:{frame.lineno} in {frame.name}", file=sys.stderr)


def precision_for_capability(capability):
    if tuple(capability) < (7, 5):
        raise PackError("This training pack requires compute capability 7.5 or newer")
    return "bfloat16" if tuple(capability) >= (8, 0) else "float16"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_from_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A1", "A2"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--max-tokens-per-batch", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overlength", choices=("drop", "clip"), default="drop")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--log-steps", type=int, default=5)
    parser.add_argument("--stop-after-seconds", type=float, default=3600)
    parser.add_argument("--allow-slow-kernels", action="store_true",
                        help="on compute capability 8.0+, train with the torch fallback when a fast kernel is missing")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    prefix = "a1_trace" if args.arm == "A1" else "a2_answer"
    args.train = args.train or Path(f"data/processed/arms/{prefix}_train.jsonl")
    args.val = args.val or Path(f"data/processed/arms/{prefix}_val.jsonl")
    args.out = args.out or Path(f".agent-runs/sft_{args.arm}")
    default_len, default_budget = (8192, 16384) if args.arm == "A1" else (2048, 8192)
    args.max_len = args.max_len if args.max_len is not None else default_len
    args.max_tokens_per_batch = (args.max_tokens_per_batch if args.max_tokens_per_batch is not None
                                 else max(default_budget, args.max_len))
    validate_config(vars(args))
    return args


def validate_config(config):
    for key in ("max_len", "max_tokens_per_batch", "batch_size", "rank", "epochs", "save_steps", "log_steps"):
        if type(config[key]) is not int or config[key] < 1:
            raise PackError(f"{key} must be a positive integer")
    if config["max_len"] < 2 or config["max_tokens_per_batch"] < config["max_len"]:
        raise PackError("Token budget must cover max_len, and max_len must be at least 2")
    if not math.isfinite(config["lr"]) or config["lr"] <= 0:
        raise PackError("lr must be positive and finite")
    if not math.isfinite(config["warmup_ratio"]) or not 0 <= config["warmup_ratio"] < 1:
        raise PackError("warmup_ratio must be in [0, 1)")
    if not math.isfinite(config["stop_after_seconds"]) or config["stop_after_seconds"] <= 0:
        raise PackError("stop_after_seconds must be positive and finite")
    if config["seed"] < 0 or config["seed"] >= 2**32:
        raise PackError("seed must be in [0, 2**32)")
    if Path(config["train"]).resolve() == Path(config["val"]).resolve():
        raise PackError("Train and Val must be separate files")
    if config["overlength"] not in ("drop", "clip"):
        raise PackError("Unknown overlength policy")


def read_rows(path, split):
    if "test_t" in path.stem.lower() or "test_r" in path.stem.lower():
        raise PackError("Test files cannot be used for SFT")
    rows, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except ValueError:
                raise PackError(f"Invalid JSON at row {number}; data omitted") from None
            if (not isinstance(row, dict) or not isinstance(row.get("id"), (str, int))
                    or not isinstance(row.get("prompt"), str) or not isinstance(row.get("completion"), str)
                    or not row["prompt"].endswith("<think>\n") or not row["completion"].endswith(END)
                    or row.get("split", split) != split or row["id"] in seen):
                raise PackError(f"Invalid SFT row {number}; data omitted")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise PackError("The data file has no rows")
    return rows


def fit_lengths(prompt_length, completion_length, max_len, policy):
    """Clip only the left of the prompt. Never remove a target or invent EOS."""
    if prompt_length < 1 or completion_length < 1:
        raise PackError("Prompt and completion must have tokens")
    if prompt_length + completion_length <= max_len:
        return prompt_length, "kept"
    if policy == "clip" and completion_length < max_len:
        return max_len - completion_length, "clipped"
    return 0, "dropped"


def prepare_rows(rows, tokenizer, max_len, policy):
    prepared = []
    counts = dict(rows=len(rows), kept_rows=0, clipped_rows=0, dropped_rows=0,
                  tokens=0, completion_tokens=0, removed_prompt_tokens=0)
    end_ids = tokenizer.encode(END, add_special_tokens=False)
    if not end_ids:
        raise PackError("Tokenizer has no end marker")
    for row in rows:
        # Separate encoding preserves the exact generation prefix at the boundary.
        prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
        completion = tokenizer.encode(row["completion"], add_special_tokens=False)
        if completion[-len(end_ids):] != end_ids:
            raise PackError("Completion tokenization lost the end marker")
        keep, action = fit_lengths(len(prompt), len(completion), max_len, policy)
        if action == "dropped":
            counts["dropped_rows"] += 1
            continue
        counts["clipped_rows"] += action == "clipped"
        counts["removed_prompt_tokens"] += len(prompt) - keep
        ids = prompt[-keep:] + completion
        prepared.append(dict(input_ids=ids, labels=[-100] * keep + completion))
        counts["tokens"] += len(ids)
        counts["completion_tokens"] += len(completion)
    counts["kept_rows"] = len(prepared)
    return prepared, counts


def token_batches(indices, lengths, budget):
    """Budget includes padding: rows times the longest sequence in the batch."""
    batches, batch, longest = [], [], 0
    for index in indices:
        size = lengths[index]
        if size < 1 or size > budget:
            raise PackError("A row cannot fit the token budget")
        if batch and max(longest, size) * (len(batch) + 1) > budget:
            batches.append(batch)
            batch, longest = [], 0
        batch.append(index)
        longest = max(longest, size)
    if batch:
        batches.append(batch)
    return batches


def epoch_batches(lengths, budget, batch_size, seed, epoch):
    """One item is one optimizer update of batch_size rows (last may be short)."""
    rng = random.Random(seed + epoch)
    indices = list(range(len(lengths)))
    rng.shuffle(indices)
    remainder = len(indices) % batch_size
    tail, indices = indices[:remainder], indices[remainder:]
    indices.sort(key=lengths.__getitem__)
    groups = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    rng.shuffle(groups)
    if tail:
        groups.append(tail)
    for group in groups:
        rng.shuffle(group)
    return [token_batches(group, lengths, budget) for group in groups]


def advance_step(state, steps_in_epoch):
    if not 0 <= state["next_step"] < steps_in_epoch:
        raise PackError("Step cursor is outside the epoch")
    return {**state, "next_step": state["next_step"] + 1, "global_step": state["global_step"] + 1}


def finish_epoch(state, steps_in_epoch, metrics):
    if state["next_step"] != steps_in_epoch:
        raise PackError("Cannot finish an incomplete epoch")
    return {**state, "epoch": state["epoch"] + 1, "next_step": 0,
            "history": [*state["history"], metrics]}


def is_best(val_loss, best_val_loss):
    """The lowest full-epoch Val loss wins. A tie keeps the earlier adapter."""
    return best_val_loss is None or val_loss < best_val_loss


def dry_plan(args, train, val):
    result = dict(mode="dry_run", estimate="ceil(UTF-8 bytes / 4) per segment; no tokenizer or model loaded",
                  arm=args.arm, max_len=args.max_len, max_tokens_per_batch=args.max_tokens_per_batch,
                  batch_size=args.batch_size, epochs=args.epochs, overlength=args.overlength)
    for name, rows in (("train", train), ("val", val)):
        lengths, clipped, dropped, tokens = [], 0, 0, 0
        for row in rows:
            p, c = (max(1, math.ceil(len(row[key].encode("utf-8")) / 4)) for key in ("prompt", "completion"))
            keep, action = fit_lengths(p, c, args.max_len, args.overlength)
            clipped += action == "clipped"
            dropped += action == "dropped"
            if keep:
                lengths.append(keep + c)
                tokens += keep + c
        batches = epoch_batches(lengths, args.max_tokens_per_batch, args.batch_size, args.seed, 0)
        result[name] = dict(rows=len(rows), kept_rows=len(lengths), clipped_rows=clipped,
                            dropped_rows=dropped, estimated_tokens=tokens,
                            estimated_optimizer_steps_per_epoch=len(batches),
                            estimated_micro_batches_per_epoch=sum(map(len, batches)))
    return result


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_path(root, name):
    path = (root / name).resolve()
    if path.parent != (root / "checkpoints").resolve() or not path.is_dir():
        raise PackError("Invalid checkpoint reference")
    return path


def load_checkpoint(root):
    manifest_path = root / "checkpoint.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = checkpoint_path(root, manifest["latest"])
    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
    return path, state, manifest


def commit_checkpoint(root, state, writer, is_best=False):
    """Publish immutable payload first. Replace the pointer only after fsync."""
    directory = root / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    previous = load_checkpoint(root)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=directory))
    target = directory / f"step-{state['global_step']:08d}-{uuid.uuid4().hex}"
    try:
        writer(temporary)
        atomic_json(temporary / "state.json", state)
        for file in temporary.rglob("*"):
            if file.is_file():
                with file.open("rb") as stream:
                    os.fsync(stream.fileno())
        sync_directory(temporary)
        os.replace(temporary, target)
        sync_directory(directory)
        latest = str(target.relative_to(root))
        manifest = dict(latest=latest, best=latest if is_best else (previous[2]["best"] if previous else None))
        atomic_json(root / "checkpoint.json", manifest)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    # Keep latest and best. A failed save never removes the prior checkpoint.
    for old in directory.glob("step-*"):
        if str(old.relative_to(root)) not in (manifest["latest"], manifest["best"]):
            shutil.rmtree(old)
    return target


def committed_adapters(root):
    """The kept adapters of the run's committed state, keyed by path: this trainer's epoch_adapters and the RL trainer's adapters."""
    loaded = load_checkpoint(root)
    return {entry["path"]: entry for entry in [*loaded[1].get("epoch_adapters", []), *loaded[1].get("adapters", [])]} if loaded else {}


def commit_epoch_adapter(root, epoch, source, revision):
    """Keep this epoch's adapter for good: the two adapter files and the base revision, no optimizer state.

    Returns the state entry. Call it inside the checkpoint writer, so the committed state never names a missing
    adapter. A published copy never changes: an identical repeat is a no-op, a different one for a committed epoch is
    refused, and the leftover of a killed save is swapped out so that a failed swap keeps the old copy."""
    directory = root / "adapters"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=directory))
    target = directory / f"epoch-{epoch}"
    try:
        for name in ADAPTER_FILES:
            shutil.copyfile(source / name, temporary / name)
            with (temporary / name).open("rb") as stream:
                os.fsync(stream.fileno())
        atomic_json(temporary / "revision.json", dict(revision=revision))
        entry = dict(epoch=epoch, path=str(target.relative_to(root)),
                     sha256={name: sha256(temporary / name) for name in EPOCH_FILES})
        sync_directory(temporary)
        if target.exists():
            published = {file.name: sha256(file) if file.is_file() else None for file in target.iterdir()}
            if published == entry["sha256"]:
                return entry
            if entry["path"] in committed_adapters(root):
                raise PackError(f"Epoch adapter {epoch} is committed and cannot change")
            # A kill after the publish but before the checkpoint commit repeats the epoch. Swap that leftover out.
            leftover = directory / f".pending-replaced-{uuid.uuid4().hex}"
            os.rename(target, leftover)
            try:
                os.replace(temporary, target)
            except BaseException:
                os.rename(leftover, target)
                raise
            shutil.rmtree(leftover)
        else:
            os.replace(temporary, target)
        sync_directory(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return entry


def runtime_versions():
    from importlib.metadata import PackageNotFoundError, version
    result = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "peft", "tokenizers", "safetensors", "numpy", "triton", "accelerate",
                 "flash-linear-attention", "fla-core", "causal-conv1d"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def objective_sha256():
    """Fingerprint of the code that decides what is trained: END and the source text of OBJECTIVE_FUNCTIONS (row clipping and
    preparation, batching, padding, the loss, its normalisation and the optimizer step). Edits elsewhere in this file do not
    change it, comments and docstrings inside those functions do."""
    lines = Path(__file__).read_text(encoding="utf-8").split("\n")
    nodes = {node.name: node for node in ast.parse("\n".join(lines)).body if isinstance(node, ast.FunctionDef)}
    parts = [END, *("\n".join(lines[nodes[name].lineno - 1:nodes[name].end_lineno]) for name in OBJECTIVE_FUNCTIONS)]
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def lengths_sha256(rows):
    """Fingerprint of the kept rows' token counts (all tokens, supervised tokens), in order. Another tokenizer build moves them."""
    counts = [[len(row["input_ids"]), sum(label != -100 for label in row["labels"])] for row in rows]
    return hashlib.sha256(json.dumps(counts).encode("utf-8")).hexdigest()


def make_identity(args, revision, dtype, versions, fingerprints):
    """What a resume must match. The other runtime versions live in the state and only warn."""
    skipped = ("out", "train", "val", "dry_run", "stop_after_seconds", "log_steps", "save_steps", "allow_slow_kernels")
    return dict(options={key: value for key, value in vars(args).items() if key not in skipped},
                revision=revision, train_sha256=sha256(args.train), val_sha256=sha256(args.val), dtype=str(dtype),
                versions={name: versions[name] for name in ("torch", "transformers", "peft")}, **fingerprints)


def check_resume(saved, identity, session):
    """Refuse a resume that would mix training objectives, and say which part differs. A checkpoint that did not record a
    fingerprint counts as different. Differences from the first session that cannot change the objective only warn."""
    before = saved["identity"]
    for key, what in (("objective_sha256", "The code that defines the training objective"),
                      ("train_lengths_sha256", "The tokenized Train lengths"),
                      ("val_lengths_sha256", "The tokenized Val lengths")):
        if before.get(key) != identity[key]:
            raise PackError(f"{what} differ from the first session's. Use a new output directory")
    if before != identity:
        raise PackError("Resume identity changed. Use a new output directory")
    first = saved["first_session"]
    changed = [name for name, value in session["versions"].items() if value != first["versions"].get(name)]
    if changed:
        print(f"WARNING: These runtime versions differ from the first session's: {', '.join(changed)}. "
              "Resume keeps batch order. Numerical replay is UNTESTED.", flush=True)
    for key, what in (("git_revision", "The code revision"), ("trainer_sha256", "The trainer source")):
        if session[key] != first[key]:
            print(f"WARNING: {what} differs from the first session's. The objective fingerprint matches, so the run continues.", flush=True)


def with_session(state, session):
    """Record this session in the state. The first session stays as it was, every session is appended, and the top-level
    fields describe this one."""
    first = state.get("first_session") or {key: session[key] for key in ("git_revision", "versions", "trainer_sha256")}
    return {**state, **{key: session[key] for key in ("git_revision", "versions", "kernels", "command")},
            "first_session": first, "sessions": [*state.get("sessions", []), session]}


def source_row_id(row):
    """The id of the row a training row came from. A3 ids are '<row id>#<sample_idx>'."""
    return str(row["id"]).rpartition("#")[0] or str(row["id"])


def check_disjoint(train_rows, val_rows):
    if {source_row_id(row) for row in train_rows} & {source_row_id(row) for row in val_rows}:
        raise PackError("Train and Val IDs overlap")


def kernel_status(allow_slow=False):
    """Run before importing the model: Transformers selects packages at import."""
    import importlib
    import torch
    capability = torch.cuda.get_device_capability(0)
    status, missing = {}, []
    for name, module, function in (("fla", "fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
                                   ("causal_conv1d", "causal_conv1d", "causal_conv1d_fn")):
        try:
            operation = getattr(importlib.import_module(module), function)
            # Probe forward and backward before the model binds its fallbacks.
            dtype = getattr(torch, precision_for_capability(torch.cuda.get_device_capability(0)))
            if name == "fla":
                if torch.cuda.get_device_capability(0) < (8, 0):
                    raise PackError("Use torch DeltaNet on Turing")
                q, k, v = [torch.randn(1, 16, 2, 32, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
                g = -torch.ones(1, 16, 2, device="cuda", dtype=torch.float32)
                beta = torch.full((1, 16, 2), 0.5, device="cuda", dtype=dtype)
                result, _ = operation(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
                result.float().sum().backward()
                finite = all(tensor.grad is not None and torch.isfinite(tensor.grad).all().item() for tensor in (q, k, v))
            else:
                values = torch.randn(1, 8, 16, device="cuda", dtype=dtype, requires_grad=True)
                weights = torch.randn(8, 4, device="cuda", dtype=dtype, requires_grad=True)
                result = operation(values, weights, activation="silu")
                result.float().sum().backward()
                finite = all(tensor.grad is not None and torch.isfinite(tensor.grad).all().item() for tensor in (values, weights))
            torch.cuda.synchronize()
            if not finite or not torch.isfinite(result).all().item():
                raise PackError("Kernel probe was non-finite")
            status[name] = "small forward/backward probe passed; full model UNTESTED"
        except Exception:
            # Leave no broken optional package for Transformers' import-time probe.
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    del sys.modules[key]
            sys.modules[name] = None
            status[name] = "torch fallback"
            missing.append(name)
            print(f"WARNING: {name} is unavailable. Use the slower Transformers torch implementation.", flush=True)
    # Turing has no fast route. From compute capability 8.0 a silent fallback would waste an A100 session.
    if missing and capability >= (8, 0) and not allow_slow:
        raise PackError(f"Fast kernels are missing or failed their probe ({', '.join(missing)}) on compute capability "
                        f"{capability[0]}.{capability[1]}. Install them or pass --allow-slow-kernels")
    return status


def local_revision(path):
    """The pin of a base model that is a local directory, which has no commit: a hash of its config, its weights index and
    the size of every weight file."""
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        if (path / name).is_file():
            digest.update(f"{name}={sha256(path / name)}\n".encode())
    for file in sorted(path.glob("*.safetensors")):
        digest.update(f"{file.name}={file.stat().st_size}\n".encode())
    return f"local-{digest.hexdigest()}"


def load_text_model(model_name, dtype, revision=None):
    """Returns the model, its config and the base revision it was loaded at: a commit for a Hub model, local_revision() for a
    local directory. A revision that is passed must match; Transformers itself ignores revisions of local directories."""
    from transformers import AutoConfig, Qwen3_5ForCausalLM
    local = Path(model_name).is_dir()
    pin = local_revision(Path(model_name)) if local else None
    if local and revision not in (None, pin):
        raise PackError("The local base model changed since this run started")
    hub = None if local else revision
    config = AutoConfig.from_pretrained(model_name, revision=hub, trust_remote_code=False)
    if config.model_type != "qwen3_5":
        raise PackError("This pack requires the dense Qwen3.5 model")
    if not local:
        hub = hub or getattr(config, "_commit_hash", None)
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        model_name, config=config.text_config, revision=hub, dtype=dtype,
        attn_implementation="sdpa", output_loading_info=True, trust_remote_code=False,
    )
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise PackError("Base text weights did not load completely")
    model.config.use_cache = False
    return model, config, pin or hub


def adapter_targets(model, linear_type):
    targets = [name for name, layer in model.named_modules()
               if name.startswith("model.layers.") and isinstance(layer, linear_type)]
    missing = PROJECTIONS - {name.rsplit(".", 1)[-1] for name in targets}
    if missing or not targets:
        raise PackError("The language-model projection coverage is incomplete")
    return targets


def collate(rows, pad_id, device, torch):
    longest = max(len(row["input_ids"]) for row in rows)
    return {key: torch.tensor([row[key] + [fill] * (longest - len(row[key])) for row in rows], device=device)
            for key, fill in (("input_ids", pad_id), ("labels", -100))} | {
                "attention_mask": torch.tensor([[1] * len(row["input_ids"]) + [0] * (longest - len(row["input_ids"]))
                                                for row in rows], device=device)}


def completion_loss(model, batch, torch, training):
    """Project only supervised positions, in recomputed 256-token chunks."""
    from torch.utils.checkpoint import checkpoint
    base = model.get_base_model()
    hidden = base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False, return_dict=True).last_hidden_state
    labels = batch["labels"][:, 1:]
    valid = labels != -100
    hidden, labels = hidden[:, :-1][valid], labels[valid]
    loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
    correct = torch.zeros((), device=hidden.device, dtype=torch.int64)

    def head_loss(states, targets):
        logits = base.lm_head(states).float()
        return torch.nn.functional.cross_entropy(logits, targets, reduction="sum")

    for start in range(0, len(labels), 256):
        states, targets = hidden[start:start + 256], labels[start:start + 256]
        if training:
            loss = loss + checkpoint(head_loss, states, targets, use_reentrant=False)
        else:
            logits = base.lm_head(states).float()
            loss += torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
            correct += (logits.argmax(-1) == targets).sum()
    return loss, correct, len(labels)


def accumulate_update(train, micro_batches, forward, scaler):
    """Backward every micro-batch of one update. Each summed loss is divided by all completion tokens in the update."""
    denominator = sum(sum(label != -100 for label in train[i]["labels"]) for batch in micro_batches for i in batch)
    step_loss = 0.0
    for indices in micro_batches:
        loss = forward([train[i] for i in indices])
        value = loss.detach().item()
        if not math.isfinite(value):
            raise PackError("Non-finite training loss. Last checkpoint is intact")
        step_loss += value
        scaler.scale(loss / denominator).backward()
    return step_loss, denominator


def step_optimizer(parameters, optimizer, scheduler, scaler, torch):
    """Unscale, clip at norm 1 and step. False: an fp16 overflow skipped the step, so retry the batch."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not torch.isfinite(norm).item():
        # A finite gradient norm can overflow too. Never step in that case.
        overflowed = any(not torch.isfinite(p.grad).all().item() for p in parameters if p.grad is not None)
        if scaler.is_enabled() and scaler.get_scale() > 1 and overflowed:
            scaler.step(optimizer)
            scaler.update()
            return False
        raise PackError("Non-finite gradients. Last checkpoint is intact; reduce lr or use bf16")
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return True


def run_training(args, raw_train, raw_val, started):
    # Disable optional Hub replacements. Local FLA/conv packages still work.
    os.environ["USE_HUB_KERNELS"] = "0"
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise PackError("CUDA is required for training; use --dry-run on CPU")
    dtype = getattr(torch, precision_for_capability(torch.cuda.get_device_capability(0)))
    kernels = kernel_status(args.allow_slow_kernels)
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    existing = load_checkpoint(args.out)
    revision = existing[1]["identity"]["revision"] if existing else None
    model, original_config, revision = load_text_model(args.model, dtype, revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=None if Path(args.model).is_dir() else revision,
                                              trust_remote_code=False)
    train, train_counts = prepare_rows(raw_train, tokenizer, args.max_len, args.overlength)
    val, val_counts = prepare_rows(raw_val, tokenizer, args.max_len, args.overlength)
    print(json.dumps(dict(train=train_counts, val=val_counts)), flush=True)
    if not train or not val:
        raise PackError("No Train or Val rows remain at this max_len")
    check_disjoint(raw_train, raw_val)
    lengths = [len(row["input_ids"]) for row in train]
    versions = runtime_versions()
    fingerprints = dict(objective_sha256=objective_sha256(), train_lengths_sha256=lengths_sha256(train),
                        val_lengths_sha256=lengths_sha256(val))
    identity = make_identity(args, revision, dtype, versions, fingerprints)
    try:
        git_revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
    session = dict(git_revision=git_revision, versions=versions, kernels=kernels, command=sys.argv,
                   trainer_sha256=sha256(Path(__file__)))
    targets = adapter_targets(model, torch.nn.Linear)
    if existing:
        check_resume(existing[1], identity, session)
        model = PeftModel.from_pretrained(model, existing[0], is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                               bias="none", target_modules=targets, task_type="CAUSAL_LM"))
    actual = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    if {name.removeprefix("base_model.model.") for name in actual} != set(targets):
        raise PackError("Actual LoRA modules differ from the requested modules")
    print(json.dumps(dict(adapter_modules=actual, train=train_counts, val=val_counts, dtype=str(dtype))), flush=True)
    model.to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise PackError("No trainable adapter parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(math.ceil(total_steps * args.warmup_ratio), total_steps - 1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    fresh = dict(identity=identity, epoch=0, next_step=0, global_step=0, history=[],
                 best_val_loss=None, epoch_adapters=[], train=train_counts, val=val_counts, adapter_modules=actual,
                 trainable_parameters=sum(p.numel() for p in parameters), tokens=0,
                 completion_tokens=0, training_seconds=0.0, epoch_loss_sum=0.0, epoch_tokens=0,
                 gpu=dict(name=torch.cuda.get_device_name(0), memory_bytes=torch.cuda.get_device_properties(0).total_memory),
                 status="running")
    if existing and existing[1]["kernels"] != kernels:
        print("WARNING: The kernel path changed. Resume keeps batch order. Numerical replay is UNTESTED.", flush=True)
    state = with_session(existing[1] if existing else fresh, session)
    if existing:
        if not 0 <= state["epoch"] <= args.epochs or not 0 <= state["next_step"] <= steps_per_epoch:
            raise PackError("Invalid resume cursor")
        # Only load checkpoints produced locally by this pack. torch state uses pickle.
        saved = torch.load(existing[0] / "training.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
    stopped = [False]
    def on_signal(signum, frame):
        stopped[0] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if not isinstance(pad_id, int):
        raise PackError("Tokenizer has no padding or EOS token")
    torch.cuda.reset_peak_memory_stats()

    def forward(rows):
        batch = collate(rows, pad_id, "cuda", torch)
        with torch.autocast("cuda", dtype=dtype):
            return completion_loss(model, batch, torch, True)[0]

    def save(best=False, epoch_end=False):
        state["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        def writer(path):
            model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
            if epoch_end:
                entry = commit_epoch_adapter(args.out, state["epoch"], path, state["identity"]["revision"])
                state["epoch_adapters"] = [*state["epoch_adapters"], entry]
            tokenizer.save_pretrained(path)
            original_config.save_pretrained(path / "base_config")
            torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all()), path / "training.pt")
        commit_checkpoint(args.out, state, writer, best)
        # A derived JSONL log is rebuilt from the committed epoch history on resume.
        log_path = args.out / "epochs.jsonl"
        with log_path.open("w", encoding="utf-8") as stream:
            for item in state["history"]:
                stream.write(json.dumps(item, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(args.out / "summary.json", state)

    save()
    while state["epoch"] < args.epochs:
        plan = epoch_batches(lengths, args.max_tokens_per_batch, args.batch_size, args.seed, state["epoch"])
        model.train()
        while state["next_step"] < len(plan):
            if stopped[0] or time.monotonic() - started >= args.stop_after_seconds:
                state["status"] = "stopped_signal" if stopped[0] else "stopped_time_limit"
                save()
                return
            micro_batches = plan[state["next_step"]]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_started = time.monotonic()
            step_loss, denominator = accumulate_update(train, micro_batches, forward, scaler)
            if not step_optimizer(parameters, optimizer, scheduler, scaler, torch):
                print(f"WARNING: fp16 overflow at step={state['global_step']}. Reduce the loss scale and retry this batch.", flush=True)
                continue
            torch.cuda.synchronize()
            seconds = time.monotonic() - step_started
            tokens = sum(lengths[i] for batch in micro_batches for i in batch)
            state = advance_step(state, len(plan))
            state.update(tokens=state["tokens"] + tokens, completion_tokens=state["completion_tokens"] + denominator,
                         training_seconds=state["training_seconds"] + seconds,
                         epoch_loss_sum=state["epoch_loss_sum"] + step_loss, epoch_tokens=state["epoch_tokens"] + denominator)
            if state["global_step"] % args.log_steps == 0:
                print(f"step={state['global_step']} loss={step_loss / denominator:.6f} tokens/s={tokens / seconds:.1f} "
                      f"peak_memory_bytes={torch.cuda.max_memory_allocated()}", flush=True)
            if state["global_step"] % args.save_steps == 0:
                save()
        # A signal/time limit can defer validation. Resume then starts at this boundary.
        if stopped[0] or time.monotonic() - started >= args.stop_after_seconds:
            state["status"] = "stopped_signal" if stopped[0] else "stopped_time_limit"
            save()
            return
        model.eval()
        val_loss, val_correct, val_tokens = 0.0, 0, 0
        val_lengths = [len(row["input_ids"]) for row in val]
        with torch.no_grad():
            for indices in token_batches(sorted(range(len(val)), key=val_lengths.__getitem__), val_lengths, args.max_tokens_per_batch):
                if stopped[0] or time.monotonic() - started >= args.stop_after_seconds:
                    state["status"] = "stopped_signal" if stopped[0] else "stopped_time_limit"
                    save()
                    return
                with torch.autocast("cuda", dtype=dtype):
                    loss, correct, count = completion_loss(model, collate([val[i] for i in indices], pad_id, "cuda", torch), torch, False)
                val_loss += loss.item()
                val_correct += correct.item()
                val_tokens += count
        val_loss /= val_tokens
        if not math.isfinite(val_loss):
            raise PackError("Non-finite validation loss. Last checkpoint is intact")
        metrics = dict(epoch=state["epoch"] + 1, step=state["global_step"], val_loss=val_loss,
                       val_token_accuracy=val_correct / val_tokens, val_completion_tokens=val_tokens,
                       train_loss=state["epoch_loss_sum"] / state["epoch_tokens"])
        best = is_best(val_loss, state["best_val_loss"])
        state = finish_epoch(state, len(plan), metrics)
        state.update(best_val_loss=val_loss if best else state["best_val_loss"], epoch_loss_sum=0.0, epoch_tokens=0,
                     status="complete" if state["epoch"] == args.epochs else "running")
        save(best, epoch_end=True)
        print(json.dumps(metrics), flush=True)


@contextlib.contextmanager
def run_lock(out):
    """Hold <out>/run.lock exclusively for the length of a run. A held lock means another trainer or an export is using the
    run. Once it is ours, the unfinished payloads of a killed save are removed."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PackError("Another trainer or an export holds this run. Wait for it to stop") from None
        for pending in out.glob("*/.pending-*"):
            if pending.is_dir():
                shutil.rmtree(pending)
        yield


def main(argv=None):
    started = time.monotonic()
    args = config_from_args(argv)
    train, val = read_rows(args.train, "train"), read_rows(args.val, "val")
    if args.dry_run:
        print(json.dumps(dry_plan(args, train, val), indent=2))
        return
    with run_lock(args.out):
        run_training(args, train, val, started)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print_failure("SFT", error)
        sys.exit(1)
