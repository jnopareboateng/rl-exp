#!/usr/bin/env python3
"""Merge a text-only Qwen3.5 adapter into the original conditional model.

CPU tests cannot prove GPU, kernel or memory behaviour. Real weight merging and
vLLM loading are UNTESTED here. Merge runs on CPU in bf16 to leave GPU RAM free.
The original vision weights and conditional-model config remain in the export.
"""

import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from sft_lora import (PackError, atomic_json, checkpoint_path, committed_adapters, local_revision, print_failure, sha256,
                      sync_directory)

EPOCH_NAME = re.compile(r"epoch-(\d+)")
# Kept adapters (the SFT trainer's epoch-N, the RL trainer's step-N) must say which base revision they were trained on.
PINNED_NAME = re.compile(r"(?:epoch|step)-\d+")
# What the two trainers put in a run, including their unfinished payloads. Any other directory is a plain adapter.
RUN_PATHS = {"adapters": re.compile(r"(?:epoch|step)-\d+|\.pending-.*"), "checkpoints": re.compile(r"step-.*|\.pending-.*")}


def owning_run(adapter):
    """The run directory that owns <run>/adapters/{epoch,step}-N or <run>/checkpoints/step-*, else None."""
    adapter = adapter.resolve()
    root, kind = adapter.parent.parent, RUN_PATHS.get(adapter.parent.name)
    inside = kind is not None and kind.fullmatch(adapter.name) and (root / "checkpoint.json").exists()
    return root if inside else None


def resolve_adapter(path):
    """A run root selects its committed best. A path inside a run must be committed: a checkpoint is the run's latest
    or best, a kept adapter is in its committed list and still matches the hashes recorded there. Any other adapter
    path works as is."""
    manifest = path / "checkpoint.json"
    if manifest.exists():
        best = json.loads(manifest.read_text(encoding="utf-8")).get("best")
        if best is None:
            raise PackError("This run has no best checkpoint (SFT: no complete validation epoch yet; RL never marks one). "
                            "Pass adapters/epoch-N, adapters/step-N or a checkpoint directory")
        return checkpoint_path(path, best)
    root = owning_run(path)
    if root is not None:
        path = path.resolve()
        if path.parent.name == "checkpoints":
            pointer = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
            if path not in {checkpoint_path(root, pointer[key]) for key in ("latest", "best") if pointer.get(key)}:
                raise PackError("This checkpoint is not the run's committed latest or best")
        else:
            entry = committed_adapters(root).get(f"adapters/{path.name}")
            if entry is None:
                raise PackError("This kept adapter is not in its run's committed list. A killed save can leave one behind")
            if {file.name: sha256(file) if file.is_file() else None for file in path.iterdir()} != entry["sha256"]:
                raise PackError("This kept adapter differs from the hashes committed with its run")
    if not (path / "adapter_config.json").is_file() or not (path / "adapter_model.safetensors").is_file():
        raise PackError("Adapter safetensors or config is missing")
    return path


def adapter_revision(adapter, requested=None):
    """The base revision an adapter was trained on. An explicit `requested` one must agree with it.

    A checkpoint records it in state.json, a kept adapter in revision.json, any other adapter in its config or nowhere. A null
    counts as nothing recorded. A kept adapter that records none is refused unless the revision is passed."""
    state, pin = adapter / "state.json", adapter / "revision.json"
    if state.exists():
        recorded = json.loads(state.read_text(encoding="utf-8"))["identity"]["revision"]
    elif pin.exists():
        recorded = json.loads(pin.read_text(encoding="utf-8"))["revision"]
    else:
        recorded = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8")).get("revision")
    if recorded is None and requested is None and PINNED_NAME.fullmatch(adapter.name):
        raise PackError("This kept adapter records no base revision. Pass --revision")
    if recorded is not None and requested is not None and recorded != requested:
        raise PackError("--revision differs from the base revision this adapter was trained on")
    return recorded if recorded is not None else requested


def verify_export(output, adapter, base_name, revision=None):
    report = json.loads((output / "merge_report.json").read_text(encoding="utf-8"))
    adapter_hashes = {name: sha256(adapter / name) for name in ("adapter_model.safetensors", "adapter_config.json")}
    if report["base_model"] != base_name or report["adapter_sha256"] != adapter_hashes:
        raise PackError("Existing export belongs to different weights. Choose a new output directory")
    revision = adapter_revision(adapter, revision)
    if revision is not None and revision != report["revision"]:
        raise PackError("Existing export has a different base revision")
    for name, digest in report["file_sha256"].items():
        file = output / name
        if file.parent != output or not file.is_file() or sha256(file) != digest:
            raise PackError("Existing export is incomplete or changed. Choose a new output directory")
    if (not {"config.json", "tokenizer_config.json"} <= report["file_sha256"].keys()
            or not any(name.endswith(".safetensors") for name in report["file_sha256"])):
        raise PackError("Existing export has no model weights or tokenizer config")
    return report


def adapter_epoch(adapter, state):
    """Epoch an adapter was saved at: from a checkpoint's state.json, else from an adapters/epoch-N directory name."""
    if state:
        return state.get("epoch")
    match = EPOCH_NAME.fullmatch(adapter.name)
    return int(match.group(1)) if match else None


def check_coverage(actual, state):
    """The adapter modules that were merged must be the ones the run trained. The SFT trainer records their names, the RL
    trainer their count."""
    trained = state.get("adapter_modules") if state else None
    if isinstance(trained, list) and set(actual) != set(trained) or isinstance(trained, int) and len(actual) != trained:
        raise PackError("The adapter module coverage changed during merge")


def print_sizes(report):
    sizes = {key: report[key] for key in ("parameters", "stored_parameters", "safetensors_bytes", "files")}
    if any(type(value) is not int or value < 0 for value in sizes.values()):
        raise PackError("Invalid export size report")
    print(json.dumps(sizes), flush=True)


def merge(base_name, adapter, output, revision=None):
    adapter = resolve_adapter(adapter)
    revision = adapter_revision(adapter, revision)
    local = Path(base_name).is_dir()        # a local base has no commit: its pin is a hash of its files
    if local:
        pin = local_revision(Path(base_name))
        if revision not in (None, pin):
            raise PackError("The local base model differs from the one this adapter was trained on")
        revision = pin
    if output.exists():
        print_sizes(verify_export(output, adapter, base_name, revision))
        return
    os.environ["USE_HUB_KERNELS"] = "0"
    import torch
    from peft import PeftConfig, PeftModel
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
    peft_config = PeftConfig.from_pretrained(adapter)
    if peft_config.base_model_name_or_path != base_name:
        raise PackError("Adapter and base model names differ")
    state_path = adapter / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    config = AutoConfig.from_pretrained(base_name, revision=None if local else revision, trust_remote_code=False)
    if config.model_type != "qwen3_5":
        raise PackError("This pack requires the dense Qwen3.5 model")
    revision = revision or getattr(config, "_commit_hash", None)
    hub = None if local else revision        # Transformers ignores a revision for a local directory
    full, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_name, config=config, revision=hub, dtype=torch.bfloat16,
        attn_implementation="sdpa", output_loading_info=True, trust_remote_code=False,
    )
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise PackError("Base weights did not load completely")
    # Share the loaded decoder/head. Do not allocate or reload a second 4B model.
    with torch.device("meta"):
        text = Qwen3_5ForCausalLM(config.text_config)
    text.model, text.lm_head = full.model.language_model, full.lm_head
    adapted = PeftModel.from_pretrained(text, adapter, is_trainable=False)
    check_coverage([name for name, module in adapted.named_modules() if hasattr(module, "lora_A")], state)
    merged = adapted.merge_and_unload(safe_merge=True)
    full.model.language_model, full.lm_head = merged.model, merged.lm_head
    full.to(dtype=torch.bfloat16)
    full.tie_weights()
    parameters = sum(parameter.numel() for parameter in full.parameters())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".merge-", dir=output.parent))
    try:
        full.save_pretrained(temporary, safe_serialization=True, max_shard_size="4GB", save_original_format=False)
        tokenizer = AutoTokenizer.from_pretrained(base_name, revision=hub, trust_remote_code=False)
        tokenizer.save_pretrained(temporary)
        reloaded = AutoConfig.from_pretrained(temporary, local_files_only=True, trust_remote_code=False)
        if reloaded.model_type != config.model_type or reloaded.text_config.vocab_size != config.text_config.vocab_size:
            raise PackError("Merged config does not match the base config")
        with torch.device("meta"):
            probe = Qwen3_5ForConditionalGeneration(reloaded)
        if sum(p.numel() for p in probe.parameters()) != parameters:
            raise PackError("Reloaded parameter count changed")
        stored, stored_parameters = {}, 0
        for file in sorted(temporary.glob("*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as archive:
                for name in archive.keys():
                    tensor = archive.get_slice(name)
                    shape = tuple(tensor.get_shape())
                    if name in stored or tensor.get_dtype() != "BF16":
                        raise PackError("Merged weights have duplicate names or a wrong dtype")
                    stored[name] = shape
                    stored_parameters += math.prod(shape)
        if stored_parameters != parameters:
            raise PackError("Saved parameter count does not match the model")
        expected = {name: tuple(value.shape) for name, value in full.state_dict().items()}
        # Safetensors saves tied input/output embeddings only once.
        for target, source in full.all_tied_weights_keys.items():
            if target not in stored and source in stored:
                expected.pop(target, None)
        if stored != expected:
            raise PackError("Saved weight names or shapes do not match the conditional model")
        sizes = dict(parameters=parameters, stored_parameters=stored_parameters,
                     safetensors_bytes=sum(file.stat().st_size for file in temporary.glob("*.safetensors")),
                     files=sum(1 for file in temporary.iterdir() if file.is_file()),
                     vllm_load="UNTESTED", epoch=adapter_epoch(adapter, state))
        if state and "epoch" in state:
            sizes["best_epoch"] = state["epoch"]
            sizes["best_step"] = state["global_step"]
        sizes.update(base_model=base_name, revision=revision,
                     adapter_sha256={name: sha256(adapter / name) for name in ("adapter_model.safetensors", "adapter_config.json")},
                     file_sha256={file.name: sha256(file) for file in temporary.iterdir() if file.is_file()})
        atomic_json(temporary / "merge_report.json", sizes)
        for file in temporary.iterdir():
            if file.is_file():
                with file.open("rb") as stream:
                    os.fsync(stream.fileno())
        sync_directory(temporary)
        os.replace(temporary, output)
        sync_directory(output.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print_sizes(sizes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", help="base model revision; it must agree with the one the adapter recorded")
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    root = args.adapter if (args.adapter / "checkpoint.json").exists() else owning_run(args.adapter)
    with ExitStack() as stack:
        locks = [(args.out.parent / ".merge.lock", fcntl.LOCK_EX, "Another merge is writing to this directory")]
        if root is not None:
            # The trainer holds this lock for the whole run, so a held lock means the files are still changing.
            locks.append((root / "run.lock", fcntl.LOCK_SH, "A trainer holds this run. Export after it stops"))
        for path, mode, refusal in locks:
            lock = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(lock, mode | fcntl.LOCK_NB)
            except BlockingIOError:
                raise PackError(refusal) from None
        for pending in args.out.parent.glob(".merge-*"):
            if pending.is_dir():
                shutil.rmtree(pending)
        merge(args.base, args.adapter, args.out, args.revision)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print_failure("Merge", error)
        sys.exit(1)
