#!/usr/bin/env python3
"""Answer-only GRPO (arm A5) from the A2 LoRA adapter: one process, HF generation, no vLLM.

Rollouts come from `model.generate` on the PEFT model being trained, so there is no engine or weight-sync mismatch. The LM head is an
fp32 copy in generation and in training. Loss: CISPO with prompt-level aggregation at importance ratio 1 (one update per rollout
batch), i.e. REINFORCE with a group baseline; the bookkeeping is in rlexp.grpo. A sample that hit the length cap is a failure (reward 0,
no valid answer) that stays in its group. --reward-mode validity and --profile-mode none are the two controls of docs/plan.md section 2
(d); --reward-mode contrastive subtracts each answer's similarity to the other drugs of the step, floored at 0. See gpu/RL.md.

CPU tests cannot prove GPU, kernel, generation or memory behaviour. Everything that needs the model (HF generate with left padding on
the GatedDeltaNet layers, PEFT adapter loading, memory at 256 sequences, the fp32 head inside generate) is UNTESTED on the target
stack until `--gate parity` and `--gate smoke` have run on the GPU. Dry runs and the loop, data and score logic need no torch.
"""

import argparse
import contextlib
import copy
import fcntl
import functools
import gc
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
import uuid

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "src", HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from rlexp import grpo  # noqa: E402  numpy only; rlexp.arms, chem and reward (rdkit) are imported where they are used
import sft_lora as sft  # noqa: E402  the reviewed LoRA trainer: checkpoints, kernel probe, model loading, adapter targets

PackError = sft.PackError
END, THINK = "<|im_end|>", "</think>"
RANK, ALPHA = 32, 64                      # a fresh adapter, exactly as sft_lora.py makes it (alpha = 2 * rank)
REWARD_MODES, PROFILE_MODES = ("full", "validity", "contrastive"), ("real", "none")
PROFILE_FREE_USER = "Drug: unknown."      # the user turn of every prompt under --profile-mode none
CHUNK = 256                               # completion positions per fp32-head chunk
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
PIN_FILE = "revision.json"                # the base model revision an adapter was trained on, as gpu/sft_lora.py writes it
KEPT_FILES = (*ADAPTER_FILES, PIN_FILE)   # what adapters/step-N/ holds: gpu/merge_lora.py reads the pin to load the same base
# What a resume does not have to match: paths (the data hashes stand in for them), the stop point, and pure performance settings.
OPERATIONAL = {"out", "train", "val", "dry_run", "gate", "steps", "stop_after_seconds", "save_every", "gen_batch", "gen_token_budget",
               "max_tokens_per_batch", "allow_slow_kernels", "init_adapter"}
# The loop fields a checkpoint carries across a resume. The cursor is the whole schedule state.
LOOP_KEYS = ("step", "global_step", "cursor", "collapse_streak", "skipped_updates", "training_seconds",
             "first_update_delta_norm", "peak_memory_bytes", "metrics", "evals", "adapters")


# ---------------------------------------------------------------- configuration
def config_from_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", help="base model revision (a Hub commit) to load. Required with --init-adapter none, where no adapter records "
                                           "it (pass the 'revision' in the A2 adapter's revision.json); must agree with the init adapter's pin")
    parser.add_argument("--init-adapter", required=True,
                        help="PEFT adapter directory to continue (the A2 adapter), or 'none' for a fresh rank-32 LoRA (zero-RL ablation)")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--out", type=Path, default=Path(".agent-runs/rl_A5"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--prompts-per-step", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=10, help="steps between Val evals (0 turns them off)")
    parser.add_argument("--eval-k", type=int, default=8, help="Val samples per row; the endpoint (fv_tanimoto) is defined on 8")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--gen-batch", type=int, default=128, help="most sequences per generate call")
    parser.add_argument("--gen-token-budget", type=int, default=49152,
                        help="most prompt tokens (rows x longest prompt) per generate call. Prefill memory grows faster than this budget "
                             "(R1 ran out of memory at 81920 on a 40 GB card and ran cleanly at 49152); lower it on an out-of-memory error")
    parser.add_argument("--max-tokens-per-batch", type=int, default=8192, help="train micro-batch token budget, padding included")
    parser.add_argument("--min-groups", type=int, default=16, help="retained groups wanted per update; half of it, rounded up, is the skip threshold")
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default="full",
                        help="full: the reward of rlexp.reward. validity: 1.0 for a valid, closed, finished answer, else 0.0 (the control). "
                             "contrastive: the full reward minus the answer's similarity to the other drugs of the step, floored at 0")
    parser.add_argument("--profile-mode", choices=PROFILE_MODES, default="real",
                        help="none: every user turn becomes a fixed text, so the policy cannot use the profile (the control)")
    parser.add_argument("--stop-after-seconds", type=float, default=3600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-slow-kernels", action="store_true",
                        help="on compute capability 8.0+, run with the torch fallback when a fast kernel is missing")
    parser.add_argument("--gate", choices=("parity", "smoke"), help="run a GPU gate instead of training")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    validate_config(vars(args))
    return args


def validate_config(config):
    for key in ("steps", "prompts_per_step", "max_new_tokens", "eval_k", "save_every", "patience", "gen_batch", "gen_token_budget",
                "max_tokens_per_batch", "min_groups"):
        if type(config[key]) is not int or config[key] < 1:
            raise PackError(f"{key} must be a positive integer")
    if type(config["group_size"]) is not int or config["group_size"] < 2:
        raise PackError("group_size must be an integer of at least 2: a group needs reward variance")
    if type(config["eval_every"]) is not int or config["eval_every"] < 0:
        raise PackError("eval_every must be a non-negative integer")
    for key in ("lr", "temperature", "stop_after_seconds"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise PackError(f"{key} must be positive and finite")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2 ** 32:
        raise PackError("seed must be in [0, 2**32)")
    if config["reward_mode"] not in REWARD_MODES or config["profile_mode"] not in PROFILE_MODES:
        raise PackError(f"reward_mode must be one of {REWARD_MODES} and profile_mode one of {PROFILE_MODES}")
    if config["reward_mode"] == "contrastive" and config["prompts_per_step"] < 2:
        raise PackError("The contrastive reward compares drugs: prompts_per_step must be at least 2")
    if Path(config["train"]).resolve() == Path(config["val"]).resolve():
        raise PackError("Train and Val must be separate files")
    for key in ("train", "val"):
        refuse_test_file(Path(config[key]))
    source = init_source(config)
    if source is None and not config["revision"]:
        raise PackError("A zero-RL run (--init-adapter none) needs --revision: the base commit the SFT arms trained on, "
                        "the 'revision' in the A2 adapter's revision.json. Without it the Hub's current revision would load")
    if source is not None:
        check_adapter_config(source)


def refuse_test_file(path):
    if "test_" in path.name.lower():
        raise PackError("A test split is for the locked evaluation only: it cannot be used for RL")


def init_source(config):
    value = config["init_adapter"]
    return None if str(value) == "none" else Path(value)


def check_adapter_config(directory):
    """The loaded adapter's config: a LoRA with no dropout (training and generation must see the same policy)."""
    directory = Path(directory)
    missing = [name for name in ADAPTER_FILES if not (directory / name).is_file()]
    if missing:
        raise PackError(f"The adapter directory lacks {', '.join(missing)}")
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or type(config.get("r")) is not int or config["r"] < 1:
        raise PackError("The init adapter is not a LoRA with a rank")
    if config.get("lora_dropout", 0.0) != 0 or config.get("use_dora") or config.get("use_rslora") or config.get("bias", "none") != "none":
        raise PackError("The init adapter must be a plain LoRA with no dropout, DoRA, rsLoRA or bias")
    return config


# ---------------------------------------------------------------- data
def rl_prompt(prompt, profile_mode):
    """The prompt a row is trained and evaluated on: the dataset prompt with the direct-answer system turn and, under profile_mode "none",
    the user turn replaced by a fixed text. The objective hash covers this function and the two it calls."""
    from rlexp import arms
    text = arms.with_system(prompt, arms.DIRECT_SYSTEM)
    if profile_mode == "none":
        head, _, tail = arms.split_prompt(text)
        text = head + PROFILE_FREE_USER + tail
    return text


def load_rows(path, split, profile_mode="real"):
    """Processed rows of one split, each with its prompt in rl_prompt (see there). Refuses a test-split file, rows without a usable
    reference, duplicate ids and drugs whose views disagree on the reference (the lift and the contrastive reward take one per drug)."""
    path = Path(path)
    refuse_test_file(path)
    if profile_mode not in PROFILE_MODES:
        raise PackError(f"profile_mode must be one of {PROFILE_MODES}")
    from rlexp import reward
    rows, seen, references, conflicting = [], set(), {}, set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except ValueError:
                raise PackError(f"Invalid JSON at row {number}; data omitted") from None
            alts = row.get("ref_alt_smiles", []) if isinstance(row, dict) else None
            if (not isinstance(row, dict) or not isinstance(row.get("id"), (str, int)) or not isinstance(row.get("group"), str)
                    or not row["group"] or not isinstance(row.get("prompt"), str) or not isinstance(row.get("ref_smiles"), str)
                    or not row["ref_smiles"] or not isinstance(alts, (list, str)) or row.get("split", split) != split
                    or row["id"] in seen):
                raise PackError(f"Invalid RL row {number} (needs id, group, prompt, ref_smiles; split must be {split}); data omitted")
            try:
                row["rl_prompt"] = rl_prompt(row["prompt"], profile_mode)
                reward.score("", row)             # raises ValueError when the reference does not standardize
            except ValueError:
                raise PackError(f"Row {number} has no system turn or an unusable reference; data omitted") from None
            if not row["rl_prompt"].endswith("<think>\n"):
                raise PackError(f"Row {number}: the prompt must end with the generation prefix '<think>' and a newline")
            reference = (row["ref_smiles"], sorted(map(str, [alts] if isinstance(alts, str) else alts)))
            if references.setdefault(row["group"], reference) != reference:
                conflicting.add(row["group"])
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise PackError("The data file has no rows")
    if conflicting:
        raise PackError(f"{len(conflicting)} drug(s) have views with different references (ref_smiles or alternates); "
                        "the lift and the contrastive reward need one per drug. Data omitted")
    return rows


def load_data(args):
    """Train and Val rows for these options (the same profile mode for both), checked to be disjoint."""
    train, val = (load_rows(path, split, args.profile_mode) for path, split in ((args.train, "train"), (args.val, "val")))
    check_disjoint(train, val)
    return train, val


def check_disjoint(train, val):
    if {r["id"] for r in train} & {r["id"] for r in val}:
        raise PackError("Train and Val IDs overlap")
    if {r["group"] for r in train} & {r["group"] for r in val}:
        raise PackError("Train and Val share a drug")


def drug_table(rows):
    """Rows grouped by drug, drugs sorted by name: the index space of the schedule."""
    by = {}
    for row in rows:
        by.setdefault(row["group"], []).append(row)
    return [by[name] for name in sorted(by)]


def collapse_stop(args):
    """The collapse rule is off without the profile: every prompt is then the same, so one constant answer is the best policy."""
    return args.profile_mode == "real"


def dry_plan(args, train, val):
    drugs = drug_table(train)
    sizes = [len(rows) for rows in drugs]
    estimate = lambda rows: sorted(max(1, math.ceil(len(r["rl_prompt"].encode("utf-8")) / 4)) for r in rows)
    quantiles = lambda xs: dict(p50=xs[len(xs) // 2], p90=xs[min(len(xs) - 1, math.ceil(0.9 * len(xs)) - 1)], max=xs[-1])
    train_tokens, val_tokens = estimate(train), estimate(val)
    samples = args.prompts_per_step * args.group_size
    return dict(
        mode="dry_run", gate=args.gate, estimate="ceil(UTF-8 bytes / 4) per prompt; no tokenizer or model loaded",
        init_adapter=str(args.init_adapter), revision=args.revision, steps=args.steps, reward_mode=args.reward_mode, profile_mode=args.profile_mode,
        train=dict(rows=len(train), drugs=len(drugs), views_min=min(sizes), views_max=max(sizes), prompt_tokens=quantiles(train_tokens)),
        val=dict(rows=len(val), drugs=len({r["group"] for r in val}), prompt_tokens=quantiles(val_tokens)),
        schedule=dict(prompts_per_step=args.prompts_per_step, steps_per_pass=round(len(drugs) / args.prompts_per_step, 2),
                      passes=round(args.steps * args.prompts_per_step / len(drugs), 2)),
        step=dict(samples=samples, max_generated_tokens=samples * args.max_new_tokens, generate_calls=math.ceil(samples / args.gen_batch),
                  min_groups=args.min_groups, skip_below=grpo.skip_below(args.min_groups), max_prompts_with_refill=2 * args.prompts_per_step),
        train_micro_batch=dict(token_budget=args.max_tokens_per_batch, largest_sample_estimate=train_tokens[-1] + args.max_new_tokens,
                               fits=train_tokens[-1] + args.max_new_tokens <= args.max_tokens_per_batch),
        eval=dict(every=args.eval_every, k=args.eval_k, samples=len(val) * args.eval_k, adapters_saved_at="every eval, never pruned"),
        stop_rules=dict(patience_evals=args.patience, collapse_below=grpo.COLLAPSE_BELOW, collapse_steps=grpo.COLLAPSE_STEPS,
                        collapse_stop=collapse_stop(args), truncation_above=grpo.TRUNCATION_ABOVE,
                        stop_after_seconds=args.stop_after_seconds))


# ---------------------------------------------------------------- identity, state, logs
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_digest(source):
    if source is None:
        return "none"
    return hashlib.sha256("".join(file_sha256(Path(source) / name) for name in ADAPTER_FILES).encode()).hexdigest()


def adapter_pin(source):
    """The base revision an adapter directory records in revision.json, else None."""
    pin = None if source is None else Path(source) / PIN_FILE
    return json.loads(pin.read_text(encoding="utf-8"))["revision"] if pin is not None and pin.is_file() else None


def base_revision(existing, source, requested=None):
    """The base model revision to load: the checkpoint's when resuming, else --revision (which must agree with the init adapter's pin
    when it records one), else the pin (None: the Hub's current one)."""
    if existing:
        return existing[1]["identity"]["revision"]
    pin = adapter_pin(source)
    if requested and pin and requested != pin:
        raise PackError("--revision differs from the base revision the init adapter was trained on")
    return requested or pin


def runtime_versions():
    from importlib.metadata import PackageNotFoundError, version
    result = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "peft", "safetensors", "numpy", "rdkit", "accelerate", "flash-linear-attention", "fla-core",
                 "causal-conv1d"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


# The functions of THIS file that are part of the objective, by name (looked up here when the hash is taken): the prompt, what ends a rollout
# (stop_ids, trim_generated), the length rule and the validity reward (finished_score, score_completions), the contrastive assembly and the
# similarity it reads (training_rewards, apply_contrast, similarities, _fingerprint) and the rewards that reach the update (plan_for). The
# scorer's own code is in rlexp/reward.py and chem.py and the update maths in rlexp/grpo.py, hashed whole. Editing any of it, a comment
# inside a function included, refuses a resume.
OBJECTIVE_FUNCTIONS = ("rl_prompt", "stop_ids", "trim_generated", "finished_score", "score_completions", "training_rewards", "apply_contrast",
                       "similarities", "_fingerprint", "plan_for")


def objective_digest():
    """sha256 of everything that defines what the policy is shown and paid for: the scorer (rlexp/reward.py, rlexp/chem.py), the update
    maths (rlexp/grpo.py), the direct-answer system turn, the profile-free user turn, the source of the prompt transformation
    (arms.with_system, arms.split_prompt) and of the functions of this file in OBJECTIVE_FUNCTIONS. A resume under another value would
    mix two objectives in one run, so it is refused."""
    import inspect
    from rlexp import arms, chem, reward
    parts = [Path(module.__file__).read_bytes() for module in (reward, chem, grpo)]
    parts += [arms.DIRECT_SYSTEM.encode(), PROFILE_FREE_USER.encode()]
    parts += [inspect.getsource(function).encode() for function in (arms.with_system, arms.split_prompt)]
    parts += [inspect.getsource(globals()[name]).encode() for name in OBJECTIVE_FUNCTIONS]
    return hashlib.sha256("".join(hashlib.sha256(part).hexdigest() for part in parts).encode()).hexdigest()


def trainer_hashes():
    """sha256 of the two trainer files by name: recorded in the state, informational (a resume warns when they changed)."""
    return {Path(path).name: file_sha256(path) for path in (__file__, sft.__file__)}


def make_identity(args, revision, dtype, versions):
    """What a resume must match: options, model revision, Train/Val hashes, the init adapter's hash, the objective hash, dtype and the
    versions of torch, transformers, peft and of rdkit and numpy (a new RDKit can change standardisation or fingerprints, and with them the
    reward)."""
    return dict(options={key: value for key, value in vars(args).items() if key not in OPERATIONAL},
                revision=revision, train_sha256=file_sha256(args.train), val_sha256=file_sha256(args.val),
                init_adapter_sha256=adapter_digest(init_source(vars(args))), objective_sha256=objective_digest(), dtype=str(dtype),
                versions={name: versions[name] for name in ("torch", "transformers", "peft", "rdkit", "numpy")})


def check_resume(saved, identity, versions, trainer=None):
    """Refuse a resume whose identity differs, and say so first when the objective is what differs (a checkpoint from before the objective
    hash was recorded counts as different). Only warn about changed runtime versions, and about changed trainer files when `trainer`
    (name -> sha256) is given: those are recorded for the reader, not pinned."""
    if saved["identity"].get("objective_sha256") != identity["objective_sha256"]:
        raise PackError("The objective changed: use a new output directory (the objective is the reward, chemistry and update code and the "
                        "prompt text; a checkpoint that does not record it resumes only with the files that wrote it)")
    if saved["identity"] != identity:
        before = saved["identity"].get("versions") or {}
        drift = [name for name, value in identity["versions"].items() if before.get(name) != value]
        raise PackError("Resume identity changed" + (f" (versions: {', '.join(drift)})" if drift else "") + ". Use a new output directory")
    changed = [name for name, value in versions.items() if value != saved["versions"].get(name)]
    if changed:
        print(f"WARNING: These runtime versions changed since the checkpoint: {', '.join(changed)}. Numerical replay is UNTESTED.", flush=True)
    moved = [name for name, value in (trainer or {}).items() if value != (saved.get("trainer_sha256") or {}).get(name)]
    if moved:
        print(f"WARNING: The trainer source changed since the checkpoint ({', '.join(moved)}). The objective is unchanged; "
              "the update path may differ from the one that wrote it.", flush=True)


def new_state(identity, versions, **record):
    return dict(identity=identity, versions=versions, status="running", step=0, global_step=0, cursor=0, collapse_streak=0,
                skipped_updates=0, training_seconds=0.0, first_update_delta_norm=None, peak_memory_bytes=0,
                metrics=[], evals=[], adapters=[], **record)


def restore_state(saved, state):
    """Carry the loop fields of a checkpoint into a fresh state. A rule stop stays final; any other stop, or a completed run asked
    for more steps, continues."""
    for key in LOOP_KEYS:
        try:
            state[key] = saved[key]
        except KeyError:
            raise PackError(f"The checkpoint state lacks {key}") from None
    if type(state["cursor"]) is not int or state["cursor"] < 0 or state["step"] < 0 or len(state["metrics"]) != state["step"]:
        raise PackError("Invalid resume cursor")
    state["status"] = saved["status"] if saved["status"] in grpo.RULE_STOPS else "running"
    return state


def write_jsonl(path, rows):
    fd, temporary = tempfile.mkstemp(prefix=".jsonl-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_logs(out, state):
    """Rebuild the derived logs from the committed state, so a resume never keeps lines of steps it will repeat."""
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "metrics.jsonl", state["metrics"])
    write_jsonl(out / "evals.jsonl", state["evals"])
    sft.atomic_json(out / "summary.json", state)


def fingerprint(value):
    return hashlib.sha1(json.dumps(value, separators=(",", ":")).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- scoring
def conn_key(answer):
    """The connectivity key of a parsed answer, or None."""
    from rlexp import chem
    mol = chem.standardize(answer)
    return None if mol is None else chem.conn(mol)


def finished_score(text, row, finished):
    """rlexp.reward.score with the length rule: an answer that never stopped is a failure however well its parsed SMILES matches (a
    deployed system cannot use it), so it scores 0 with no valid answer. The one seam to metrics v3: there this body becomes
    `return reward.score(text, row, finished=finished)`."""
    from rlexp import reward
    scored = reward.score(text, row)
    if finished:
        return scored
    return dict(scored, valid=False, tanimoto=0.0, scaffold_match=False, conn_match=False, stereo_match=False, reward=0.0)


def score_completions(decode, completions, row, reward_mode="full"):
    """Score generated completions ({ids, finish}) of one prompt against its reference. A completion that hit the length cap is a failure
    (reward 0, no valid answer) that stays in its group and in the update, never masked. `reward` is what the update trains on: the full
    reward, under reward_mode "validity" 1.0 for a valid, closed, finished answer and 0.0 otherwise, and under "contrastive" None until
    apply_contrast sets it over the whole batch. `reward_full` is always the full one. `answer` and `key` (its connectivity key) are set
    for a valid answer only. `decode` turns token ids into text."""
    if reward_mode not in REWARD_MODES:
        raise PackError(f"reward_mode must be one of {REWARD_MODES}")
    samples = []
    for completion in completions:
        finished = completion["finish"] != "length"
        scored = finished_score(decode(completion["ids"]), row, finished)
        full = float(scored["reward"])
        binary = float(bool(scored["valid"] and scored["closed"] and finished))
        samples.append(dict(ids=completion["ids"], reward={"full": full, "validity": binary, "contrastive": None}[reward_mode], reward_full=full,
                            valid=bool(scored["valid"]), closed=bool(scored["closed"]), conn_match=bool(scored["conn_match"]),
                            scaffold_match=bool(scored["scaffold_match"]), tanimoto=float(scored["tanimoto"]),
                            n_tokens=len(completion["ids"]), truncated=not finished,
                            answer=scored["answer"] if scored["valid"] else None,
                            key=conn_key(scored["answer"]) if scored["valid"] else None))
    return samples


@functools.lru_cache(maxsize=1 << 16)
def _fingerprint(smiles):
    """The count fingerprint of the standardized SMILES that rlexp.chem.tanimoto compares (chem's own generator), or None."""
    from rlexp import chem
    mol = chem.standardize(smiles)
    return None if mol is None else chem._FP.GetCountFingerprint(mol)


@functools.lru_cache(maxsize=1 << 18)
def _similarity(a, b):
    from rdkit import DataStructs
    fa, fb = _fingerprint(a), _fingerprint(b)
    return 0.0 if fa is None or fb is None else DataStructs.TanimotoSimilarity(fa, fb)


def similarity(a, b):
    """Count-vector Tanimoto of two SMILES strings after standardization, equal to rlexp.chem.tanimoto; 0.0 if either does not
    standardize. Symmetric and cached."""
    return _similarity(*sorted((a, b)))


def similarities(answer, refs):
    """T(answer, ref) for every ref in one bulk call on cached fingerprints (a reference is fingerprinted once, ever); 0.0 for a string
    that does not standardize."""
    from rdkit import DataStructs
    fa, fps = _fingerprint(answer), [_fingerprint(r) for r in refs]
    out = [0.0] * len(refs)
    if fa is not None:
        known = [i for i, f in enumerate(fps) if f is not None]
        for i, value in zip(known, DataStructs.BulkTanimotoSimilarity(fa, [fps[i] for i in known])):
            out[i] = value
    return out


def apply_contrast(samples):
    """Set `reward` of every sample to its contrastive reward over the drugs of `samples`: reward_full minus the weight of the reward's
    similarity term (W_TANI) times the mean of T(answer, reference) over the OTHER drugs, floored at 0 (no usable answer scores below a
    failure), and 0 for a sample that is not valid, closed and finished. Each sample carries drug, ref (its drug's reference), answer,
    reward_full, valid, closed and truncated. The drugs are all those attempted in the step, one reference each, before any zero-variance
    filtering; call it again whenever the drug set changes (a refill) so that every sample sees the same others. Each distinct answer gets
    one bulk similarity against the references."""
    from rlexp import reward
    refs = {}
    for s in samples:
        refs.setdefault(s["drug"], s["ref"])
    if len(refs) < 2:
        raise PackError("The contrastive reward needs at least two drugs in the batch")
    position = {drug: i for i, drug in enumerate(refs)}
    usable = [grpo.usable(s) for s in samples]
    cache = {}
    for s, ok in zip(samples, usable):
        if ok and s["answer"] not in cache:
            cache[s["answer"]] = similarities(s["answer"], list(refs.values()))
    rewards = grpo.contrastive_rewards([s["reward_full"] for s in samples], usable, [position[s["drug"]] for s in samples],
                                       [cache[s["answer"]] if ok else None for s, ok in zip(samples, usable)], reward.W_TANI)
    for s, r in zip(samples, rewards):
        s["reward"] = r


def training_rewards(samples, reward_mode):
    """Make `reward` final for these samples. Only the contrastive reward needs it: it depends on the batch."""
    if reward_mode == "contrastive":
        apply_contrast(samples)


def reference_keys(rows):
    """Connectivity keys of the references (and their alternates) of these rows: what copying a Train structure means."""
    from rlexp import chem
    keys = set()
    for row in rows:
        alternates = row.get("ref_alt_smiles") or []
        for smiles in [row["ref_smiles"], *([alternates] if isinstance(alternates, str) else alternates)]:
            mol = chem.standardize(smiles)
            key = None if mol is None else chem.conn(mol)
            if key:
                keys.add(key)
    return frozenset(keys)


def lift_of(samples):
    """Profile-specific lift (docs/plan.md section 2) of samples that carry drug, ref and answer; the one seam to metrics v3's lift."""
    return grpo.drug_lift(samples, similarity)


def batch_diagnostics(groups, train_keys):
    """Per-step diagnostics over all attempted groups, before the zero-variance filter (the arithmetic is in rlexp.grpo; here only the
    chemistry): answer concentration, copies of another Train drug's reference (an exact hit on the drug's own is not one), the modal
    answers' similarity, the reward's terms and the profile-specific lift against the other drugs of the batch."""
    from rlexp import reward
    keys = [[s["key"] for s in g["samples"]] for g in groups]
    answers = [[s["answer"] for s in g["samples"]] for g in groups]
    lift_samples = [dict(drug=g["row"]["group"], ref=g["row"]["ref_smiles"], answer=s["answer"]) for g in groups for s in g["samples"]]
    return dict(top_answer_share=grpo.top_answer_share(keys), distinct_answer_share=grpo.distinct_answer_share(keys),
                modal_pairwise_tanimoto=grpo.mean_pairwise(grpo.modal_answers(keys, answers), similarity),
                train_copy_share=grpo.copy_share(keys, train_keys, [reference_keys([g["row"]]) for g in groups]),
                collapse_all=grpo.collapse_all(keys),
                lift_batch=lift_of(lift_samples),
                **grpo.reward_components([s for g in groups for s in g["samples"]], reward.W_VALID, reward.W_TANI, reward.W_SCAF))


# ---------------------------------------------------------------- generation (torch is passed in or imported on use)
def stop_ids(im_end, tokenizer_eos, config_eos):
    """Every token that ends a rollout, as vLLM stops: <|im_end|>, the model's generation_config eos_token_id (one id, a list or None)
    and the tokenizer's eos, each once, in that order. Training rollouts and evaluation then agree on what a finished answer is."""
    found = [im_end]
    for value in (*(config_eos if isinstance(config_eos, (list, tuple)) else [config_eos]), tokenizer_eos):
        if isinstance(value, int) and not isinstance(value, bool) and value not in found:
            found.append(value)
    return found


def trim_generated(ids, stops, max_new):
    """The generated tokens up to and including the first stop token ("stop"; `stops` is one id or a collection of ids); without one,
    all of them ("length")."""
    stops = {stops} if isinstance(stops, int) else set(stops)
    for i, token in enumerate(ids):
        if token in stops:
            return ids[:i + 1], "stop"
    return ids[:max_new], "length"


def generate_all(call, prompts, n, pad_id, stops, max_new, batch, torch, device, token_budget=None):
    """n completions of each prompt through `call(input_ids, attention_mask) -> (sequences, logprobs or None)`. Sequences are
    left-padded and generated in sub-batches of at most `batch` sequences and, when `token_budget` is given, at most
    token_budget // (longest prompt of the sub-batch) sequences (the prefill memory grows faster than rows x prompt length: 192
    sequences of 1.8k-token prompts ran out of memory on a 40 GB card, and so did 81920 tokens), longest prompt first (less padding);
    results come back in prompt order as {ids, finish[, logp]} per completion, cut at the first of the `stops` ids."""
    flat = [(k, list(p)) for k, p in enumerate(prompts) for _ in range(n)]
    if any(not p for _, p in flat):
        raise PackError("A prompt has no tokens")
    order = sorted(range(len(flat)), key=lambda j: (-len(flat[j][1]), j))
    out = [None] * len(flat)
    start = 0
    while start < len(order):
        longest = len(flat[order[start]][1])
        size = batch if token_budget is None else max(1, min(batch, token_budget // longest))
        chunk = order[start:start + size]
        start += len(chunk)
        ids = torch.tensor([[pad_id] * (longest - len(flat[j][1])) + flat[j][1] for j in chunk], device=device)
        mask = torch.tensor([[0] * (longest - len(flat[j][1])) + [1] * len(flat[j][1]) for j in chunk], device=device)
        sequences, logp = call(ids, mask)
        new = sequences[:, longest:].tolist()
        for b, j in enumerate(chunk):
            tokens, finish = trim_generated(new[b], stops, max_new)
            out[j] = dict(ids=tokens, finish=finish)
            if logp is not None:
                out[j]["logp"] = logp[b, :len(tokens)].tolist()
    return [out[k * n:(k + 1) * n] for k in range(len(prompts))]


# ---------------------------------------------------------------- the fp32 head and the weighted loss (torch)
def fp32_head(weight):
    """A head holding an fp32 copy of `weight`, computing in fp32 even inside an autocast region (autocast would otherwise run the
    matmul in bf16). Defined here so importing this file needs no torch."""
    import torch

    class Fp32Head(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.register_buffer("weight", w.detach().to(torch.float32).clone(), persistent=False)

        def forward(self, hidden):
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                return torch.nn.functional.linear(hidden.to(torch.float32), self.weight)

    return Fp32Head(weight)


def install_fp32_head(model):
    """Replace the tied LM head of the HF model by an fp32 copy; the embedding table stays bf16 and untouched."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.lm_head = fp32_head(base.lm_head.weight)
    return base.lm_head


def micro_batches(lengths, budget):
    """Greedy micro-batches of sample indices, longest first, rows times the longest sequence (padding included) within the budget."""
    batches, batch, longest = [], [], 0
    for i in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        if lengths[i] < 1 or lengths[i] > budget:
            raise PackError("A sample cannot fit the token budget. Raise --max-tokens-per-batch")
        if batch and max(longest, lengths[i]) * (len(batch) + 1) > budget:
            batches.append(batch)
            batch, longest = [], 0
        batch.append(i)
        longest = max(longest, lengths[i])
    if batch:
        batches.append(batch)
    return batches


def collate(samples, pad_id, device, torch):
    """Right-pad (prompt_ids, completion_ids, weight) triples. Only completion positions are supervised."""
    longest = max(len(p) + len(c) for p, c, _ in samples)
    fill = lambda p, c: longest - len(p) - len(c)
    return dict(
        input_ids=torch.tensor([p + c + [pad_id] * fill(p, c) for p, c, _ in samples], device=device),
        labels=torch.tensor([[-100] * len(p) + c + [-100] * fill(p, c) for p, c, _ in samples], device=device),
        attention_mask=torch.tensor([[1] * (len(p) + len(c)) + [0] * fill(p, c) for p, c, _ in samples], device=device),
        weights=torch.tensor([w for _, _, w in samples], dtype=torch.float32, device=device))


def completion_logprobs(model, batch, temperature, chunk=CHUNK):
    """Per-token log pi(y_t | y_<t) of the supervised positions (labels != -100), flat in row-major order, at the sampling
    temperature. The head runs on completion positions only, `chunk` at a time; with autograd on, each chunk is recomputed in the
    backward pass, so the (chunk x vocab) fp32 logits never pile up."""
    import torch
    from torch.utils.checkpoint import checkpoint
    base = model.get_base_model()
    hidden = base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False, return_dict=True).last_hidden_state
    labels = batch["labels"][:, 1:]
    valid = labels != -100
    hidden, targets = hidden[:, :-1][valid], labels[valid]

    def head_logp(states, wanted):
        return -torch.nn.functional.cross_entropy(base.lm_head(states) / temperature, wanted, reduction="none")

    parts = []
    for start in range(0, len(targets), chunk):
        states, wanted = hidden[start:start + chunk], targets[start:start + chunk]
        parts.append(checkpoint(head_logp, states, wanted, use_reentrant=False) if torch.is_grad_enabled() else head_logp(states, wanted))
    return torch.cat(parts) if parts else hidden.new_zeros(0, dtype=torch.float32)


def update_gradients(model, batches, temperature, chunk=CHUNK, context=None):
    """Backward every micro-batch of one update and return the summed loss -sum(w * log pi). Each token's w is already
    A_i / (T_g * G), so micro-batches just add up: how the update is split cannot change its gradient. `context` (autocast)
    wraps the forward only."""
    total = 0.0
    for batch in batches:
        with context() if context else contextlib.nullcontext():
            logp = completion_logprobs(model, batch, temperature, chunk)
        valid = batch["labels"][:, 1:] != -100
        loss = -(batch["weights"][:, None].expand_as(valid)[valid] * logp).sum()
        value = loss.item()
        if not math.isfinite(value):
            raise PackError("Non-finite loss. The last checkpoint is intact")
        loss.backward()
        total += value
    return total


# ---------------------------------------------------------------- the model side
def attach_adapter(model, source, torch):
    """Wrap the base text model with its LoRA: continue the PEFT adapter in `source`, or make the fresh rank-32 one when it is None.
    Fails unless the adapter modules are exactly the language-model projections and the trainable parameter count is the expected
    one (rank * (in + out) per module). Returns the PEFT model and its adapter module names."""
    from peft import LoraConfig, PeftModel, get_peft_model
    targets = sft.adapter_targets(model, torch.nn.Linear)
    layers = dict(model.named_modules())
    if source is None:
        rank = RANK
        model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, bias="none",
                                                 target_modules=targets, task_type="CAUSAL_LM"))
    else:
        rank = check_adapter_config(source)["r"]
        model = PeftModel.from_pretrained(model, source, is_trainable=True)
    actual = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    if {name.removeprefix("base_model.model.") for name in actual} != set(targets):
        raise PackError("Actual LoRA modules differ from the requested modules")
    trainable = [p for p in model.parameters() if p.requires_grad]
    expected = sum(rank * (layers[name].in_features + layers[name].out_features) for name in targets)
    if sum(p.numel() for p in trainable) != expected:
        raise PackError("The trainable parameter count differs from the expected LoRA size")
    for parameter in trainable:
        if parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(torch.float32)         # LoRA masters are fp32; autocast runs them in bf16
    return model, actual


def publish_adapter(root, step, write, revision):
    """Keep step N's adapter for good in <root>/adapters/step-N: `write(directory)` saves the two adapter files into a temporary
    directory, the base-revision pin follows, then the directory is renamed into place. No optimizer state. Returns the state entry
    (path and the sha256 of the three files). An identical repeat is a no-op. A different copy of a step the checkpoint has not
    committed (the leftover of a killed run) is swapped in: the old directory is renamed aside first and renamed back if the swap
    fails, so a failed swap never loses it. A committed step never changes. Call it before the checkpoint that records the step."""
    directory = root / "adapters"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=directory))
    target = directory / f"step-{step}"
    try:
        write(temporary)
        for extra in set(os.listdir(temporary)) - set(ADAPTER_FILES):
            (temporary / extra).unlink()
        sft.atomic_json(temporary / PIN_FILE, dict(revision=revision))
        for name in KEPT_FILES:
            with (temporary / name).open("rb") as stream:
                os.fsync(stream.fileno())
        entry = dict(step=step, path=str(target.relative_to(root)), sha256={name: file_sha256(temporary / name) for name in KEPT_FILES})
        sft.sync_directory(temporary)
        if target.exists():
            if {file.name: file_sha256(file) if file.is_file() else None for file in target.iterdir()} == entry["sha256"]:
                return entry
            loaded = sft.load_checkpoint(root)
            if entry["path"] in {kept["path"] for kept in (loaded[1].get("adapters", []) if loaded else [])}:
                raise PackError(f"Adapter step {step} is committed and cannot change")
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
        sft.sync_directory(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return entry


class Policy:
    """The trainable PEFT model with its tokenizer: rollouts, Val sampling, the update and the checkpoints. UNTESTED on a GPU."""

    def __init__(self, model, tokenizer, args, dtype, device="cuda", revision=None):
        import torch
        from transformers import GenerationConfig
        self.torch, self.model, self.tokenizer, self.args, self.dtype, self.device = torch, model, tokenizer, args, dtype, device
        self.revision = revision
        ends = tokenizer.encode(END, add_special_tokens=False)
        if len(ends) != 1:
            raise PackError("The end marker is not a single token")
        self.eos = ends[0]
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if not isinstance(self.pad, int):
            raise PackError("Tokenizer has no padding or EOS token")
        probe = tokenizer.decode(tokenizer.encode("\n</think>\n\nCC", add_special_tokens=False) + [self.eos],
                                 skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if THINK not in probe:
            raise PackError("skip_special_tokens removes </think>; the reward would never see a closed think block")
        self.prompts = {}
        self.params = [p for p in model.parameters() if p.requires_grad]
        if not self.params:
            raise PackError("No trainable adapter parameters")
        base = model.get_base_model()
        base.config.use_cache = True                  # the training forward passes use_cache=False explicitly
        install_fp32_head(model)
        # The checkpoint's generation config lists the model's stop tokens (vLLM stops on them): read it before it is replaced below.
        self.stops = stop_ids(self.eos, tokenizer.eos_token_id, getattr(getattr(base, "generation_config", None), "eos_token_id", None))
        # Explicit sampling, and a clean generation config on the model: the checkpoint's own sets top_k, top_p and temperature, and
        # transformers fills every field left unset from it.
        self.sampling = dict(do_sample=True, temperature=args.temperature, top_k=0, top_p=1.0, repetition_penalty=1.0, typical_p=1.0,
                             no_repeat_ngram_size=0, epsilon_cutoff=0.0, eta_cutoff=0.0, num_beams=1, use_cache=True,
                             max_new_tokens=args.max_new_tokens, eos_token_id=list(self.stops), pad_token_id=self.pad)
        base.generation_config = GenerationConfig(**self.sampling)
        self.optimizer = torch.optim.AdamW(self.params, lr=args.lr, weight_decay=0.0)
        self.initial = None

    def autocast(self):
        return self.torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype)

    def prompt_ids(self, row):
        key = row["id"]
        if key not in self.prompts:
            self.prompts[key] = self.tokenizer.encode(row["rl_prompt"], add_special_tokens=False)
        return self.prompts[key]

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _generate(self, logprobs, ids, mask):
        torch = self.torch
        out = self.model.generate(input_ids=ids, attention_mask=mask, **self.sampling,
                                  **(dict(return_dict_in_generate=True, output_scores=True) if logprobs else {}))
        if not logprobs:
            return out, None
        # The scores are the exact distribution that was sampled from (temperature included).
        length = ids.shape[1]
        steps = [torch.log_softmax(score.float(), dim=-1).gather(1, out.sequences[:, length + t:length + t + 1])
                 for t, score in enumerate(out.scores)]
        return out.sequences, torch.cat(steps, dim=1)

    def generate(self, prompts, n, seed, logprobs=False):
        """n completions of each prompt, torch seeded by `seed`. Per prompt a list of {ids, finish[, logp]}."""
        self.torch.manual_seed(seed)
        self.model.eval()
        with self.torch.no_grad(), self.autocast():
            out = generate_all(lambda ids, mask: self._generate(logprobs, ids, mask), prompts, n, self.pad, self.stops,
                               self.args.max_new_tokens, self.args.gen_batch, self.torch, self.device,
                               self.args.gen_token_budget)
        if self.device == "cuda":
            self.torch.cuda.empty_cache()         # hand the KV caches back before training allocates its activations
        return out

    def rollout(self, rows, seed):
        return self.generate([self.prompt_ids(row) for row in rows], self.args.group_size, seed)

    def evaluate(self, rows, k, seed):
        return self.generate([self.prompt_ids(row) for row in rows], k, seed)

    def train(self, items, lr):
        """One optimizer step on (row, completion ids, weight) items. Returns loss, grad_norm and the micro-batch count."""
        torch = self.torch
        samples = [(self.prompt_ids(row), list(ids), weight) for row, ids, weight in items]
        batches = micro_batches([len(p) + len(c) for p, c, _ in samples], self.args.max_tokens_per_batch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        loss = update_gradients(self.model, (collate([samples[i] for i in batch], self.pad, self.device, torch) for batch in batches),
                                self.args.temperature, context=self.autocast)
        norm = torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        if not torch.isfinite(norm).item():
            raise PackError("Non-finite gradients. The last checkpoint is intact; reduce lr")
        self.optimizer.step()      # ponytail: one update per rollout batch (ratio 1); several would need old log-probs and CISPO's clipped ratio
        self.model.eval()
        return dict(loss=loss, grad_norm=float(norm), micro_batches=len(batches))

    def token_logprobs(self, pairs):
        """Training-forward log-probs of (prompt ids, completion ids) pairs: right-padded micro-batches, chunked fp32 head, no grad."""
        torch = self.torch
        out = [None] * len(pairs)
        self.model.train()
        with torch.no_grad():
            for batch in micro_batches([len(p) + len(c) for p, c in pairs], self.args.max_tokens_per_batch):
                with self.autocast():
                    flat = completion_logprobs(self.model, collate([(*pairs[i], 0.0) for i in batch], self.pad, self.device, torch),
                                               self.args.temperature).tolist()
                at = 0
                for i in batch:
                    out[i] = flat[at:at + len(pairs[i][1])]
                    at += len(pairs[i][1])
        self.model.eval()
        return out

    def mark_initial(self):
        self.initial = [p.detach().cpu().clone() for p in self.params]

    def delta_norm(self):
        return math.sqrt(sum(float(((p.detach().cpu() - q) ** 2).sum()) for p, q in zip(self.params, self.initial)))

    def snapshot(self):
        """CPU copies of the trainable parameters and the AdamW moments (flat float32, in parameter order, with the parameter names), the
        AdamW step counts and the python, numpy, torch (and cuda) RNG states: what the smoke gate compares (a parameter without optimizer
        state has None moments)."""
        import numpy as np
        flat = lambda tensor: None if tensor is None else tensor.detach().cpu().float().flatten().clone()
        states = [self.optimizer.state.get(p, {}) for p in self.params]
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=self.torch.get_rng_state().clone())
        if self.device == "cuda":
            rng["cuda"] = [state.clone() for state in self.torch.cuda.get_rng_state_all()]
        return dict(names=[name for name, p in self.model.named_parameters() if p.requires_grad], parameters=[flat(p) for p in self.params],
                    exp_avg=[flat(s.get("exp_avg")) for s in states], exp_avg_sq=[flat(s.get("exp_avg_sq")) for s in states],
                    step=[float(s["step"]) if "step" in s else 0.0 for s in states], rng=rng)

    def reset_peak(self):
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    def peak_memory(self):
        return int(self.torch.cuda.max_memory_allocated()) if self.device == "cuda" else 0

    def describe(self):
        if self.device != "cuda":
            return dict(name=self.device, memory_bytes=0)
        return dict(name=self.torch.cuda.get_device_name(0), memory_bytes=self.torch.cuda.get_device_properties(0).total_memory)

    def save(self, state):
        """Atomic resumable checkpoint: adapter, optimizer, python/numpy/torch/cuda RNG and the loop state."""
        import numpy as np
        torch = self.torch

        def writer(path):
            self.model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
            name, keys, position, gauss, cached = np.random.get_state()
            training = dict(optimizer=self.optimizer.state_dict(), python_rng=random.getstate(),
                            numpy_rng=(name, keys.tolist(), position, gauss, cached), torch_rng=torch.get_rng_state())
            if self.device == "cuda":
                training["cuda_rng"] = torch.cuda.get_rng_state_all()
            torch.save(training, path / "training.pt")

        sft.commit_checkpoint(self.args.out, state, writer)

    def load_training(self, directory):
        import numpy as np
        torch = self.torch
        saved = torch.load(Path(directory) / "training.pt", map_location="cpu", weights_only=True)    # tensors and plain types only
        self.optimizer.load_state_dict(saved["optimizer"])
        random.setstate(saved["python_rng"])
        name, keys, position, gauss, cached = saved["numpy_rng"]
        np.random.set_state((name, np.array(keys, dtype=np.uint32), position, gauss, cached))
        torch.set_rng_state(saved["torch_rng"])
        if self.device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])

    def save_adapter(self, step):
        """Keep this step's adapter for good (see publish_adapter)."""
        return publish_adapter(self.args.out, step, lambda path: self.model.save_pretrained(path, safe_serialization=True,
                                                                                            save_embedding_layers=False), self.revision)


def prepare_runtime(args):
    """Imports, bf16 and kernel gates, seeds. Transformers selects its kernels at import, so the probe runs first."""
    os.environ["USE_HUB_KERNELS"] = "0"
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise PackError("CUDA is required for RL; use --dry-run on CPU")
    if sft.precision_for_capability(torch.cuda.get_device_capability(0)) != "bfloat16":
        raise PackError("The RL pack needs bf16 (compute capability 8.0+): fp16 would need loss scaling")
    kernels = sft.kernel_status(args.allow_slow_kernels)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return torch, torch.bfloat16, kernels


def load_policy(args, existing=None):
    """The model, tokenizer and adapter of a run, on the GPU: from the checkpoint when resuming, else the init adapter."""
    torch, dtype, kernels = prepare_runtime(args)
    from transformers import AutoTokenizer
    source = init_source(vars(args))
    revision = base_revision(existing, source, args.revision)
    if revision is None and source is not None:
        print("WARNING: The init adapter records no base revision. The base model loads at the Hub's current one; "
              "compare identity.revision with the A2 run's.", flush=True)
    model, config, revision = sft.load_text_model(args.model, dtype, revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
    model, modules = attach_adapter(model, existing[0] if existing else source, torch)
    model.to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    policy = Policy(model, tokenizer, args, dtype, revision=revision)
    return policy, dict(revision=revision, kernels=kernels, dtype=dtype, adapter_modules=len(modules), stop_ids=policy.stops,
                        trainable_parameters=sum(p.numel() for p in policy.params))


# ---------------------------------------------------------------- the loop (no torch: `policy` does all model work)
def collect(policy, drugs, picks, seed, reward_mode="full"):
    """Sample and score the groups of the picked (drug, view) pairs. Returns (groups, (generate seconds, score seconds))."""
    rows = [drugs[d][v] for d, v in picks]
    began = time.monotonic()
    completions = policy.rollout(rows, seed)
    generated = time.monotonic()
    groups = []
    for row, group in zip(rows, completions):                         # each sample knows its drug and that drug's reference
        scored = score_completions(policy.decode, group, row, reward_mode)
        groups.append(dict(row=row, samples=[dict(s, drug=row["group"], ref=row["ref_smiles"]) for s in scored]))
    return groups, (generated - began, time.monotonic() - generated)


def plan_for(groups):
    return grpo.plan_update([[s["reward"] for s in g["samples"]] for g in groups],
                            [[s["n_tokens"] for s in g["samples"]] for g in groups])


def run_step(args, state, policy, drugs, sizes, step, train_keys=frozenset()):
    """One step: draw drugs, sample groups, refill once if too few groups carry signal, update. -> (metrics, new cursor)."""
    began = time.monotonic()
    policy.reset_peak()
    picks, cursor = grpo.draw(sizes, args.seed, state["cursor"], args.prompts_per_step)
    groups, (generate_s, score_s) = collect(policy, drugs, picks, grpo.rollout_seed(args.seed, step), args.reward_mode)
    training_rewards([s for g in groups for s in g["samples"]], args.reward_mode)
    plan = plan_for(groups)
    added = 0
    extra = grpo.refill_count(plan["groups"], len(groups), args.min_groups)
    if extra:
        more, cursor = grpo.draw(sizes, args.seed, cursor, extra, exclude={d for d, _ in picks})
        if more:                                  # none left when the batch already holds every drug
            picks, added = picks + more, len(more)
            add_groups, (g2, s2) = collect(policy, drugs, more, grpo.rollout_seed(args.seed, step, 1), args.reward_mode)
            groups, generate_s, score_s = groups + add_groups, generate_s + g2, score_s + s2
            training_rewards([s for g in groups for s in g["samples"]], args.reward_mode)       # the drug set changed: every sample, again
            plan = plan_for(groups)
    lr = grpo.lr_at(step, args.lr)
    trained = plan["groups"] >= grpo.skip_below(args.min_groups)
    result, train_s = {}, 0.0
    if trained:
        items = [(g["row"], s["ids"], w) for g, weights in zip(groups, plan["weights"]) if weights is not None
                 for s, w in zip(g["samples"], weights)]
        first = state["first_update_delta_norm"] is None
        if first:
            policy.mark_initial()
        trained_at = time.monotonic()
        result = policy.train(items, lr)
        train_s = time.monotonic() - trained_at
        if first:
            state["first_update_delta_norm"] = policy.delta_norm()
    all_samples = [s for g in groups for s in g["samples"]]
    summary = grpo.summarize(all_samples)
    metrics = dict(
        step=step, prompts=len(groups), refilled_prompts=added, groups_kept=plan["groups"],
        zero_variance_share=plan["zero_variance"] / len(groups), advantage_std=plan["std"], skipped_update=not trained,
        reward_mean=summary["reward_mean"], reward_std=summary["reward_std"], reward_full_mean=summary["reward_full_mean"],
        reward_full_std=summary["reward_full_std"], valid=summary["valid"], closed=summary["closed"],
        truncation=summary["truncation"], length_mean=summary["length_mean"], length_p90=summary["length_p90"],
        core=summary["core"], exact=summary["exact"], tanimoto=summary["tanimoto"],
        grad_norm=result.get("grad_norm"), loss=result.get("loss"), lr=lr,
        collapse_score=grpo.collapse_score([[s["key"] for s in g["samples"]] for g in groups]),
        seconds_generate=generate_s, seconds_score=score_s, seconds_train=train_s, seconds_step=time.monotonic() - began,
        peak_memory_bytes=policy.peak_memory(), cursor=cursor, rollout_seed=grpo.rollout_seed(args.seed, step),
        schedule_hash=fingerprint([[d, v] for d, v in picks]),
        rollout_hash=fingerprint([[s["ids"] for s in g["samples"]] for g in groups]), **batch_diagnostics(groups, train_keys))
    return metrics, cursor


def format_step(m):
    number = lambda x, spec: "n/a" if x is None else format(x, spec)
    return (f"step={m['step']} prompts={m['prompts']} kept={m['groups_kept']} zero_var={m['zero_variance_share']:.2f} "
            f"reward={m['reward_mean']:.3f}+-{m['reward_std']:.3f} valid={m['valid']:.2f} closed={m['closed']:.2f} "
            f"trunc={m['truncation']:.2f} len={m['length_mean']:.0f}/{m['length_p90']:.0f} core={m['core']:.3f} "
            f"exact={m['exact']:.3f} tani={m['tanimoto']:.3f} gnorm={number(m['grad_norm'], '.3f')} loss={number(m['loss'], '.4f')} "
            f"lr={m['lr']:.1e} collapse={number(m['collapse_score'], '.2f')} lift={number(m['lift_batch'], '+.3f')} "
            f"top={number(m['top_answer_share'], '.2f')} pair={number(m['modal_pairwise_tanimoto'], '.2f')} "
            f"copy={number(m['train_copy_share'], '.2f')}{' SKIPPED' if m['skipped_update'] else ''} "
            f"gen/score/train={m['seconds_generate']:.0f}/{m['seconds_score']:.0f}/{m['seconds_train']:.0f}s "
            f"mem={m['peak_memory_bytes'] / 2 ** 30:.1f}GiB")


def evaluate(args, state, policy, val_rows):
    """Sample every Val row eval_k times at the training temperature; log drug-balanced means, the profile-specific lift (the other Val
    drugs are the others), both rewards and the first-valid policy (fv_*: the first usable sample of each row's eval_k). `val` is
    fv_tanimoto, the endpoint, which the patience rule follows; `val_reward` keeps the Val mean training reward. Keep the adapter
    (never pruned)."""
    began = time.monotonic()
    step = state["step"]
    completions = policy.evaluate(val_rows, args.eval_k, grpo.eval_seed(args.seed, step))
    samples = []
    for row, group in zip(val_rows, completions):
        for sample in score_completions(policy.decode, group, row, args.reward_mode):
            samples.append(dict(sample, drug=row["group"], ref=row["ref_smiles"], prompt=row["id"]))
    training_rewards(samples, args.reward_mode)                      # contrastive: against the other Val drugs
    summary = dict(grpo.drug_balanced(samples), lift=lift_of(samples), **grpo.first_valid_policy(samples, similarity))
    previous = state["evals"][-1]["step"] if state["evals"] else 0
    entry = dict(step=step, val=summary["fv_tanimoto"], val_reward=summary["reward"], train=grpo.train_window(state["metrics"], previous),
                 metrics=summary, adapter=None, seconds=0.0)
    if step > 0:                                  # step 0 is the init adapter itself
        kept = policy.save_adapter(step)
        state["adapters"].append(kept)
        entry["adapter"] = kept["path"]
    entry["seconds"] = time.monotonic() - began
    state["evals"].append(entry)
    print(json.dumps(dict(eval=entry), allow_nan=False), flush=True)
    return entry


def run_steps(args, state, policy, drugs, val_rows, started, stopped, train_keys=frozenset()):
    """The training loop. Returns the state; state["status"] says why it ended. train_keys: connectivity keys of the Train references."""
    sizes = [len(rows) for rows in drugs]

    def checkpoint():
        policy.save(state)
        write_logs(args.out, state)

    if state["status"] in grpo.RULE_STOPS:
        print(f"The run already ended with {state['status']}. Use a new output directory to start over.", flush=True)
        return state
    state["status"] = "running"
    if args.eval_every and not state["evals"]:    # the baseline: the init adapter on Val, by the same sampler
        evaluate(args, state, policy, val_rows)
        checkpoint()
    while state["step"] < args.steps:
        if stopped() or time.monotonic() - started >= args.stop_after_seconds:
            state["status"] = "stopped_signal" if stopped() else "stopped_time_limit"
            checkpoint()
            return state
        step = state["step"] + 1
        began = time.monotonic()
        metrics, cursor = run_step(args, state, policy, drugs, sizes, step, train_keys)
        metrics["similar_streak"] = grpo.similarity_streak([*state["metrics"], metrics])
        state["metrics"].append(metrics)
        state.update(step=step, global_step=step, cursor=cursor,
                     collapse_streak=grpo.collapse_streak(state["collapse_streak"], metrics["collapse_score"]),
                     skipped_updates=state["skipped_updates"] + int(metrics["skipped_update"]),
                     training_seconds=state["training_seconds"] + time.monotonic() - began,
                     peak_memory_bytes=max(state["peak_memory_bytes"], metrics["peak_memory_bytes"]))
        append_jsonl(args.out / "metrics.jsonl", metrics)
        print(format_step(metrics), flush=True)
        if metrics["similar_streak"] >= grpo.SIMILAR_STEPS:        # a warning, never a stop
            print(f"WARNING: The modal answers have been near duplicates (mean pairwise Tanimoto above {grpo.SIMILAR_ABOVE}) for "
                  f"{metrics['similar_streak']} steps; the collapse score counts exact structures only. Read top_answer_share, lift_batch "
                  "and the Val lift.", flush=True)
        status = grpo.stop_status(state["collapse_streak"], metrics["truncation"], state["evals"], args.patience, collapse_stop(args))
        evaluated = False
        if not status and args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            evaluate(args, state, policy, val_rows)
            evaluated = True
            status = grpo.stop_status(state["collapse_streak"], metrics["truncation"], state["evals"], args.patience, collapse_stop(args))
        if status:
            state["status"] = status
            checkpoint()
            return state
        if evaluated or step % args.save_every == 0:
            checkpoint()
    state["status"] = "complete"
    checkpoint()
    return state


def install_signals():
    flag = [False]

    def on_signal(signum, frame):
        flag[0] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    return lambda: flag[0]


def run_training(args, train_rows, val_rows, started, loader=None, restored=None):
    """Train from the init adapter, or resume the run in args.out. Returns the final state. `loader` (default load_policy) is the
    seam the CPU tests use to supply a tiny model; `restored(policy)` is called on a resume right after the checkpoint is loaded and
    before any step (the smoke gate compares the restored state there)."""
    existing = sft.load_checkpoint(args.out)
    policy, info = (loader or load_policy)(args, existing)
    versions = runtime_versions()
    identity = make_identity(args, info["revision"], info["dtype"], versions)
    try:
        git_revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
    state = new_state(identity, versions, gpu=policy.describe(), kernels=info["kernels"], git_revision=git_revision, command=sys.argv,
                      trainer_sha256=trainer_hashes(), adapter_modules=info["adapter_modules"], trainable_parameters=info["trainable_parameters"],
                      stop_token_ids=info.get("stop_ids"), train=dict(rows=len(train_rows), drugs=len(drug_table(train_rows))),
                      val=dict(rows=len(val_rows), drugs=len({r["group"] for r in val_rows})))
    if existing:
        check_resume(existing[1], identity, versions, state["trainer_sha256"])
        if existing[1].get("kernels") != info["kernels"]:
            print("WARNING: The kernel path changed since the checkpoint. Numerical replay is UNTESTED.", flush=True)
        state = restore_state(existing[1], state)
        policy.load_training(existing[0])
        if restored:
            restored(policy)
        write_logs(args.out, state)               # drop the lines of steps that were logged but never committed
    print(json.dumps(dict(adapter_modules=info["adapter_modules"], trainable_parameters=info["trainable_parameters"],
                          resumed_at_step=state["step"], train=state["train"], val=state["val"], dtype=str(info["dtype"]),
                          stop_token_ids=info.get("stop_ids"))), flush=True)
    policy.reset_peak()
    if not existing:
        policy.save(state)                        # the first checkpoint exists before any step can fail
        write_logs(args.out, state)
    return run_steps(args, state, policy, drug_table(train_rows), val_rows, started, install_signals(), reference_keys(train_rows))


# ---------------------------------------------------------------- GPU gates (implemented here, run by the orchestrator)
def parity_stats(generated, trained):
    """Mean and max |diff| of per-token log-probs, the share above 0.05, and the mean of exp(trained - generated)."""
    diffs = [t - g for gen, tra in zip(generated, trained) for g, t in zip(gen, tra)]
    if not diffs:
        raise PackError("No tokens to compare")
    absolute = [abs(d) for d in diffs]
    return dict(tokens=len(diffs), mean_abs=sum(absolute) / len(absolute), max_abs=max(absolute),
                share_above_0_05=sum(a > 0.05 for a in absolute) / len(absolute),
                mean_ratio=sum(math.exp(max(min(d, 50.0), -50.0)) for d in diffs) / len(diffs))


SMOKE_TOLERANCE = 1e-6                # relative norm of difference that two runs on CPU-equivalent paths reach


def parity_ok(stats):
    """One case passes when mean |diff| <= 0.03, at most 15% of tokens are above 0.05 and the mean ratio is within 2% of 1. Measured, then
    chosen (not derived): the accepted A100 runs showed a mean |diff| of 0.015-0.018, 8-11% of tokens above 0.05 and a mean ratio of
    0.994-1.000. The gate exits 0 only if every judged case (batch, mixed lengths, alone) passes."""
    return stats["mean_abs"] <= 0.03 and stats["share_above_0_05"] <= 0.15 and abs(stats["mean_ratio"] - 1) <= 0.02


def run_parity(args, train_rows, started):
    """Sampling log-probs against the training forward, on the same tokens: batched, heavily left-padded, and alone."""
    policy, _ = load_policy(args)
    drugs = drug_table(train_rows)
    sizes = [len(rows) for rows in drugs]
    batch_rows = [drugs[d][v] for d, v in grpo.draw(sizes, args.seed, 0, min(8, len(drugs)))[0]]
    by_length = sorted(train_rows, key=lambda r: (len(policy.prompt_ids(r)), r["id"]))
    mixed_rows = by_length[:4] + by_length[-4:]
    shortest = mixed_rows[0]
    print(json.dumps(dict(sampling={k: v for k, v in policy.sampling.items() if k != "eos_token_id"}, stop_token_ids=policy.stops,
                          effective={k: getattr(policy.model.get_base_model().generation_config, k, None)
                                     for k in ("do_sample", "temperature", "top_k", "top_p", "min_p", "repetition_penalty")})), flush=True)
    report, ok = {}, True
    for name, rows in (("batch", batch_rows), ("mixed_lengths", mixed_rows), ("alone", [shortest])):
        prompts = [policy.prompt_ids(r) for r in rows]
        n = 4
        completions = policy.generate(prompts, n, grpo.rollout_seed(args.seed, 1), logprobs=True)
        pairs = [(prompts[k], c["ids"]) for k, group in enumerate(completions) for c in group]
        flat = [c for group in completions for c in group]
        trained = policy.token_logprobs(pairs)
        report[name] = parity_stats([c["logp"] for c in flat], trained)
        report[name].update(prompts=len(rows), prompt_tokens=[min(map(len, prompts)), max(map(len, prompts))])
        if name == "mixed_lengths":              # the most padded prompt on its own (a large pad)
            first = [i for i in range(len(flat)) if i < n]
            report["mixed_shortest_prompt"] = parity_stats([flat[i]["logp"] for i in first], [trained[i] for i in first])
        ok = ok and parity_ok(report[name])
    for name, stats in report.items():
        print(json.dumps(dict(case=name, **stats)), flush=True)
    print("PARITY " + ("PASS" if ok else "FAIL: a case is outside the thresholds in gpu/RL.md"), flush=True)
    return ok


def snapshot_distances(a, b):
    """Relative norm of the difference between two Policy.snapshot()s, per component (parameters, AdamW exp_avg and exp_avg_sq):
    ||a - b|| / ||a||, with a missing tensor read as zeros (an optimizer that lost its state differs from one that kept it) and 0.0 for
    two zero components; the AdamW step counts must match exactly or `step` is 1.0."""
    out = {}
    for key in ("parameters", "exp_avg", "exp_avg_sq"):
        gap = size = 0.0
        for x, y in zip(a[key], b[key]):
            if x is None and y is None:
                continue
            x, y = (y * 0 if x is None else x), (x * 0 if y is None else y)
            gap += float(((x.double() - y.double()) ** 2).sum())
            size += float((x.double() ** 2).sum())
        out[key] = math.sqrt(gap / size) if size else math.sqrt(gap)
    out["step"] = float(a["step"] != b["step"])
    return out


def snapshot_difference(a, b):
    """The exact comparison of two Policy.snapshot()s: the largest absolute difference of the parameters, of the AdamW exp_avg and
    exp_avg_sq (a missing tensor reads as zeros) and of the step counts; `names` is 1.0 when the parameter names or their order differ
    and `rng` counts the RNG states that differ. inf where the shapes or lengths disagree, NaN stays NaN. All zeros means identical."""
    def worst(diffs):
        return math.nan if any(d != d for d in diffs) else max(diffs, default=0.0)

    out = {}
    for key in ("parameters", "exp_avg", "exp_avg_sq"):
        diffs = [math.inf] if len(a[key]) != len(b[key]) else []
        for x, y in zip(a[key], b[key]):
            if x is None and y is None:
                continue
            x, y = (y * 0 if x is None else x), (x * 0 if y is None else y)
            diffs.append(math.inf if x.shape != y.shape else float((x.double() - y.double()).abs().max()) if x.numel() else 0.0)
        out[key] = worst(diffs)
    out["step"] = worst([math.inf] if len(a["step"]) != len(b["step"]) else [abs(p - q) for p, q in zip(a["step"], b["step"])])
    out["names"] = float(a["names"] != b["names"])

    def same(x, y):
        if isinstance(x, list):
            return isinstance(y, list) and len(x) == len(y) and all(same(i, j) for i, j in zip(x, y))
        if hasattr(x, "shape"):
            return hasattr(y, "shape") and x.shape == y.shape and not bool((x != y).any())
        if isinstance(x, tuple) and len(x) == 5 and hasattr(x[1], "tolist"):         # numpy's state: name, keys, position, gauss, cached
            return isinstance(y, tuple) and x[0] == y[0] and x[1].tolist() == y[1].tolist() and x[2:] == y[2:]
        return x == y

    out["rng"] = float(sum(not same(a["rng"].get(key), b["rng"].get(key)) for key in {*a["rng"], *b["rng"]}))
    return out


def smoke_checks(full, resumed, distances=None, restored=None):
    """Pass/fail checks (and informational lines) of the smoke gate from the final states of the uninterrupted run and the run that
    was stopped after step 2, rebuilt from disk and finished. `restored` (snapshot_difference between the state the interrupted run
    ended with and the state of the resumed process right after the checkpoint was loaded, before any step: no sampling in between, so
    no GPU noise) must be all zeros: a broken optimizer, RNG or parameter-order restore FAILS here. `distances` (snapshot_distances of
    the two runs' final parameters and AdamW state, after the third update): PASS when all are within SMOKE_TOLERANCE; otherwise only an
    INFO line with the numbers, because rollouts that differ bitwise (GPU nondeterminism) change the third update."""
    delta = full["first_update_delta_norm"]
    a, b = full["metrics"], resumed["metrics"]
    ran = len(a) == 3 and len(b) == 3
    same3 = lambda key: ran and a[2][key] == b[2][key]
    checks = {
        "three steps ran": ran and full["status"] == resumed["status"] == "complete",
        "first update changed the adapter (delta norm > 0 and finite)": delta is not None and math.isfinite(delta) and delta > 0,
        "losses finite": all(m["loss"] is None or math.isfinite(m["loss"]) for m in a + b),
        "resumed from step 2": resumed["step"] == 3 and [m["step"] for m in b] == [1, 2, 3],
        "step 3 schedule equal": same3("schedule_hash") and same3("cursor"),
        "step 3 rollout seed equal": same3("rollout_seed"),
        "steps 1-2 identical in both runs (schedule, seed)": ran and all(a[i][key] == b[i][key] for i in (0, 1) for key in ("schedule_hash", "rollout_seed")),
    }
    info = {"step 3 rollouts bitwise equal (GPU nondeterminism can break this)": same3("rollout_hash"),
            "step 3 loss (uninterrupted, resumed)": [a[2]["loss"], b[2]["loss"]] if ran else None}
    if restored is not None:
        checks["right after the load, the resumed weights, AdamW state and RNG states equal the checkpoint's "
               f"(exact; max abs difference {restored})"] = all(value == 0 for value in restored.values())
    if distances is not None:
        if all(value <= SMOKE_TOLERANCE for value in distances.values()):
            checks[f"step 3 parameters and AdamW state equal the uninterrupted run (relative norm of difference <= {SMOKE_TOLERANCE:g})"] = True
        else:
            info["step 3 parameters and AdamW state differ from the uninterrupted run, relative norm of difference "
                 f"(above {SMOKE_TOLERANCE:g}: compare the rollout line)"] = distances
    return checks, info


def run_smoke(args, train_rows, val_rows, started, loader=None):
    """Three real steps with 8 prompts, then kill-and-resume equivalence: steps 1-2, checkpoint, rebuild everything from disk,
    step 3, against an uninterrupted run of the same three steps; afterwards the final parameters and AdamW state of the two are
    compared (snapshot_distances). `loader` is the seam run_training has."""
    root = args.out / "gate-smoke"
    if root.exists():
        shutil.rmtree(root)

    def sub(name, steps):
        sub_args = copy.copy(args)
        sub_args.out, sub_args.steps, sub_args.prompts_per_step = root / name, steps, 8
        sub_args.min_groups, sub_args.eval_every, sub_args.save_every, sub_args.gate = min(args.min_groups, 4), 0, 1, None
        sub_args.stop_after_seconds = max(args.stop_after_seconds, 3600)
        sub_args.out.mkdir(parents=True, exist_ok=True)
        return sub_args

    def settle():
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    def finish(name, steps, restored=None):
        """Run to `steps`; return the final state and a CPU snapshot of the final weights, AdamW state and RNG states; free the model."""
        made = []

        def load(run_args, existing=None):
            made.append((loader or load_policy)(run_args, existing))
            return made[-1]

        state = run_training(sub(name, steps), train_rows, val_rows, time.monotonic(), loader=load, restored=restored)
        snapshot = made[0][0].snapshot()
        made.clear()
        settle()
        return state, snapshot

    full, snapshot_full = finish("a", 3)
    _, snapshot_saved = finish("b", 2)            # what the checkpoint of the interrupted run holds
    exact = []
    resumed, snapshot_resumed = finish("b", 3, restored=lambda policy: exact.append(snapshot_difference(snapshot_saved, policy.snapshot())))
    checks, info = smoke_checks(full, resumed, snapshot_distances(snapshot_full, snapshot_resumed), exact[0] if exact else dict(restored=math.inf))
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    for name, value in info.items():
        print(f"INFO {name}: {value}", flush=True)
    print(f"first update delta norm: {full['first_update_delta_norm']}", flush=True)
    return all(checks.values())


# ---------------------------------------------------------------- entry
def main(argv=None):
    started = time.monotonic()
    args = config_from_args(argv)
    train, val = load_data(args)
    if len(drug_table(train)) < args.prompts_per_step:
        raise PackError("prompts_per_step exceeds the number of Train drugs")
    if args.dry_run:
        print(json.dumps(dry_plan(args, train, val), indent=2))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for pending in args.out.glob("*/.pending-*"):
            if pending.is_dir():
                shutil.rmtree(pending)
        if args.gate == "parity":
            return 0 if run_parity(args, train, started) else 1
        if args.gate == "smoke":
            return 0 if run_smoke(args, train, val, started) else 1
        state = run_training(args, train, val, started)
        print(json.dumps(dict(status=state["status"], step=state["step"], evals=len(state["evals"]),
                              skipped_updates=state["skipped_updates"])), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        sft.print_failure("RL", error)
        sys.exit(1)
