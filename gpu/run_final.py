#!/usr/bin/env python3
"""Operator script for the locked evaluation on the Colab A100 (gpu/FINAL.md section 1): merge the frozen adapters, check on Val that each merged model reproduces its Val score,
then sample every arm of the frozen spec in a fixed order, clearing the Hugging Face cache between model families and syncing runs/final to Drive after each stage.

  python gpu/run_final.py --spec results/frozen_arms.json --selection results/selection.json --manifest data/MANIFEST.sha256 [--drive /content/drive/MyDrive/rl-exp] [--dry-run]

The spec hash is read from `<spec>.sha256`-style file `results/frozen_arms.sha256` (--spec-sha256-file). The order is fixed in STAGES below and holds no choice: each stage is a list of arm ids that
gpu/final_eval.py samples under the frozen spec; an arm of the spec that no stage names stops the script before anything runs, so no frozen arm is skipped silently. A stage whose ids are not all in the spec
runs the ones that are (an arm dropped before the freeze is an omission recorded in the protocol). The Val sanity gate calls gpu/sample_vllm.py on Val for each merged adapter arm at its frozen decoding and
compares the first-valid similarity with the selection's score: more than --tolerance apart stops the script (the model on disk is not the selected one). Needs both venvs of gpu/bootstrap_colab.sh.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VLLM_PY = "gpu/.venv/bin/python"
TRAIN_PY = ".agent-runs/sft-venv/bin/python"
SANITY = Path(".agent-runs/sanity")      # Val samples of the merged models: never inside runs/final, which holds Test samples only
DIRECT = "Reply with only the SMILES string of the drug described. No explanation."
# (stage, arm ids, clear the HF cache and the merged models before the stage: the 4B stage holds the 4B base plus two merged models (27 GB of the runtime's 35 free), so every later family starts clean).
# The confirmatory pair comes first.
STAGES = [("4B arms", ["RL-full", "A2", "A0n", "A4", "A2-shuffle", "A2-blank", "RL-full-shuffle", "RL-full-blank", "name-4B", "ident-4B"], False),
          ("9B", ["9B", "name-9B", "ident-9B"], True),
          ("Qwen3.5-27B", ["27B-3.5", "name-27B", "ident-27B"], True),
          ("Qwen3.8-27B", ["27B-3.8", "name-3.8", "ident-3.8"], True)]


def plan(spec):
    """[(stage, ids present in the spec, clear_cache)]; ValueError if a spec arm belongs to no stage."""
    named = {i for _, ids, _ in STAGES for i in ids}
    missing = [a["id"] for a in spec["arms"] if a["id"] not in named]
    if missing:
        raise ValueError(f"spec arms in no stage of gpu/run_final.py: {missing}")
    present = {a["id"] for a in spec["arms"]}
    return [(name, [i for i in ids if i in present], clear) for name, ids, clear in STAGES if any(i in present for i in ids)]


def run(cmd, dry, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd, cwd=ROOT, **kw).returncode


def sanity(arm, selection, tolerance, dry):
    """Sample a merged adapter arm on Val at its frozen decoding and compare its first-valid similarity with the selection's score."""
    chosen = selection["arms"][arm["id"]]
    cmd = [VLLM_PY, "gpu/sample_vllm.py", "--model", arm["model"], "--data", "data/processed", "--splits", "val", "--mode", arm["mode"], "--k", "8", "--max-tokens", str(arm["max_tokens"]),
           "--temperature", str(arm["temperature"]), "--prompts-per-batch", "auto", "--out", str(SANITY / arm["id"]), "--system-text", DIRECT]
    if run(cmd, dry):
        sys.exit(f"error: the Val sanity sample of {arm['id']} failed")
    if dry:
        return
    samples = SANITY / arm["id"] / "samples.jsonl"
    env = {**os.environ, "PYTHONPATH": "src"}
    scored, report = samples.with_name("scored.v3.jsonl"), samples.with_name("report.v3.json")
    for args in (["score", "--samples", str(samples), "--data", "data/processed", "--out", str(scored)],
                 ["report", "--scored", str(scored), "--split", "val", "--data", "data/processed", "--boot", "100", "--json", str(report)]):
        if run([TRAIN_PY, "-m", "rlexp.metrics", *args], False, env=env, capture_output=True):
            sys.exit(f"error: scoring the Val sanity sample of {arm['id']} failed")
    value = json.loads(report.read_text(encoding="utf-8"))["splits"]["val"]["metrics"]["fv_tanimoto"]
    value = value["value"] if isinstance(value, dict) else value
    print(f"sanity {arm['id']}: Val first-valid similarity {value:.4f} against the selection's {chosen['score']:.4f}", flush=True)
    if abs(value - chosen["score"]) > tolerance:
        sys.exit(f"error: {arm['id']} reproduces {value:.4f}, the selection says {chosen['score']:.4f} (tolerance {tolerance}): stop")


def sync(out, drive, dry):
    if drive:
        run(["rsync", "-a", f"{out}/", f"{drive}/final/"], dry)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, default=Path("results/frozen_arms.json"))
    ap.add_argument("--spec-sha256-file", type=Path, default=Path("results/frozen_arms.sha256"))
    ap.add_argument("--selection", type=Path, default=Path("results/selection.json"))
    ap.add_argument("--manifest", type=Path, default=Path("data/MANIFEST.sha256"))
    ap.add_argument("--out", type=Path, default=Path("runs/final"))
    ap.add_argument("--models", type=Path, default=Path("final_models"))
    ap.add_argument("--drive")
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--skip-sanity", action="store_true", help="only to resume after the sanity gate passed once")
    ap.add_argument("--from-stage", help="resume at this stage name (earlier stages are not run)")
    ap.add_argument("--keep-cache", action="store_true", help="do not clear the Hugging Face cache or the merged models before a stage (resuming inside a stage)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    spec = json.loads((ROOT / args.spec).read_text(encoding="utf-8"))
    selection = json.loads((ROOT / args.selection).read_text(encoding="utf-8"))
    digest = (ROOT / args.spec_sha256_file).read_text(encoding="utf-8").split()[0]
    stages = plan(spec)
    common = ["--spec", str(args.spec), "--spec-sha256", digest, "--selection", str(args.selection)]
    adapter_ids = [a["id"] for a in spec["arms"] if a.get("adapter") and a.get("role") is None]
    if adapter_ids:
        if run([TRAIN_PY, "gpu/prepare_final.py", *common, "--models", str(args.models), "--only", ",".join(adapter_ids)], args.dry_run):
            sys.exit("error: prepare_final failed")
        if not args.skip_sanity:
            for arm in spec["arms"]:
                if arm["id"] in adapter_ids:
                    sanity(arm, selection, args.tolerance, args.dry_run)
    if args.from_stage and args.from_stage not in [n for n, _, _ in stages]:
        sys.exit(f"error: --from-stage {args.from_stage!r} is not a stage of this spec: {[n for n, _, _ in stages]}")
    started = args.from_stage is None
    for name, ids, clear in stages:
        started = started or name == args.from_stage
        if not started:
            continue
        print(f"== stage {name}: {', '.join(ids)}", flush=True)
        if clear and not args.dry_run and not args.keep_cache:
            shutil.rmtree(Path.home() / ".cache/huggingface/hub", ignore_errors=True)
            shutil.rmtree(args.models, ignore_errors=True)
        if run([VLLM_PY, "gpu/final_eval.py", *common, "--manifest", str(args.manifest), "--data", "data", "--out", str(args.out), "--only", ",".join(ids)], args.dry_run):
            sys.exit(f"error: stage {name} failed; start the same command again to resume")
        sync(args.out, args.drive, args.dry_run)
    print("locked evaluation sampling finished", flush=True)


if __name__ == "__main__":
    main()
