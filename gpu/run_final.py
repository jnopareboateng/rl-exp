#!/usr/bin/env python3
"""Operator script for the locked evaluation on the Colab A100 (gpu/FINAL.md section 1): merge the frozen adapters, check on Val that each merged model reproduces its Val score,
then sample every arm of the frozen spec in a fixed order, clearing the Hugging Face cache between model families and syncing runs/final to Drive after each stage.

  python gpu/run_final.py --spec results/frozen_arms.json --selection results/selection.json --manifest data/MANIFEST.sha256 [--drive /content/drive/MyDrive/rl-exp] [--dry-run]

The spec hash is read from `<spec>.sha256`-style file `results/frozen_arms.sha256` (--spec-sha256-file). The order is fixed in STAGES below and holds no choice: each stage is a list of arm ids that
gpu/final_eval.py samples under the frozen spec; an arm of the spec that no stage names stops the script before anything runs, so no frozen arm is skipped silently. A stage whose ids are not all in the spec
runs the ones that are (an arm dropped before the freeze is an omission recorded in the protocol). When the run ends, every arm of the plan must be in <out>/manifest.json, else the exit is 1 and
"finished" is not printed (a --from-stage that skipped an unfinished stage cannot pass for a complete run).

The Val sanity gate. prepare_final and the gate run only when a stage that still has to run (from --from-stage on) holds an adapter arm. For each such arm the gate calls gpu/sample_vllm.py on Val at
its frozen decoding (seed 0) and compares the first-valid similarity with the selection's score: more than final_eval.SANITY_TOLERANCE apart stops the script (the model on disk is not the selected
one). Every attempt is appended to <out>/sanity.json, {"tolerance", "attempts": [{arm, value, selection_score, passed, time}]}; nothing is overwritten, a failure stays on record, and the latest attempt
of an arm decides (scripts/confirmatory.py requires it passed and prints the attempts). --skip-sanity runs no gate: it needs a passed latest attempt of each such arm already in sanity.json and keeps it.
A stale .agent-runs/sanity/<id> is deleted before its sample (the sampler would resume it and report the old samples).

Drive. With --drive the path must exist (mount Drive first). If <out> holds no arm directory while <drive>/final holds content, an earlier runtime was lost: that content is copied back first
(otherwise the next sync would replace the longer partial files on Drive with the empty ones here); a file already in <out> is not overwritten by it. Every copy to Drive that fails is reported;
the last one aborts the script.
Needs both venvs of gpu/bootstrap_colab.sh.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from final_eval import SANITY_TOLERANCE      # noqa: E402  the one definition of the gate's tolerance

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


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def hub_cache():
    """The Hugging Face hub cache the downloads of this runtime go to (huggingface_hub's own HF_HUB_CACHE, which honours HF_HOME and HF_HUB_CACHE), else ~/.cache/huggingface/hub."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return Path.home() / ".cache/huggingface/hub"
    return Path(HF_HUB_CACHE)


def run(cmd, dry, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd, cwd=ROOT, **kw).returncode


def sanity(arm, selection, dry):
    """Sample a merged adapter arm on Val at its frozen decoding and compare its first-valid similarity with the selection's score. Returns the attempt for sanity.json
    ({"arm", "value", "selection_score", "passed", "time"}); None in a dry run. The sampler runs with its default seed 0."""
    chosen = selection["arms"][arm["id"]]
    if not dry:
        shutil.rmtree(SANITY / arm["id"], ignore_errors=True)          # a stale directory would be resumed (same path identity) and its old samples reported
    cmd = [VLLM_PY, "gpu/sample_vllm.py", "--model", arm["model"], "--data", "data/processed", "--splits", "val", "--mode", arm["mode"], "--k", "8", "--max-tokens", str(arm["max_tokens"]),
           "--temperature", str(arm["temperature"]), "--prompts-per-batch", "auto", "--out", str(SANITY / arm["id"]), "--system-text", DIRECT]
    if run(cmd, dry):
        sys.exit(f"error: the Val sanity sample of {arm['id']} failed")
    if dry:
        return None
    samples = SANITY / arm["id"] / "samples.jsonl"
    env = {**os.environ, "PYTHONPATH": "src"}
    scored, report = samples.with_name("scored.v3.jsonl"), samples.with_name("report.v3.json")
    for args in (["score", "--samples", str(samples), "--data", "data/processed", "--out", str(scored)],
                 ["report", "--scored", str(scored), "--split", "val", "--data", "data/processed", "--boot", "100", "--json", str(report)]):
        if run([TRAIN_PY, "-m", "rlexp.metrics", *args], False, env=env, capture_output=True):
            sys.exit(f"error: scoring the Val sanity sample of {arm['id']} failed")
    value = json.loads(report.read_text(encoding="utf-8"))["splits"]["val"]["metrics"]["fv_tanimoto"]
    value = value["value"] if isinstance(value, dict) else value
    print(f"sanity {arm['id']}: Val first-valid similarity {value:.4f} against the selection's {chosen['score']:.4f} (tolerance {SANITY_TOLERANCE})", flush=True)
    return {"arm": arm["id"], "value": value, "selection_score": chosen["score"], "passed": abs(value - chosen["score"]) <= SANITY_TOLERANCE, "time": now()}


def read_sanity(out):
    """<out>/sanity.json as {"tolerance", "attempts"} (new and empty without the file). ValueError if it was made with another tolerance than the gate's."""
    path = Path(out) / "sanity.json"
    if not path.is_file():
        return {"tolerance": SANITY_TOLERANCE, "attempts": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("tolerance") != SANITY_TOLERANCE or not isinstance(data.get("attempts"), list):
        raise ValueError(f"{path} was made with tolerance {data.get('tolerance')}, the gate's is {SANITY_TOLERANCE}: it is kept as it is; use a new --out")
    return data


def last_attempt(data, arm_id):
    return next((a for a in reversed(data["attempts"]) if a.get("arm") == arm_id), None)


def append_sanity(out, attempt):
    """Append `attempt` to <out>/sanity.json: every attempt of every arm stays, the latest of an arm decides."""
    data = read_sanity(out)
    data["attempts"].append(attempt)
    path = Path(out) / "sanity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def restore_from_drive(out, drive, dry):
    """If <out> holds no arm directory while <drive>/final holds content, copy that back first and say so. True if it did (or, in a dry run, would). A file already in <out> is kept
    (--ignore-existing): a gate record written here since the last copy to Drive has more attempts than Drive's."""
    final, here = Path(drive) / "final", ROOT / out
    if (here.is_dir() and any(p.is_dir() for p in here.iterdir())) or not (final.is_dir() and any(final.iterdir())):
        return False
    print(f"{out} holds no arm directory but {final} holds an earlier run: restoring it from Drive before anything else", flush=True)
    code = run(["rsync", "-a", "--ignore-existing", f"{final}/", f"{out}/"], dry)
    if code:
        sys.exit(f"error: restoring {out} from {final} failed (rsync code {code}); nothing was sampled")
    return True


def sync(out, drive, dry, last=False):
    """Copy <out> to <drive>/final. A failure is reported; the last copy's aborts the script (the Test samples must be on Drive before the runtime ends)."""
    if not drive:
        return
    try:
        code = run(["rsync", "-a", f"{out}/", f"{drive}/final/"], dry)
    except OSError as e:
        code = type(e).__name__
    if code:
        message = f"the copy of {out} to {drive}/final failed (rsync code {code})"
        if last:
            sys.exit(f"error: {message}: copy it by hand before the runtime ends")
        print(f"WARNING: {message}", file=sys.stderr, flush=True)


def sync_forever(out, drive, every, stop):
    """Copy runs/final to Drive every `every` seconds (a lost runtime keeps the finished prompts of a long arm: gpu/sample_vllm.py resumes from them). A failed copy is reported."""
    while not stop.wait(every):
        try:
            code = subprocess.run(["rsync", "-a", f"{out}/", f"{drive}/final/"], cwd=ROOT, capture_output=True).returncode
        except OSError as e:
            code = type(e).__name__
        if code:
            print(f"WARNING: the periodic copy of {out} to {drive}/final failed (rsync code {code})", file=sys.stderr, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, default=Path("results/frozen_arms.json"))
    ap.add_argument("--spec-sha256-file", type=Path, default=Path("results/frozen_arms.sha256"))
    ap.add_argument("--selection", type=Path, default=Path("results/selection.json"))
    ap.add_argument("--manifest", type=Path, default=Path("data/MANIFEST.sha256"))
    ap.add_argument("--out", type=Path, default=Path("runs/final"))
    ap.add_argument("--models", type=Path, default=Path("final_models"))
    ap.add_argument("--drive")
    ap.add_argument("--sync-every", type=float, default=180.0, help="seconds between background copies of runs/final to --drive")
    ap.add_argument("--skip-sanity", action="store_true", help="run no Val gate: each adapter arm still to be sampled needs a passed latest attempt in <out>/sanity.json, which is kept")
    ap.add_argument("--from-stage", help="resume at this stage name (earlier stages are not run)")
    ap.add_argument("--keep-cache", action="store_true", help="do not clear the Hugging Face cache or the merged models before the first stage that runs (resuming inside a stage); later stages still clear")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    spec = json.loads((ROOT / args.spec).read_text(encoding="utf-8"))
    selection = json.loads((ROOT / args.selection).read_text(encoding="utf-8"))
    digest = (ROOT / args.spec_sha256_file).read_text(encoding="utf-8").split()[0]
    stages = plan(spec)
    names = [n for n, _, _ in stages]
    if args.from_stage and args.from_stage not in names:
        sys.exit(f"error: --from-stage {args.from_stage!r} is not a stage of this spec: {names}")
    pending = stages[names.index(args.from_stage):] if args.from_stage else stages
    common = ["--spec", str(args.spec), "--spec-sha256", digest, "--selection", str(args.selection)]
    if args.drive:
        if not Path(args.drive).is_dir():
            sys.exit(f"error: --drive {args.drive} does not exist: mount Drive first")
        restore_from_drive(args.out, args.drive, args.dry_run)
    ahead = {i for _, ids, _ in pending for i in ids}
    needed = {a.get("of", a["id"]) if a.get("role") == "diagnostic" else a["id"] for a in spec["arms"] if a.get("adapter") and a["id"] in ahead}      # a diagnostic samples its base arm's export
    adapter_ids = [a["id"] for a in spec["arms"] if a.get("adapter") and a["id"] in needed]            # a diagnostic is not in `needed`: its base arm is
    if adapter_ids:
        if run([TRAIN_PY, "gpu/prepare_final.py", *common, "--models", str(args.models), "--only", ",".join(adapter_ids)], args.dry_run):
            sys.exit("error: prepare_final failed")
        try:
            if args.skip_sanity:
                recorded = read_sanity(ROOT / args.out)
                unchecked = [i for i in adapter_ids if (last_attempt(recorded, i) or {}).get("passed") is not True]
                if unchecked and not args.dry_run:
                    sys.exit(f"error: --skip-sanity needs a passed Val check of {unchecked} in {args.out}/sanity.json already: run the gate (leave --skip-sanity out)")
            else:
                for arm in spec["arms"]:
                    if arm["id"] not in adapter_ids:
                        continue
                    attempt = sanity(arm, selection, args.dry_run)
                    if attempt is not None:
                        append_sanity(ROOT / args.out, attempt)        # a failure stays on record
                        if not attempt["passed"]:
                            sys.exit(f"error: {arm['id']} reproduces {attempt['value']:.4f}, the selection says {attempt['selection_score']:.4f} (tolerance {SANITY_TOLERANCE}): stop")
        except ValueError as e:
            sys.exit(f"error: {e}")
    stop = threading.Event()
    if args.drive and not args.dry_run:
        threading.Thread(target=sync_forever, args=(args.out, args.drive, args.sync_every, stop), daemon=True).start()
    for index, (name, ids, clear) in enumerate(pending):
        print(f"== stage {name}: {', '.join(ids)}", flush=True)
        if clear and not args.dry_run and not (args.keep_cache and index == 0):         # --keep-cache protects the stage being resumed, every later family starts clean
            shutil.rmtree(hub_cache(), ignore_errors=True)
            shutil.rmtree(args.models, ignore_errors=True)
        if run([VLLM_PY, "gpu/final_eval.py", *common, "--manifest", str(args.manifest), "--data", "data", "--out", str(args.out), "--only", ",".join(ids)], args.dry_run):
            sys.exit(f"error: stage {name} failed; resume it with the same command plus --skip-sanity --from-stage {name!r} --keep-cache (the gate has passed and nothing is merged again)")
        sync(args.out, args.drive, args.dry_run)
    stop.set()
    sync(args.out, args.drive, args.dry_run, last=True)
    if not args.dry_run:
        manifest = ROOT / args.out / "manifest.json"
        done = set(json.loads(manifest.read_text(encoding="utf-8")).get("arms", {})) if manifest.is_file() else set()
        missing = [i for _, ids, _ in stages for i in ids if i not in done]
        if missing:
            sys.exit(f"error: not finished: {len(missing)} arm(s) of the plan are not in {args.out}/manifest.json: {missing}. Start the same command again (a stage that stopped resumes).")
    print("locked evaluation sampling finished", flush=True)


if __name__ == "__main__":
    main()
