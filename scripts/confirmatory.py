#!/usr/bin/env python3
"""The one confirmatory contrast and its exploratory siblings (docs/prereg-v1.md section 4), run ONLY after the lock.

  .venv/bin/python scripts/confirmatory.py --selection results/selection.json --spec-sha256 <hex> --final runs/final --out results/confirmatory.json

For the pair chosen on Val (selection.json: "rl" and "comparator" arm ids) this scores nothing new: it reads the scored Test samples
(runs/final/<arm>/scored.v3.jsonl, made by `rlexp.metrics score`) and calls `rlexp.metrics report --scored RL --vs COMPARATOR` on test_t (the
confirmatory split) and test_r (exploratory), 10,000 bootstrap resamples, seed 0, then copies the paired fv_tanimoto block (wins, losses, ties, the cluster-level
sign-flip p-value) into one JSON with the command lines, the sha256 of the input files and of the spec, selection and run manifest. No arm, decoding or comparator is chosen here.

Two checks stand between the files and a result. Before pairing: <final>/manifest.json (written by gpu/final_eval.py) must carry --spec-sha256 (itself and
every arm entry) and the sha256 of --selection; each sampled arm of the pair needs an entry whose settings (model, mode, temperature, max_tokens, k and, when
recorded, data_sha256) equal the sampler's own <final>/<arm>/meta.json. A CPU control (null temperature in the selection) has no sampler: its
<final>/<arm>/meta.json, as scripts/final_controls.py writes it, must say control == the arm id, k == 8, splits test_t and test_r, rows == 180 and give a source_sha256
(C3 also a pool_sha256).
After pairing: the pair covers the whole cohort, test_t 41 drugs and 100 prompt rows, test_r 30 and 80, with fv_tanimoto defined for every drug. Otherwise the exit
is non-zero and no result file is written.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

CONFIRMATORY, EXPLORATORY = "test_t", "test_r"
COHORT = {CONFIRMATORY: (41, 100), EXPLORATORY: (30, 80)}           # (drugs, prompt rows) of the frozen Test splits
SAMPLER = ("model", "mode", "temperature", "max_tokens", "k")        # the sampler fields of a run manifest entry that meta.json must repeat
CONTROL_ROWS = sum(rows for _, rows in COHORT.values())              # 180 prompt rows with one answer each, replicated to k=8 by scripts/final_controls.py


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def commands(selection, final, data, out_dir, boot=10000, seed=0):
    """{split: command} for the paired reports of the chosen RL arm against the chosen comparator."""
    rl, comparator = selection["rl"], selection["comparator"]
    a, b = Path(final) / rl / "scored.v3.jsonl", Path(final) / comparator / "scored.v3.jsonl"
    return {split: [sys.executable, "-m", "rlexp.metrics", "report", "--scored", str(a), "--vs", str(b), "--split", split, "--data", str(data),
                    "--boot", str(boot), "--seed", str(seed), "--json", str(Path(out_dir) / f"paired_{split}.json")]
            for split in (CONFIRMATORY, EXPLORATORY)}


def check_control(final, arm):
    """The CPU control `arm` was made by scripts/final_controls.py for both Test splits (its meta.json); ValueError if not."""
    path = Path(final) / arm / "meta.json"
    if not path.is_file():
        raise ValueError(f"{path} is missing: the CPU control {arm} is made by scripts/final_controls.py")
    meta = json.loads(path.read_text(encoding="utf-8"))
    frozen = {"control": arm, "k": 8, "splits": [CONFIRMATORY, EXPLORATORY], "rows": CONTROL_ROWS}
    differs = sorted(key for key, value in frozen.items() if meta.get(key) != value)
    if differs:
        raise ValueError(f"control {arm}: {path} differs from {frozen} in {differs}")
    missing = [key for key in ("source_sha256", *(["pool_sha256"] if arm == "C3" else [])) if not meta.get(key)]
    if missing:
        raise ValueError(f"control {arm}: {path} records no {', '.join(missing)}")


def check_runs(selection, selection_path, final, spec_sha256):
    """Check that the pair to be reported was sampled under the frozen spec (see the module docstring); ValueError if not. Returns the run manifest's path."""
    path = Path(final) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"{path} is missing: the samples were not made by gpu/final_eval.py")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("arms")
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{path} lists no arms")
    if manifest.get("spec_sha256") != spec_sha256 or any(e.get("spec_sha256") != spec_sha256 for e in entries.values()):
        raise ValueError("the run manifest is not (entirely) from the spec given by --spec-sha256")
    if manifest.get("selection_sha256") != sha256(selection_path):
        raise ValueError("the selection file is not the one the spec was frozen against (the run manifest's selection_sha256)")
    for arm in (selection["rl"], selection["comparator"]):
        try:
            sampled = selection["arms"][arm]["temperature"] is not None
        except KeyError as e:
            raise ValueError(f"the selection lacks {e}") from None
        if not sampled:
            check_control(final, arm)                                                 # a CPU control has no sampler and no manifest entry
            continue
        entry = entries.get(arm)
        if entry is None:
            raise ValueError(f"arm {arm} has no entry in the run manifest: it was not sampled by gpu/final_eval.py under this spec")
        meta = Path(final) / arm / "meta.json"
        if not meta.is_file():
            raise ValueError(f"{meta} is missing")
        run = json.loads(meta.read_text(encoding="utf-8")).get("configuration") or {}
        if any(key not in entry for key in SAMPLER):
            raise ValueError(f"the run manifest entry of arm {arm} lacks some of {list(SAMPLER)}")
        frozen = {key: entry[key] for key in (*SAMPLER, *(["data_sha256"] if entry.get("data_sha256") else []))}
        differs = sorted(key for key, value in frozen.items() if run.get(key) != value)
        if differs:
            raise ValueError(f"arm {arm}: the run's meta.json differs from the run manifest in {differs}")
    return path


def check_cohort(split, report):
    """The paired block of a `rlexp.metrics report` JSON for `split` covers its whole cohort, with fv_tanimoto defined on every drug; else ValueError."""
    drugs, rows = COHORT[split]
    paired = (report.get("splits") or {}).get(split, {}).get("paired") or {}
    fv = (paired.get("metrics") or {}).get("fv_tanimoto") or {}
    got = (paired.get("n_drugs"), paired.get("n_rows"), fv.get("n_drugs"))
    if got != (drugs, rows, drugs) or fv.get("p_value") is None:
        raise ValueError(f"{split}: the pair covers {got[0]} drugs and {got[1]} prompt rows, fv_tanimoto {got[2]} drugs, p-value {fv.get('p_value')}; "
                         f"the cohort is {drugs} drugs and {rows} rows, all with fv_tanimoto and a p-value")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--spec-sha256", help="sha256 of the frozen spec (required unless --dry-run)")
    ap.add_argument("--final", type=Path, default=Path("runs/final"))
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.spec_sha256:
        ap.error("--spec-sha256 is required")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    cmds = commands(selection, args.final, args.data, args.out.parent, args.boot)
    if args.dry_run:
        print(json.dumps(cmds, indent=1))
        return
    try:
        manifest_path = check_runs(selection, args.selection, args.final, args.spec_sha256)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    inputs = {}
    result = {"rl": selection["rl"], "comparator": selection["comparator"], "confirmatory_split": CONFIRMATORY, "commands": cmds, "paired": {},
              "spec_sha256": args.spec_sha256, "selection_sha256": sha256(args.selection), "run_manifest_sha256": sha256(manifest_path)}
    for split, cmd in cmds.items():
        for path in (cmd[cmd.index("--scored") + 1], cmd[cmd.index("--vs") + 1]):
            if not Path(path).is_file():
                sys.exit(f"error: missing scored file {path}: run rlexp.metrics score on the final samples first")
            inputs[path] = sha256(path)
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode:
            sys.exit(f"error: report failed for {split}: {done.stderr.strip()[-300:]}")
        paired = json.loads(Path(cmd[cmd.index("--json") + 1]).read_text(encoding="utf-8"))
        try:
            check_cohort(split, paired)
        except ValueError as e:
            sys.exit(f"error: {e}")
        result["paired"][split] = paired
        (args.out.parent / f"paired_{split}.txt").write_text(done.stdout, encoding="utf-8")
    result["input_sha256"] = inputs
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {selection['rl']} against {selection['comparator']} on {CONFIRMATORY} (confirmatory) and {EXPLORATORY} (exploratory)")


if __name__ == "__main__":
    main()
