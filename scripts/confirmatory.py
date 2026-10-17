#!/usr/bin/env python3
"""The one confirmatory contrast and its exploratory siblings (docs/prereg-v1.md section 4), run ONLY after the lock.

  .venv/bin/python scripts/confirmatory.py --selection results/selection.json --final runs/final --out results/confirmatory.json

For the pair chosen on Val (selection.json: "rl" and "comparator" arm ids) this scores nothing new: it reads the scored Test samples
(runs/final/<arm>/scored.v3.jsonl, made by `rlexp.metrics score`) and calls `rlexp.metrics report --scored RL --vs COMPARATOR` on test_t (the
confirmatory split) and test_r (exploratory), 10,000 bootstrap resamples, seed 0, then copies the paired fv_tanimoto block (wins, losses, ties, the cluster-level
sign-flip p-value) into one JSON with the command lines and the sha256 of the input files. No arm, decoding or comparator is chosen here.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

CONFIRMATORY, EXPLORATORY = "test_t", "test_r"


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--final", type=Path, default=Path("runs/final"))
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    cmds = commands(selection, args.final, args.data, args.out.parent, args.boot)
    if args.dry_run:
        print(json.dumps(cmds, indent=1))
        return
    inputs = {}
    result = {"rl": selection["rl"], "comparator": selection["comparator"], "confirmatory_split": CONFIRMATORY, "commands": cmds, "paired": {}}
    for split, cmd in cmds.items():
        for path in (cmd[cmd.index("--scored") + 1], cmd[cmd.index("--vs") + 1]):
            if not Path(path).is_file():
                sys.exit(f"error: missing scored file {path}: run rlexp.metrics score on the final samples first")
            inputs[path] = sha256(path)
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode:
            sys.exit(f"error: report failed for {split}: {done.stderr.strip()[-300:]}")
        paired = json.loads(Path(cmd[cmd.index("--json") + 1]).read_text(encoding="utf-8"))
        result["paired"][split] = paired
        (args.out.parent / f"paired_{split}.txt").write_text(done.stdout, encoding="utf-8")
    result["input_sha256"] = inputs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {selection['rl']} against {selection['comparator']} on {CONFIRMATORY} (confirmatory) and {EXPLORATORY} (exploratory)")


if __name__ == "__main__":
    main()
