#!/usr/bin/env python3
"""Identification rates of the untrained models on Val (hypothesis H3, docs/prereg-v1.md section 3 and docs/paper/README.md).

Runs the `rlexp.ident` scorer on the four Val identification runs and writes rates only (no text) to results/ident_val.json.

  .venv/bin/python scripts/ident_val.py [--runs runs/colab-s1] [--data data/processed] [--out results/ident_val.json]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

RUNS = {"4B": "ident-4b", "9B": "ident-9b", "Qwen3.5-27B-FP8": "ident-27b", "Qwen3.8-27B-FP8": "ident-38"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/colab-s1")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results/ident_val.json")
    args = ap.parse_args()
    rows = {}
    for model, run in RUNS.items():
        cmd = [sys.executable, "-m", "rlexp.ident", "score", "--samples", str(Path(args.runs) / run / "samples.jsonl"), "--data", args.data]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()
        rows[model] = {"run": run, **json.loads(out[-1])}
    result = {"split": "val", "scorer": "rlexp.ident score (usable samples only; drug-balanced)", "rows": rows}
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for model, r in rows.items():
        pk = r["pass_k"]
        print(f"{model}: name_match@1 {r['name_match@1']:.4f}, name_pass@{pk} {r[f'name_pass@{pk}']:.4f}, answered {r['answered']:.3f}, samples {r['samples']}")


if __name__ == "__main__":
    main()
