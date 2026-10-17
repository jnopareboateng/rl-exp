#!/usr/bin/env python3
"""Turn a one-answer-per-row control (C0 to C3) into k identical samples per row, so the first-valid metrics (which need k >= 8 per row) are defined and
equal the mean Tanimoto of the deterministic answer.

  .venv/bin/python scripts/replicate_samples.py --samples runs/X/samples.jsonl --k 8 --out runs/X8/samples.jsonl
"""
import argparse
import json
import sys
from pathlib import Path


def replicate(rows, k):
    """Each row k times with sample_idx 0..k-1 (the rest of the row unchanged). ValueError for a row without an integer sample_idx, one whose sample_idx is not 0
    (several samples per row already), and a repeated (id, sample_idx) pair (the same id twice at index 0, which the metrics would reject as duplicate samples)."""
    out, seen = [], set()
    for n, row in enumerate(rows, 1):
        idx = row.get("sample_idx")
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise ValueError(f"row {n} has no integer sample_idx")
        if idx != 0:
            raise ValueError("input already has several samples per row")
        if row.get("id") in seen:
            raise ValueError(f"row {n} repeats the (id, sample_idx) pair of an earlier row: one answer per id is expected")
        seen.add(row.get("id"))
        out += [{**row, "sample_idx": i} for i in range(k)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.k < 1:
        sys.exit("error: k must be positive")
    try:
        rows = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]
        out = replicate(rows, args.k)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    print(f"{len(rows)} rows -> {len(out)} samples ({args.k} per row)")


if __name__ == "__main__":
    main()
