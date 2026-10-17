#!/usr/bin/env python3
"""Strict-reader sensitivity analysis (docs/prereg-v1.md section 8): rewrite every sample so that the unchanged scorer reads exactly what the strict reader reads
("the last line after the last </think> that is a SMILES by itself"), then score as usual. The main analysis keeps the lenient reader the RL arms were trained on.

  .venv/bin/python scripts/strict_rescore.py --samples runs/X/samples.jsonl --out runs/X-strict/samples.jsonl
  .venv/bin/python -m rlexp.metrics score --samples runs/X-strict/samples.jsonl --data data/processed --out runs/X-strict/scored.v3.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_audit import last_line_answer  # noqa: E402


def strict_text(text):
    """The text the scorer must see for the strict reader: a closed think block and, if the strict reader finds one, only its answer on its own line."""
    answer = last_line_answer(text)
    return "</think>\n" + (answer if answer is not None else "")


def rewrite(rows):
    """Rows with `text` replaced by strict_text; everything else (finish_reason, n_tokens, ids) unchanged."""
    return [{**row, "text": strict_text(row["text"])} for row in rows]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = rewrite(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    print(f"{len(rows)} samples rewritten to the strict reader; {sum(last_line_answer(r['text']) is None for r in rows)} have no strict answer")


if __name__ == "__main__":
    main()
