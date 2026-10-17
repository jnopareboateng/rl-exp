#!/usr/bin/env python3
"""Stage 1: audit the SFT file before building anything on it.

Run on the machine that has the data:
    python scripts/audit_data.py data/raw/train_think.jsonl

It prints one block of plain text. Paste the block back into the thread.
It never prints whole prompts or completions: only counts, lengths, and the
last 300 characters of two completions per task type (that is where the final
answer lives, and it is what the reward extractor has to parse).

What each section is for:
  SHAPE      are the fields and counts what we think they are
  SPLIT KEYS how many distinct drugs, so we can split train/val/test by drug
  LENGTHS    how long rollouts will be, which sets memory use and max length
  FORMAT     do prompts end where generation starts; does text = prompt + completion
  ANSWERS    can we pull a molecule out of the SFT completions at all
"""
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rlexp.chem import extract_answer as _extract_smiles  # same extractor the reward uses

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except Exception:
    HAVE_RDKIT = False


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main(path):
    rows = []
    bad_lines = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1

    print("=== SHAPE ===")
    print(f"rows: {len(rows)}   unparseable lines: {bad_lines}")
    keys = Counter(k for r in rows for k in r)
    print("fields (count of rows that have each):", dict(keys))
    print("corpus:", dict(Counter(r.get("corpus") for r in rows)))
    print("task_type:", dict(Counter(r.get("task_type") for r in rows)))

    print("\n=== SPLIT KEYS ===")
    per_group = Counter(r.get("group") for r in rows)
    sizes = sorted(per_group.values(), reverse=True)
    print(f"distinct group values: {len(per_group)}")
    print(f"rows per group: max {sizes[0]}, median {st.median(sizes)}, "
          f"groups with 1 row: {sum(1 for s in sizes if s == 1)}")
    print("largest groups:", per_group.most_common(5))
    drug_ne_group = sum(1 for r in rows if r.get("drug") != r.get("group"))
    print(f"rows where drug != group: {drug_ne_group}")

    print("\n=== LENGTHS (tokens, as recorded in the file) ===")
    for name in ("n_prompt_tokens", "n_completion_tokens", "n_tokens"):
        xs = [r[name] for r in rows if isinstance(r.get(name), int)]
        if xs:
            print(f"{name}: min {min(xs)}  p50 {pct(xs, .5)}  p90 {pct(xs, .9)}  "
                  f"p99 {pct(xs, .99)}  max {max(xs)}")
    by_task = defaultdict(list)
    for r in rows:
        if isinstance(r.get("n_completion_tokens"), int):
            by_task[r.get("task_type")].append(r["n_completion_tokens"])
    for t, xs in sorted(by_task.items(), key=lambda kv: str(kv[0])):
        print(f"  completion tokens, task {t}: n={len(xs)} p50 {pct(xs, .5)} p90 {pct(xs, .9)} max {max(xs)}")

    print("\n=== FORMAT ===")
    n_think = sum(1 for r in rows if r.get("prompt", "").rstrip().endswith("<think>"))
    n_text_ok = sum(1 for r in rows if r.get("text", "").startswith(r.get("prompt", "\0")))
    n_close = sum(1 for r in rows if "</think>" in r.get("completion", ""))
    print(f"prompt ends with '<think>': {n_think}/{len(rows)}")
    print(f"text starts with prompt:    {n_text_ok}/{len(rows)}")
    print(f"completion contains '</think>': {n_close}/{len(rows)}")
    dup = len(rows) - len({r.get("prompt") for r in rows})
    print(f"duplicate prompts: {dup}")

    print("\n=== ANSWERS (can the reward extractor read the SFT answers?) ===")
    stats = defaultdict(lambda: Counter())
    tails = defaultdict(list)
    for r in rows:
        t = r.get("task_type")
        comp = r.get("completion", "")
        smi = _extract_smiles(comp)
        stats[t]["rows"] += 1
        if smi:
            stats[t]["extracted"] += 1
            if HAVE_RDKIT and Chem.MolFromSmiles(smi) is not None:
                stats[t]["rdkit_valid"] += 1
        if len(tails[t]) < 2:
            tails[t].append(comp[-300:])
    print("rdkit available:", HAVE_RDKIT)
    for t, c in sorted(stats.items(), key=lambda kv: str(kv[0])):
        print(f"  task {t}: rows {c['rows']}  extracted {c['extracted']}  rdkit_valid {c['rdkit_valid']}")
    print("\n--- last 300 chars of two completions per task type ---")
    for t, ts in sorted(tails.items(), key=lambda kv: str(kv[0])):
        for i, tail in enumerate(ts):
            print(f"[task {t} #{i}] ...{tail!r}")


if __name__ == "__main__":
    main(sys.argv[1])
