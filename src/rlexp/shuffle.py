#!/usr/bin/env python3
"""Shuffled-profile control (docs/plan.md section 2 (a) and (d)): every drug is shown ANOTHER drug's profile and its samples are
scored against its own reference. A frozen arm that uses the profile loses delivered similarity (fv_tanimoto) and lift here; one
that only emits a good constant does not. Prompt length, section order and format are preserved because the prompts are real
profiles of other held-out drugs.

  python -m rlexp.shuffle write --data data/processed --out data/processed/arms/shuffle [--seed 0] [--splits val]

Per split: the drugs (sorted by name) are deranged by a seeded permutation with no fixed point and, whenever the split allows it,
no drug mapped into its own component (cluster_id). Row i of a drug (rows sorted by id) takes the prompt of row i mod n of the drug
it is mapped to. Every other field is kept, so gpu/sample_vllm.py --data out reads the files unchanged and the scorer sees the
original references. The mapping is written to <split>.meta.json (drug names are public ChEMBL names; no prompt text).
"""
import argparse
import random
import sys
from pathlib import Path

from rlexp import arms

SPLITS = ("val", "test_r", "test_t")


def derange(drugs, cluster_of, seed):
    """{drug: other drug}: a permutation of `drugs` without fixed points, and where possible with cluster_of[d] != cluster_of[image].
    Seeded and deterministic; raises ValueError for fewer than 2 drugs."""
    drugs = sorted(drugs)
    if len(drugs) < 2:
        raise ValueError("a shuffle needs at least two drugs")
    rng = random.Random(seed)
    best = None
    for _ in range(2000):
        image = drugs[:]
        rng.shuffle(image)
        fixed = sum(a == b for a, b in zip(drugs, image))
        same = sum(cluster_of[a] == cluster_of[b] for a, b in zip(drugs, image))
        if fixed == 0 and same == 0:
            return dict(zip(drugs, image))
        if fixed == 0 and (best is None or same < best[0]):
            best = (same, image)
    if best is None:                                  # a cyclic shift is always fixed-point free
        best = (None, drugs[1:] + drugs[:1])
    return dict(zip(drugs, best[1]))


def shuffle_rows(rows, seed):
    """The rows with `prompt` replaced as described in the module docstring. Returns (new rows, {drug: image drug})."""
    by_drug = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        by_drug.setdefault(row["group"], []).append(row)
    cluster_of = {d: rs[0].get("cluster_id", d) for d, rs in by_drug.items()}
    image = derange(by_drug, cluster_of, seed)
    out = []
    for d, rs in sorted(by_drug.items()):
        source = by_drug[image[d]]
        out += [{**r, "prompt": source[i % len(source)]["prompt"]} for i, r in enumerate(rs)]
    return out, image


def write(data_dir, out_dir, seed=0, splits=SPLITS):
    for split in splits:
        rows, image = shuffle_rows(arms._read(Path(data_dir) / f"{split}.jsonl"), seed)
        arms._write(Path(out_dir) / f"{split}.jsonl", rows, lines=True)
        arms._write(Path(out_dir) / f"{split}.meta.json", {"file": f"{split}.jsonl", "rows": len(rows), "control": "shuffled_profile",
                                                           "seed": seed, "mapping": image})
        print(f"{split}.jsonl: {len(rows)} rows, {len(image)} drugs shown another drug's profile")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="write the shuffled-profile prompt files")
    w.add_argument("--data", default="data/processed")
    w.add_argument("--out", default="data/processed/arms/shuffle")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--splits", default=",".join(SPLITS), help="comma-separated; Test splits only after the lock")
    a = ap.parse_args(argv)
    try:
        write(a.data, a.out, a.seed, tuple(s for s in a.splits.split(",") if s))
    except (ValueError, KeyError, OSError) as e:
        sys.exit(f"error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
