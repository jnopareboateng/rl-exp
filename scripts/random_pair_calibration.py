#!/usr/bin/env python3
"""What a similarity score means: the distribution of count-vector Tanimoto between randomly paired approved drugs (docs/prereg-v1-addendum1.md, S4).

  .venv/bin/python scripts/random_pair_calibration.py --refs data/processed/refs.jsonl --out results/random_pair_calibration.json [--md results/random_pair_calibration.md]

Every pair of distinct reference structures of the splits named by --splits (Train, Val, Test-R and Test-T by default: the drugs the study uses) is scored with
`rlexp.chem.tanimoto`, the same count-vector Morgan similarity as the endpoint, and the distribution is summarised. Structures already identical by connectivity key
are counted once, so a duplicated drug cannot inject a similarity of 1. `--at` then reports where given scores sit in that distribution: the share of random pairs of
approved drugs that are at least that similar to each other.

This reads reference structures only. It touches no model output, no prompt file and no sample, so it is not an evaluation of any arm and says nothing about any split's
results: it is a property of the drug population, computable before or after any freeze (docs/prereg-v1.md section 3 allows reference-only computations on every split).
It exists because a graded similarity needs a scale: a delivered similarity means little until one knows what two unrelated approved drugs score. Only counts and
aggregate statistics are printed; no structure and no drug name is written.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts/ is not a package

from rlexp import chem, metrics      # noqa: E402

USED_SPLITS = ("train", "val", "test_r", "test_t")
PERCENTILES = (5, 25, 50, 75, 90, 95, 97.5, 99, 99.5, 99.9)
DEFAULT_AT = (0.176, 0.1865, 0.1949, 0.1954, 0.28, 0.31)


def structures(rows, splits):
    """The distinct reference structures (count fingerprints as mols) of those splits, one per connectivity key."""
    seen, mols = set(), []
    for r in rows:
        if r.get("split") not in splits or not r.get("ref_smiles"):
            continue
        mol = chem.standardize(r["ref_smiles"])
        if mol is None:
            raise metrics.InputError("a reference structure does not parse: the calibration would be over an incomplete population")
        key = r.get("ref_conn") or chem.conn(mol)
        if key not in seen:
            seen.add(key)
            mols.append(mol)
    if len(mols) < 2:
        raise metrics.InputError(f"fewer than two distinct reference structures in {sorted(splits)}: nothing to pair")
    return mols


def pair_similarities(mols):
    """Every unordered pair once, as a 1-D array of length n(n-1)/2."""
    return np.array([chem.tanimoto(mols[i], mols[j]) for i in range(len(mols)) for j in range(i + 1, len(mols))], dtype=float)


def summary(sims, at=DEFAULT_AT):
    return {"n_pairs": int(sims.size), "mean": float(sims.mean()), "sd": float(sims.std(ddof=1)), "min": float(sims.min()), "max": float(sims.max()),
            "percentiles": {str(p): float(np.percentile(sims, p)) for p in PERCENTILES},
            "at": {f"{v:.4f}": {"percentile": float(100 * (sims < v).mean()), "share_at_or_above": float(100 * (sims >= v).mean())} for v in at}}


def markdown(res):
    out = [f"# What a similarity score means: random pairs of approved drugs", "",
           f"Count-vector Morgan Tanimoto (the endpoint's own similarity) between every pair of the {res['n_structures']} distinct reference structures of "
           f"{', '.join(res['splits'])} ({res['n_pairs']:,} pairs): mean {res['mean']:.3f}, standard deviation {res['sd']:.3f}.", "",
           "| Percentile | Similarity |", "|---|---:|"]
    out += [f"| {p} | {v:.3f} |" for p, v in res["percentiles"].items()]
    out += ["", "| Score | Percentile of random pairs | Share of random pairs at least this similar |", "|---|---:|---:|"]
    out += [f"| {s} | {e['percentile']:.1f} | {e['share_at_or_above']:.1f}% |" for s, e in res["at"].items()]
    out += ["", "A delivered similarity should be read against this scale: a score whose percentile is low is a score many unrelated pairs of approved drugs reach.", ""]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", type=Path, default=Path("data/processed/refs.jsonl"))
    ap.add_argument("--splits", default=",".join(USED_SPLITS), help="the splits whose drugs form the population")
    ap.add_argument("--at", default=",".join(str(v) for v in DEFAULT_AT), help="scores to locate in the distribution")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--md", type=Path)
    args = ap.parse_args(argv)
    splits = tuple(s for s in args.splits.split(",") if s)
    try:
        at = tuple(float(v) for v in args.at.split(",") if v)
    except ValueError:
        sys.exit(f"error: --at must be comma-separated numbers, got {args.at!r}")
    if not args.refs.is_file():
        sys.exit(f"error: {args.refs} does not exist")
    mols = structures(metrics.read_jsonl(args.refs), set(splits))
    sims = pair_similarities(mols)
    res = {"refs": str(args.refs), "splits": list(splits), "n_structures": len(mols), **summary(sims, at)}
    print(f"{res['n_structures']} distinct reference structures in {', '.join(splits)}; {res['n_pairs']:,} pairs; "
          f"mean {res['mean']:.4f}, median {res['percentiles']['50']:.4f}, p90 {res['percentiles']['90']:.4f}, p99 {res['percentiles']['99']:.4f}")
    for score, entry in res["at"].items():
        print(f"  {score}: percentile {entry['percentile']:.2f}, {entry['share_at_or_above']:.2f}% of random pairs are at least this similar")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(markdown(res), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
