#!/usr/bin/env python3
"""Independent recomputation of the confirmatory contrast from two scored files, with numpy and json only (no rlexp code), as a check on results/confirmatory.json.

  .venv/bin/python scripts/recompute_confirmatory.py --final runs/final --a RL-full --b C3 --split test_t --allow-test --out results/confirmatory_recompute.json

Endpoint: per prompt row the Tanimoto of the first sample (by sample_idx, first eight) that is valid, closed and finished, else 0; mean over a drug's rows, then over drugs.
Test: two-sided sign-flip of the per-drug differences (each drug its own unit: the Test-T drugs lie in 41 components), Monte Carlo with its own seed; bootstrap over drugs.
Aggregates only.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def per_drug(path, split):
    by = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["split"] == split:
            by.setdefault((r["group"], r["id"]), []).append(r)
    drug = {}
    for (g, _), samples in by.items():
        value = 0.0
        for s in sorted(samples, key=lambda s: s["sample_idx"])[:8]:
            if s["valid"] and s["closed"] and s.get("finish_reason", "stop") == "stop":
                value = float(s["tanimoto"])
                break
        drug.setdefault(g, []).append(value)
    return {g: float(np.mean(v)) for g, v in drug.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", type=Path, default=Path("runs/final"))
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--split", default="test_t")
    ap.add_argument("--flips", type=int, default=400_000)
    ap.add_argument("--boot", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    if args.split.startswith("test") and not args.allow_test:
        raise SystemExit("refused: a Test split needs --allow-test (locked evaluation, after the prereg-v1 tag)")
    a, b = per_drug(args.final / args.a / "scored.v3.jsonl", args.split), per_drug(args.final / args.b / "scored.v3.jsonl", args.split)
    drugs = sorted(set(a) & set(b))
    d = np.array([a[g] - b[g] for g in drugs])
    rng = np.random.default_rng(args.seed)
    stat = np.abs((rng.choice([-1.0, 1.0], size=(args.flips, len(d))) * d).sum(axis=1))
    p = float((1 + (stat >= abs(d.sum()) - 1e-12).sum()) / (args.flips + 1))
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.boot)])
    out = {"a": args.a, "b": args.b, "split": args.split, "n_drugs": len(drugs), "mean_a": float(np.mean([a[g] for g in drugs])), "mean_b": float(np.mean([b[g] for g in drugs])),
           "difference": float(d.mean()), "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()), "ties": int((np.abs(d) <= 1e-9).sum()),
           "sign_flip_p": p, "flips": args.flips, "bootstrap_ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])], "independent_of": "rlexp.metrics (numpy and json only)"}
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{args.a} minus {args.b} on {args.split}: {out['difference']:+.4f} ({out['wins']} wins, {out['losses']} losses), sign-flip p {p:.4f}, interval {out['bootstrap_ci'][0]:+.4f} to {out['bootstrap_ci'][1]:+.4f}")


if __name__ == "__main__":
    main()
