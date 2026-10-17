#!/usr/bin/env python3
"""How good can ANY constant answer be on a split? The oracle constant: the candidate (Train references plus the leakage-filtered ChEMBL pool of C3) with the highest mean
count-vector Tanimoto to the split's own references, drug-balanced. It reads reference structures only (no model output) and says how much headroom a constant has over C3, which is chosen from Train alone.

  .venv/bin/python scripts/constant_ceiling.py --splits val [--out results/constant_ceiling.json]      # Test splits need --allow-test (after the tag)

Prints counts and similarities only, never a structure.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

from rdkit import DataStructs

HERE = Path(__file__).resolve().parent


def load_best_constant():
    spec = importlib.util.spec_from_file_location("best_constant", HERE / "best_constant.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["best_constant"] = module
    spec.loader.exec_module(module)
    return module


def ceiling(bc, refs, candidates, c3_smiles):
    """{oracle, c3, c3_rank, top5, candidates}: mean Tanimoto of every candidate to the references of one split."""
    fps = [f for f in (bc.count_fp(s) for s in refs.values()) if f is not None]
    scored = []
    for smiles in candidates:
        fp = bc.count_fp(smiles)
        if fp is not None:
            scored.append((sum(DataStructs.BulkTanimotoSimilarity(fp, fps)) / len(fps), smiles))
    scored.sort(key=lambda t: (-t[0], t[1]))
    rank = next((i + 1 for i, (_, s) in enumerate(scored) if s == c3_smiles), None)
    return {"oracle": scored[0][0], "c3": None if rank is None else scored[rank - 1][0], "c3_rank": rank, "top5": [round(x[0], 4) for x in scored[:5]], "candidates": len(scored), "drugs": len(fps)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--splits", default="val")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    splits = args.splits.split(",")
    if any(s.startswith("test") for s in splits) and not args.allow_test:
        sys.exit("error: a Test split needs --allow-test (after the tag prereg-v1)")
    bc = load_best_constant()
    train_refs = bc.references(bc.read_rows(args.data, "train"))
    pool_rows, pool_sha256 = bc.read_pool()
    pool = [bc.row_smiles(r) for r in pool_rows if bc.row_smiles(r)]
    c3_smiles = bc.choose(train_refs, pool)[0]
    candidates = list(dict.fromkeys([*train_refs.values(), *pool]))
    result = {"pool_sha256": pool_sha256, "splits": {s: ceiling(bc, bc.references(bc.read_rows(args.data, s)), candidates, c3_smiles) for s in splits}}
    for s, r in result["splits"].items():
        print(f"{s}: oracle constant {r['oracle']:.4f}, C3 {r['c3']:.4f} (rank {r['c3_rank']} of {r['candidates']}), {r['drugs']} drugs")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
