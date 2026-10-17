#!/usr/bin/env python3
"""C3, the reward-optimal constant control (docs/prereg-v1.md section 3): one molecule answered for every prompt, chosen ONLY from Train.

Candidates are the Train references plus, when data/raw/chembl_pool/pool.jsonl exists, the leakage-filtered ChEMBL pool (public approved drugs
outside the held-out components). The objective is the mean count-vector Tanimoto (rlexp.chem.tanimoto's fingerprint) to the Train references,
leaving out the candidate's own Train entry. This is what an RL policy that ignores the profile can at best learn, so RL has to beat it before
its delivered similarity counts as more than a better constant. Writes controls in the standard sample format for `rlexp.metrics score`.

  .venv/bin/python scripts/best_constant.py --splits val --out runs/colab-s1/val-control_constant_pool
"""
import argparse
import json
import sys
from pathlib import Path

from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from rlexp import chem, metrics

POOL = Path("data/raw/chembl_pool/pool.jsonl")
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def count_fp(smiles):
    mol = chem.standardize(smiles)
    return None if mol is None else _GEN.GetCountFingerprint(mol)


def read_rows(data_dir, split):
    with open(Path(data_dir) / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def references(rows):
    """{drug: ref_smiles}, the first reference of each drug."""
    out = {}
    for row in rows:
        out.setdefault(row["group"], row["ref_smiles"])
    return out


def choose(train_refs, pool_smiles=()):
    """(smiles, mean Tanimoto to the Train references, source) of the best candidate. A Train candidate never scores against itself."""
    train = {g: (s, count_fp(s)) for g, s in train_refs.items()}
    train = {g: v for g, v in train.items() if v[1] is not None}
    fps = [v[1] for v in train.values()]
    best = None
    candidates = [(s, f, g, "train") for g, (s, f) in train.items()] + [(s, count_fp(s), None, "pool") for s in pool_smiles]
    names = list(train)
    for smiles, fp, own, source in candidates:
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        if own is not None:
            sims = [x for g, x in zip(names, sims) if g != own]
        score = sum(sims) / len(sims)
        if best is None or score > best[1] or (score == best[1] and smiles < best[0]):
            best = (smiles, score, source)
    return best


def pool_smiles(path=POOL):
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        smiles = row.get("smiles") or row.get("parent_smiles") or row.get("canonical_smiles")
        if smiles:
            out.append(smiles)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--splits", default="val", help="comma-separated; Test splits only after the lock")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    train_refs = references(read_rows(args.data, "train"))
    smiles, score, source = choose(train_refs, pool_smiles())
    rows = [r for s in args.splits.split(",") for r in read_rows(args.data, s)]
    samples = metrics.answer_samples(rows, [smiles] * len(rows), "control_constant_pool")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics.write_jsonl(out / "samples.jsonl", samples)
    meta = {"control": "constant_pool", "objective": "mean count-vector Tanimoto to Train references", "train_score": round(score, 4),
            "source": source, "candidates": len(train_refs) + len(pool_smiles()), "splits": args.splits.split(","), "answer_chars": len(smiles)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"error: {type(e).__name__}: {e}")
