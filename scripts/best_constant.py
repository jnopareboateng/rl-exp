#!/usr/bin/env python3
"""C3, the reward-optimal constant control (docs/prereg-v1.md section 3): one molecule answered for every prompt, chosen ONLY from Train.

Candidates are the Train references plus, when data/raw/chembl_pool/pool.jsonl exists, the leakage-filtered ChEMBL pool (public approved drugs
outside the held-out components). The objective is the mean count-vector Tanimoto (rlexp.chem.tanimoto's fingerprint) to the Train references,
leaving out the candidate's own Train entry. This is what an RL policy that ignores the profile can at best learn, so RL has to beat it before
its delivered similarity counts as more than a better constant. Writes controls in the standard sample format for `rlexp.metrics score`.

  .venv/bin/python scripts/best_constant.py --splits val --out runs/colab-s1/val-control_constant_pool [--pool data/raw/chembl_pool/pool.jsonl]

The pool is verified before it is used: its sha256 must equal derived.files["pool.jsonl"].sha256 in the manifest.json beside it (recorded by scripts/build_pool.py);
a mismatch, or a pool with no recorded hash, is an error. meta.json records that sha256 and, when the winner is a pool molecule, its chembl_id and connectivity
key (`pool_winner`); the molecule itself is never printed.
"""
import argparse
import hashlib
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


def row_smiles(row):
    return row.get("smiles") or row.get("parent_smiles") or row.get("canonical_smiles")


def read_pool(path=POOL):
    """(rows, sha256) of the pool file, ([], None) when there is none. ValueError unless the sha256 of the bytes read equals the one build_pool.py recorded in the
    manifest.json beside the pool (derived.files["pool.jsonl"].sha256)."""
    path = Path(path)
    if not path.exists():
        return [], None
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = path.with_name("manifest.json")
    try:
        recorded = json.loads(manifest.read_text(encoding="utf-8"))["derived"]["files"]["pool.jsonl"]["sha256"]
    except (OSError, KeyError, TypeError, ValueError):
        raise ValueError(f"{path} has no recorded sha256 ({manifest}: derived.files['pool.jsonl']): run scripts/build_pool.py") from None
    if digest != recorded:
        raise ValueError(f"{path} differs from the sha256 recorded by build_pool.py: run scripts/build_pool.py again")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()], digest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--splits", default="val", help="comma-separated; Test splits only after the lock")
    ap.add_argument("--pool", default=str(POOL), help="the leakage-filtered ChEMBL pool of scripts/build_pool.py (used only if the file exists)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    train_refs = references(read_rows(args.data, "train"))
    pool_rows, pool_sha256 = read_pool(args.pool)
    pool = [(row_smiles(row), row) for row in pool_rows if row_smiles(row)]
    by_smiles = {}
    for smiles, row in pool:
        by_smiles.setdefault(smiles, row)
    smiles, score, source = choose(train_refs, list(by_smiles))
    rows = [r for s in args.splits.split(",") for r in read_rows(args.data, s)]
    samples = metrics.answer_samples(rows, [smiles] * len(rows), "control_constant_pool")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics.write_jsonl(out / "samples.jsonl", samples)
    meta = {"control": "constant_pool", "objective": "mean count-vector Tanimoto to Train references", "train_score": round(score, 4),
            "source": source, "candidates": len(train_refs) + len(pool), "splits": args.splits.split(","), "answer_chars": len(smiles),
            "pool_sha256": pool_sha256}
    if source == "pool":
        meta["pool_winner"] = {key: by_smiles[smiles].get(key) for key in ("chembl_id", "conn")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"error: {type(e).__name__}: {e}")
