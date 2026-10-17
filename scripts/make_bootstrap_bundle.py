#!/usr/bin/env python3
"""Pack everything a fresh Colab GPU runtime needs into one tarball (no Test files, no raw data, no runs).

  .venv/bin/python scripts/make_bootstrap_bundle.py --out runs/bootstrap.tgz
  # on Colab: upload it, `mkdir -p /content/rl-exp && tar xzf bootstrap.tgz -C /content/rl-exp && cd /content/rl-exp && sh gpu/bootstrap_colab.sh`

Contents: gpu/, src/rlexp/, pyproject.toml, data/processed/{train,val}.jsonl and, per arm directory, only val.jsonl and its meta file.
The Test splits are never packed: they reach the GPU runtime only through the locked-evaluation step after the tag.
"""
import argparse
import io
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM_DIRS = ("name", "ident", "shuffle", "blank", "a4")
ARM_FILES = ("a2_answer_train.jsonl", "a2_answer_train.meta.json", "a2_answer_val.jsonl", "a2_answer_val.meta.json")


def members(root=ROOT):
    """(relative path) of every file in the bundle, sorted; raises if a Test file would be included."""
    out = [p for p in sorted((root / "gpu").rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    out += [p for p in sorted((root / "src" / "rlexp").glob("*.py"))]
    out += [root / "pyproject.toml"]
    out += [root / "data" / "processed" / f"{s}.jsonl" for s in ("train", "val")]
    arms = root / "data" / "processed" / "arms"
    out += [arms / f for f in ARM_FILES]
    for d in ARM_DIRS:
        out += [arms / d / f"val{suffix}" for suffix in (".jsonl", ".meta.json") if (arms / d / f"val{suffix}").is_file()]
    rel = [p.relative_to(root).as_posix() for p in out if p.is_file()]
    bad = [r for r in rel if "test_" in Path(r).name]
    if bad:
        raise ValueError(f"a Test file would be bundled: {bad}")
    return rel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    files = members()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out, "w:gz") as tar:
        for rel in files:
            tar.add(ROOT / rel, arcname=rel)
    print(f"{len(files)} files, {args.out.stat().st_size / 2**20:.1f} MiB -> {args.out}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
