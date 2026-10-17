#!/usr/bin/env python3
"""Pack everything a fresh Colab GPU runtime needs into one tarball (no Test files, no raw data, no runs).

  .venv/bin/python scripts/make_bootstrap_bundle.py --out runs/bootstrap.tgz
  # on Colab: upload it, `mkdir -p /content/rl-exp && tar xzf bootstrap.tgz -C /content/rl-exp && cd /content/rl-exp && sh gpu/bootstrap_colab.sh`

Contents: gpu/ (only .py .sh .md .txt .json files; the others are listed as skipped), src/rlexp/, pyproject.toml, data/processed/{train,val}.jsonl and,
per arm directory, only val.jsonl and its meta file. The Test splits are never packed: they reach the GPU runtime only through the locked-evaluation
step after the tag. Three guards, each an error: a file whose name contains `test_`; a file whose sha256 equals that of any `test_*` entry of
data/MANIFEST.sha256 (a Test split under a neutral name, in gpu/ or an arm directory); a symlink (the file or a directory above it).
"""
import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM_DIRS = ("name", "ident", "shuffle", "blank", "a4")
ARM_FILES = ("a2_answer_train.jsonl", "a2_answer_train.meta.json", "a2_answer_val.jsonl", "a2_answer_val.meta.json")
GPU_SUFFIXES = (".py", ".sh", ".md", ".txt", ".json")


def leak_hashes(root):
    """The sha256 of every `test_*` entry of data/MANIFEST.sha256 (the manifest lists 'sha256  path' lines); empty when there is no manifest."""
    manifest = root / "data" / "MANIFEST.sha256"
    if not manifest.is_file():
        return set()
    lines = [line.strip().partition("  ") for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {digest for digest, _, name in lines if "test_" in Path(name.strip()).name}


def has_symlink(root, rel):
    """True when `rel` or any directory above it (below the root) is a symlink."""
    path = root
    for part in Path(rel).parts:
        path = path / part
        if path.is_symlink():
            return True
    return False


def members(root=ROOT, skipped=None):
    """(relative path) of every file in the bundle, sorted; raises if a Test file, a Test file's content or a symlink would be included.
    `skipped`, if a list, receives the gpu/ files left out for their suffix (anything but .py .sh .md .txt .json; caches are not reported)."""
    gpu = [p for p in sorted((root / "gpu").rglob("*")) if (p.is_file() or p.is_symlink()) and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    if skipped is not None:
        skipped += [p.relative_to(root).as_posix() for p in gpu if p.suffix not in GPU_SUFFIXES]
    out = [p for p in gpu if p.suffix in GPU_SUFFIXES]
    out += [p for p in sorted((root / "src" / "rlexp").glob("*.py"))]
    out += [root / "pyproject.toml"]
    out += [root / "data" / "processed" / f"{s}.jsonl" for s in ("train", "val")]
    arms = root / "data" / "processed" / "arms"
    out += [arms / f for f in ARM_FILES]
    for d in ARM_DIRS:
        out += [arms / d / f"val{suffix}" for suffix in (".jsonl", ".meta.json") if (arms / d / f"val{suffix}").is_file()]
    rel = [p.relative_to(root).as_posix() for p in out if p.is_file() or p.is_symlink()]
    bad = [r for r in rel if "test_" in Path(r).name]
    if bad:
        raise ValueError(f"a Test file would be bundled: {bad}")
    links = [r for r in rel if has_symlink(root, r)]
    if links:
        raise ValueError(f"a symlink would be bundled: {links}")
    leaks = leak_hashes(root)
    same = [r for r in rel if leaks and hashlib.sha256((root / r).read_bytes()).hexdigest() in leaks]
    if same:
        raise ValueError(f"the content of a Test file would be bundled under another name: {same}")
    return rel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    skipped = []
    files = members(skipped=skipped)
    if skipped:
        print(f"skipped {len(skipped)} gpu/ file(s) with another suffix: {skipped}")
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
