#!/usr/bin/env python3
"""Pack everything a fresh Colab GPU runtime needs into one tarball (no Test files, no raw data, no runs).

  .venv/bin/python scripts/make_bootstrap_bundle.py --out runs/bootstrap.tgz
  # on Colab: upload it, `mkdir -p /content/rl-exp && tar xzf bootstrap.tgz -C /content/rl-exp && cd /content/rl-exp && sh gpu/bootstrap_colab.sh`

Contents: gpu/ (only .py .sh .md .txt .json files; the others are listed as skipped), src/rlexp/, pyproject.toml, data/processed/{train,val}.jsonl and,
per arm directory, only val.jsonl and its meta file. The Test splits are never packed: they reach the GPU runtime only through the locked-evaluation
step after the tag. Three guards, each an error: a file whose name contains `test_`; a file whose sha256 equals that of any `test_*` entry of
data/MANIFEST.sha256 (a Test split under a neutral name, in gpu/ or an arm directory); a symlink (the file or a directory above it).

The code on the GPU runtime must be the code of the tag. The bundle is refused unless HEAD is the commit of the annotated tag `prereg-v1`, no tracked file
differs from HEAD, and every packed code file (all but data/) is tracked; it then holds the member BUNDLE_COMMIT with that commit id. `--allow-untagged` skips
those checks for the Val runs before the tag exists: its BUNDLE_COMMIT reads `untagged`, and the locked evaluation's code hashes still decide at confirmation.
"""
import argparse
import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "prereg-v1"
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


def git(root, *args):
    """stdout of `git -C <root> <args>`; ValueError when git is missing or exits non-zero."""
    try:
        done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        raise ValueError(f"git is not usable ({e})") from e
    if done.returncode != 0:
        raise ValueError(f"git {args[0]} failed: {done.stderr.strip()}")
    return done.stdout


def bundle_commit(root, files, tag=TAG):
    """The commit the bundle is made from. ValueError unless HEAD is the commit of the annotated `tag`, no tracked file differs from HEAD and every packed
    code file (all but data/, whose files are ignored by git and pinned by data/MANIFEST.sha256) is tracked, so the code is exactly the tag's."""
    try:
        kind = git(root, "cat-file", "-t", f"refs/tags/{tag}").strip()
    except ValueError as e:
        raise ValueError(f"cannot read the tag {tag} ({e}): tag the frozen commit first, or pass --allow-untagged for a bundle that only samples Val") from e
    if kind != "tag":
        raise ValueError(f"{tag} is not an annotated tag (git cat-file -t says {kind!r})")
    head, tagged = git(root, "rev-parse", "HEAD").strip(), git(root, "rev-parse", f"refs/tags/{tag}^{{commit}}").strip()
    if head != tagged:
        raise ValueError(f"HEAD ({head[:12]}) is not the commit of {tag} ({tagged[:12]}): check out the tag")
    changed = [line[3:] for line in git(root, "status", "--porcelain", "--untracked-files=no").splitlines()]
    if changed:
        raise ValueError(f"{len(changed)} tracked file(s) differ from {tag}: {changed}: commit or restore them")
    tracked = set(git(root, "ls-files", "-z").split("\0"))
    loose = [r for r in files if not r.startswith("data/") and r not in tracked]
    if loose:
        raise ValueError(f"code that git does not track would be bundled, so it is not in {tag}: {loose}")
    return head


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-untagged", action="store_true", help="skip the tag and clean-tree checks (Val runs before the tag exists); BUNDLE_COMMIT reads 'untagged'")
    args = ap.parse_args(argv)
    skipped = []
    files = members(ROOT, skipped=skipped)
    commit = "untagged" if args.allow_untagged else bundle_commit(ROOT, files)
    if skipped:
        print(f"skipped {len(skipped)} gpu/ file(s) with another suffix: {skipped}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out, "w:gz") as tar:
        for rel in files:
            tar.add(ROOT / rel, arcname=rel)
        text = f"{commit}\n".encode()
        info = tarfile.TarInfo("BUNDLE_COMMIT")
        info.size = len(text)
        tar.addfile(info, io.BytesIO(text))
    print(f"{len(files)} files at {commit[:12]}, {args.out.stat().st_size / 2**20:.1f} MiB -> {args.out}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
