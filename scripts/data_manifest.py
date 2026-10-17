#!/usr/bin/env python3
"""Pin the data behind every result: sha256 of each file under data/ (raw, curated, processed).

    python scripts/data_manifest.py            write data/MANIFEST.sha256 and print its own sha256
    python scripts/data_manifest.py --check    exit 1 if any file differs from the committed manifest

The manifest is tracked in git (hashes of private files reveal nothing usable); the files themselves are not.
Runs record the printed manifest hash as `data_manifest_sha256` in meta.json.
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = DATA / "MANIFEST.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build() -> str:
    files = sorted(p for p in DATA.rglob("*") if p.is_file() and p != MANIFEST and not p.name.startswith("."))
    return "".join(f"{sha256(p)}  {p.relative_to(DATA).as_posix()}\n" for p in files)


if __name__ == "__main__":
    text = build()
    if "--check" in sys.argv:
        same = MANIFEST.exists() and MANIFEST.read_text() == text
        print("manifest matches" if same else "manifest differs from the files on disk")
        sys.exit(0 if same else 1)
    MANIFEST.write_text(text)
    print(f"{len(text.splitlines())} files; data_manifest_sha256={hashlib.sha256(text.encode()).hexdigest()}")
