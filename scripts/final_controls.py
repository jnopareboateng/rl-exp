#!/usr/bin/env python3
"""The CPU controls of the locked evaluation (docs/prereg-v1.md section 3): C0 to C3 on the two Test splits, eight identical samples per row, scored with metrics v3.

  .venv/bin/python scripts/final_controls.py --final runs/final [--data data/processed] [--k 8]

C0, C1 and C2 are the Test rows of the controls CLI's files (`runs/control_*/samples.jsonl`, written before the lock from Train alone); C3 is chosen from Train and the
pool by scripts/best_constant.py and answered for the Test rows. Each control lands in `<final>/<id>/` with samples.jsonl, scored.v3.jsonl and a meta.json that records the
sha256 of its source. Run it after the tag: the Test rows are read here for the first time by anything that scores them. No answer text is printed.
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from rlexp import metrics

SPLITS = ("test_t", "test_r")
SOURCES = {"C0": "runs/control_constant", "C1": "runs/control_copy_nearest", "C2": "runs/control_constant_scaffold"}
HERE = Path(__file__).resolve().parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_rows(samples_path):
    """The rows of a controls file that belong to the Test splits (one answer per row); ValueError if either split is missing."""
    rows = [r for r in (json.loads(line) for line in Path(samples_path).read_text(encoding="utf-8").splitlines() if line.strip()) if r.get("split") in SPLITS]
    missing = [s for s in SPLITS if not any(r["split"] == s for r in rows)]
    if missing:
        raise ValueError(f"{samples_path} has no rows of {missing}")
    return rows


def write_control(final, control, rows, k, source, extra=None):
    out = Path(final) / control
    out.mkdir(parents=True, exist_ok=True)
    metrics.write_jsonl(out / "samples.jsonl", load_script("replicate_samples").replicate(rows, k))
    meta = {"control": control, "k": k, "splits": list(SPLITS), "rows": len(rows), "source": str(source), "source_sha256": sha256(source), **(extra or {})}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return out


def score(out, data):
    done = subprocess.run([sys.executable, "-m", "rlexp.metrics", "score", "--samples", str(out / "samples.jsonl"), "--data", str(data), "--out", str(out / "scored.v3.jsonl")],
                          capture_output=True, text=True)
    if done.returncode:
        sys.exit(f"error: scoring {out.name} failed: {done.stderr.strip()[-300:]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", type=Path, default=Path("runs/final"))
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args(argv)
    for control, source in SOURCES.items():
        path = Path(source) / "samples.jsonl"
        score(write_control(args.final, control, test_rows(path), args.k, path), args.data)
    with tempfile.TemporaryDirectory() as tmp:
        load_script("best_constant").main(["--data", str(args.data), "--splits", ",".join(SPLITS), "--out", tmp])
        raw = Path(tmp) / "samples.jsonl"
        pool = json.loads((Path(tmp) / "meta.json").read_text(encoding="utf-8"))
        score(write_control(args.final, "C3", test_rows(raw), args.k, raw, {"pool_sha256": pool.get("pool_sha256"), "train_score": pool.get("train_score"), "source_kind": pool.get("source")}), args.data)
    print(f"wrote C0 to C3 under {args.final} (k={args.k}, splits {', '.join(SPLITS)})")


if __name__ == "__main__":
    main()
