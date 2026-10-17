#!/usr/bin/env python3
"""Build the merged models of a frozen spec (docs/prereg-v1.md section 5): for every arm that names an adapter, check the adapter's files against
the hashes recorded at the freeze, then merge it into <models>/<arm id> with gpu/merge_lora.py at the pinned base revision.

  python gpu/prepare_final.py --spec results/frozen_arms.json --spec-sha256 <hex> --models final_models [--only a,b] [--dry-run]

An arm with an "adapter" block {"path", "base", "sha256": {file: hex}} is merged; an arm without one (untrained models, CPU controls) is left alone.
Re-running skips an arm whose PREPARED.json already records the same adapter hashes. Needs the CPU only (merge runs in bf16 on the CPU), a few
minutes and 9 GB of disk per 4B arm: delete a merged model after its Test samples exist.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_adapter(block):
    """The adapter directory's files match the frozen hashes, else ValueError (nothing is merged from an unregistered adapter)."""
    directory = Path(block["path"])
    missing = [f for f in ADAPTER_FILES if not (directory / f).is_file()]
    if missing:
        raise ValueError(f"adapter {block['path']} lacks {missing}")
    for name, expected in block["sha256"].items():
        if file_sha256(directory / name) != expected:
            raise ValueError(f"adapter file {name} differs from the frozen hash")
    if set(block["sha256"]) < set(ADAPTER_FILES):
        raise ValueError("the spec must freeze the hash of every adapter file")


def merge_command(arm_id, block, models):
    return [sys.executable, str(HERE / "merge_lora.py"), "--base", block.get("base", "Qwen/Qwen3.5-4B"), "--adapter", block["path"], "--out", str(Path(models) / arm_id)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--spec-sha256", required=True)
    ap.add_argument("--models", type=Path, default=Path("final_models"))
    ap.add_argument("--only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if file_sha256(args.spec) != args.spec_sha256:
        sys.exit("error: the spec's sha256 differs from the frozen one")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("frozen") is not True:
        sys.exit("error: the spec is not marked frozen")
    wanted = set(args.only.split(",")) if args.only else None
    done = []
    for arm in spec["arms"]:
        block = arm.get("adapter")
        if block is None or (wanted is not None and arm["id"] not in wanted):
            continue
        try:
            check_adapter(block)
        except ValueError as e:
            sys.exit(f"error: arm {arm['id']}: {e}")
        out = args.models / arm["id"]
        marker = out / "PREPARED.json"
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("adapter_sha256") == block["sha256"]:
            print(f"{arm['id']}: already prepared")
            continue
        cmd = merge_command(arm["id"], block, args.models)
        print(json.dumps(cmd) if args.dry_run else f"merging {arm['id']}", flush=True)
        if args.dry_run:
            continue
        if subprocess.run(cmd).returncode:
            sys.exit(f"error: merge of {arm['id']} failed")
        out.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"adapter_sha256": block["sha256"], "spec_sha256": args.spec_sha256}, indent=1) + "\n", encoding="utf-8")
        done.append(arm["id"])
    if wanted and not (wanted <= {a["id"] for a in spec["arms"]}):
        sys.exit("error: --only names an arm that is not in the spec")
    print(json.dumps({"prepared": done}))


if __name__ == "__main__":
    main()
