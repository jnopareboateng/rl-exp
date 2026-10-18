#!/usr/bin/env python3
"""Build the merged models of a frozen spec (docs/prereg-v1.md section 5): for every arm that names an adapter, check the adapter's files against
the hashes recorded at the freeze, then merge it into <models>/<arm id> with gpu/merge_lora.py at the pinned base revision.

  python gpu/prepare_final.py --spec results/frozen_arms.json --spec-sha256 <hex> --selection results/selection.json --models final_models [--only a,b] [--dry-run]

The spec is validated by gpu/final_eval.py's load_spec (the same frozen grid, and it must be what --selection chose), so a spec that final_eval would
refuse is refused here before anything is merged.
An arm with an "adapter" block {"path", "base", "sha256": {file: hex}} is merged into <models>/<arm id>, which must be the arm's "model" (what final_eval
samples); an arm without one (untrained models) is left alone, and so is a "diagnostic" arm, which samples the merged model of the selected arm it belongs to.
PREPARED.json records the adapter hashes and the sha256 of the merged export's *.safetensors and config.json. Re-running skips an arm only when the marker
records the same adapter hashes AND the merged files on disk still hash as recorded; otherwise merge_lora.py runs (it checks an existing export itself).
Needs the CPU only (merge runs in bf16 on the CPU), a few minutes and 9 GB of disk per 4B arm: delete a merged model after its Test samples exist.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from final_eval import file_sha256, load_spec, merged_hashes    # noqa: E402  one frozen grid, one hash function, one definition of the merged files

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


def check_adapter(block):
    """The adapter directory's files match the frozen hashes, else ValueError (nothing is merged from an unregistered adapter)."""
    directory = Path(block["path"])
    missing = [f for f in (*ADAPTER_FILES, *block["sha256"]) if not (directory / f).is_file()]
    if missing:
        raise ValueError(f"adapter {block['path']} lacks {sorted(set(missing))}")
    for name, expected in block["sha256"].items():
        if file_sha256(directory / name) != expected:
            raise ValueError(f"adapter file {name} differs from the frozen hash")
    if set(block["sha256"]) < set(ADAPTER_FILES):
        raise ValueError("the spec must freeze the hash of every adapter file")


def is_prepared(marker, block, out):
    """True when the marker records this adapter and the merged files in `out` still hash as the marker says."""
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return recorded.get("adapter_sha256") == block["sha256"] and bool(recorded.get("merged_sha256")) and recorded["merged_sha256"] == merged_hashes(out)


def merge_command(arm_id, block, models):
    return [sys.executable, str(HERE / "merge_lora.py"), "--base", block.get("base", "Qwen/Qwen3.5-4B"), "--adapter", block["path"], "--out", str(Path(models) / arm_id)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--spec-sha256", required=True)
    ap.add_argument("--selection", type=Path, required=True, help="results/selection.json: the Val selection the spec must realise")
    ap.add_argument("--models", type=Path, default=Path("final_models"))
    ap.add_argument("--only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    try:
        spec = load_spec(args.spec, args.spec_sha256, args.selection)
    except (ValueError, OSError) as e:
        sys.exit(f"error: {e}")
    wanted = set(args.only.split(",")) if args.only else None
    if wanted and not wanted <= {a["id"] for a in spec["arms"]}:
        sys.exit("error: --only names an arm that is not in the spec")
    done = []
    for arm in spec["arms"]:
        block = arm.get("adapter")
        if block is None or arm.get("role") == "diagnostic" or (wanted is not None and arm["id"] not in wanted):
            continue
        out = args.models / arm["id"]
        if Path(arm["model"]).resolve() != out.resolve():
            sys.exit(f"error: arm {arm['id']}: the spec samples from {arm['model']} but the merge is written to {out}")
        try:
            check_adapter(block)
        except ValueError as e:
            sys.exit(f"error: arm {arm['id']}: {e}")
        marker = out / "PREPARED.json"
        if is_prepared(marker, block, out):
            print(f"{arm['id']}: already prepared")
            continue
        cmd = merge_command(arm["id"], block, args.models)
        print(json.dumps(cmd) if args.dry_run else f"merging {arm['id']}", flush=True)
        if args.dry_run:
            continue
        if subprocess.run(cmd).returncode:
            sys.exit(f"error: merge of {arm['id']} failed (an existing {out} that no longer matches its merge report must be deleted first)")
        merged = merged_hashes(out)
        if not any(name.endswith(".safetensors") for name in merged):
            sys.exit(f"error: the merge of {arm['id']} wrote no *.safetensors weights to {out}")
        marker.write_text(json.dumps({"adapter_sha256": block["sha256"], "merged_sha256": merged, "spec_sha256": args.spec_sha256}, indent=1) + "\n", encoding="utf-8")
        done.append(arm["id"])
    print(json.dumps({"prepared": done}))


if __name__ == "__main__":
    main()
