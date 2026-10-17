#!/usr/bin/env python3
"""Locked evaluation driver (docs/prereg-v1.md): sample every frozen arm on the Test splits, nothing else.

  python gpu/final_eval.py --spec results/frozen_arms.json --spec-sha256 <hex> --data data --out runs/final [--only a,b] [--dry-run]

The spec lists, per arm, the exact sampler settings frozen at the tag. The driver refuses a spec whose sha256 differs from the one given
(record the hash in the tag message), an unfrozen spec, a setting outside the frozen grid, and any split other than test_t and test_r.
It calls gpu/sample_vllm.py once per arm over both splits; that script is resumable and prompt-atomic, so a killed run can be started again with
the same command. Scoring happens later, on the CPU, with rlexp.metrics. Needs the GPU environment of gpu/sample_vllm.py (Colab A100).
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = ("test_t", "test_r")
TEMPERATURES = (0.3, 0.6, 1.0)          # the selectable grid, chosen on Val by fv_tanimoto at k=8
NEAR_GREEDY = 0.01                       # reported at k=1 only, never selected
MODES = ("think", "nothink")
DATA_VARIANTS = {"direct": "processed", "icl": "processed/arms/a4", "name": "processed/arms/name",
                 "ident": "processed/arms/ident", "shuffle": "processed/arms/shuffle"}   # directories under --data holding <split>.jsonl
SYSTEM_TEXTS = {"direct": "Reply with only the SMILES string of the drug described. No explanation.", "dataset": None}
REQUIRED = ("id", "model", "mode", "data", "system", "temperature", "k", "max_tokens")


class SpecError(ValueError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_spec(path, expected_sha256):
    """The validated spec. Raises SpecError for a hash mismatch, an unfrozen spec, or a setting outside the frozen grid."""
    if file_sha256(path) != expected_sha256:
        raise SpecError("The spec's sha256 differs from the frozen one: this is not the registered protocol")
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("frozen") is not True:
        raise SpecError("The spec is not marked frozen")
    if tuple(spec.get("splits", ())) != SPLITS:
        raise SpecError(f"splits must be exactly {list(SPLITS)}")
    seen = set()
    for arm in spec.get("arms", []):
        missing = [key for key in REQUIRED if key not in arm]
        if missing:
            raise SpecError(f"arm {arm.get('id')!r} lacks {missing}")
        if arm["id"] in seen or not str(arm["id"]).replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise SpecError(f"arm id {arm['id']!r} is repeated or not a plain name")
        seen.add(arm["id"])
        if arm["mode"] not in MODES or arm["data"] not in DATA_VARIANTS or arm["system"] not in SYSTEM_TEXTS:
            raise SpecError(f"arm {arm['id']}: mode, data or system outside the frozen vocabulary")
        if arm["temperature"] == NEAR_GREEDY:
            if arm.get("k") != 1:
                raise SpecError(f"arm {arm['id']}: near-greedy decoding is a k=1 report, not a selectable temperature")
        elif arm["temperature"] not in TEMPERATURES:
            raise SpecError(f"arm {arm['id']}: temperature {arm['temperature']} is outside the frozen grid {TEMPERATURES}")
        if not (isinstance(arm["k"], int) and 1 <= arm["k"] <= 8) or not (isinstance(arm["max_tokens"], int) and arm["max_tokens"] >= 1):
            raise SpecError(f"arm {arm['id']}: k must be an integer in 1..8 and max_tokens a positive integer")
    if not seen:
        raise SpecError("the spec has no arms")
    return spec


def command(arm, data_root, out_root):
    """The sample_vllm.py command line of one arm over both Test splits."""
    out = [sys.executable, str(HERE / "sample_vllm.py"), "--model", arm["model"], "--data", str(Path(data_root) / DATA_VARIANTS[arm["data"]]),
           "--splits", ",".join(SPLITS), "--mode", arm["mode"], "--k", str(arm["k"]), "--max-tokens", str(arm["max_tokens"]),
           "--temperature", str(arm["temperature"]), "--prompts-per-batch", "auto", "--out", str(Path(out_root) / arm["id"])]
    if SYSTEM_TEXTS[arm["system"]] is not None:
        out += ["--system-text", SYSTEM_TEXTS[arm["system"]]]
    if arm.get("quantization"):
        out += ["--quantization", arm["quantization"]]
    if arm.get("gpu_memory_utilization"):
        out += ["--gpu-memory-utilization", str(arm["gpu_memory_utilization"])]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--spec-sha256", required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("runs/final"))
    ap.add_argument("--only", help="comma-separated arm ids")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args(argv)
    try:
        spec = load_spec(args.spec, args.spec_sha256)
    except SpecError as e:
        sys.exit(f"error: {e}")
    wanted = set(args.only.split(",")) if args.only else None
    arms = [a for a in spec["arms"] if wanted is None or a["id"] in wanted]
    if wanted and {a["id"] for a in arms} != wanted:
        sys.exit("error: --only names an arm that is not in the spec")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps({"spec_sha256": args.spec_sha256, "arms": [a["id"] for a in arms]}, indent=1) + "\n")
    for arm in arms:
        cmd = command(arm, args.data, args.out)
        print(" ".join(cmd[:2] + ["..."] + cmd[-4:]) if not args.dry_run else json.dumps(cmd), flush=True)
        if not args.dry_run and subprocess.run(cmd).returncode:
            sys.exit(f"error: arm {arm['id']} failed; rerun the same command to resume")


if __name__ == "__main__":
    main()
