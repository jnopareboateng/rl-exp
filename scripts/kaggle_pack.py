#!/usr/bin/env python3
"""Build the folders for the Kaggle batch route under .agent-runs/kaggle/ (ignored by git):
  pack/    the private dataset jnopareboateng/rl-exp-pack: GPU scripts plus the Train and Val prompt files, flat
  kernel/  the script kernel jnopareboateng/rl-exp-s1: gpu/kaggle_batch.py plus its metadata

    python scripts/kaggle_pack.py [--shape NvidiaTeslaA100] [--internet]
Then: uvx kaggle datasets create -p .agent-runs/kaggle/pack   (or `datasets version -m MESSAGE` afterwards)
      uvx kaggle kernels push -p .agent-runs/kaggle/kernel -t SECONDS
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".agent-runs/kaggle"
OWNER = "jnopareboateng"

ap = argparse.ArgumentParser()
ap.add_argument("--shape", default="NvidiaTeslaA100")
ap.add_argument("--slug", default="rl-exp-s1")
args = ap.parse_args()
pack, kernel = BASE / "pack", BASE / "kernel"
for d in (pack, kernel):
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
for name in ("env_check.py", "sample_vllm.py", "session1.py", "run_session1.sh"):
    shutil.copy(ROOT / "gpu" / name, pack / name)
for split in ("val", "train"):
    shutil.copy(ROOT / "data/processed" / f"{split}.jsonl", pack / f"{split}.jsonl")
(pack / "dataset-metadata.json").write_text(json.dumps(
    {"title": "rl-exp pack", "id": f"{OWNER}/rl-exp-pack", "licenses": [{"name": "other"}]}, indent=1))
shutil.copy(ROOT / "gpu/kaggle_batch.py", kernel / "run.py")
(kernel / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{OWNER}/{args.slug}", "title": args.slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
    "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true", "machine_shape": args.shape,
    "dataset_sources": [f"{OWNER}/rl-exp-pack"], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=1))
print(f"built {pack} ({sum(1 for _ in pack.iterdir())} files) and {kernel}")
