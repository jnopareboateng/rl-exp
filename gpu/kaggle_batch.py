#!/usr/bin/env python3
"""Kaggle batch driver (kernel type script): runs gpu/run_session1.sh from the attached private dataset and saves the
small artifacts. A batch run has a hard timeout and no interactive session, so it cannot leak GPU quota.

The venv, model cache and scratch live under /tmp (outside /kaggle/working, which Kaggle saves as run output).
Environment overrides: NOTHINK_SECONDS, THINK_SECONDS, K_NOTHINK, K_THINK, DEADLINE_SECONDS (driver stops the runner
this long after start so the artifacts are copied before Kaggle's own timeout), CUDA_VISIBLE_DEVICES.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

START = time.time()
DEADLINE = float(os.environ.get("DEADLINE_SECONDS", "4800"))
INPUT, WORK, OUT = Path("/kaggle/input"), Path("/tmp/rl-exp"), Path("/kaggle/working/out")
OUT.mkdir(parents=True, exist_ok=True)


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with (OUT / "driver.log").open("a") as f:
        f.write(line + "\n")


pack = next((p.parent for p in INPUT.rglob("run_session1.sh")), None)
if pack is None:
    log("pack not found under /kaggle/input: " + ", ".join(sorted(str(p) for p in INPUT.glob("*"))))
    sys.exit(1)
shutil.rmtree(WORK, ignore_errors=True)
(WORK / "gpu").mkdir(parents=True)
(WORK / "data/processed").mkdir(parents=True)
for name in ("env_check.py", "sample_vllm.py", "session1.py", "run_session1.sh"):
    shutil.copy(pack / name, WORK / "gpu" / name)
for split in ("val", "train"):
    shutil.copy(pack / f"{split}.jsonl", WORK / "data/processed" / f"{split}.jsonl")
log(f"pack at {pack}; workdir {WORK}; deadline {DEADLINE:.0f}s")
env = dict(os.environ, NOTHINK_SECONDS=os.environ.get("NOTHINK_SECONDS", "300"), THINK_SECONDS=os.environ.get("THINK_SECONDS", "700"))
proc = subprocess.Popen(["sh", "gpu/run_session1.sh"], cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    for line in proc.stdout:  # stream the runner's log into the Kaggle log as it happens
        print(line, end="", flush=True)
        if time.time() - START > DEADLINE:
            log("driver deadline reached; terminating the runner")
            proc.terminate()
            break
    proc.wait(timeout=120)
except subprocess.TimeoutExpired:
    proc.kill()
log(f"runner exit code {proc.returncode}")
for rel in ("results/session1_report.json", "gpu/session1.log"):
    if (WORK / rel).exists():
        shutil.copy(WORK / rel, OUT / Path(rel).name)
runs = WORK / ".agent-runs/session1"
if runs.exists():
    for item in runs.iterdir():
        if item.is_dir() and item.name != "archive":
            for fname in ("samples.jsonl", "meta.json"):
                if (item / fname).exists():
                    (OUT / "runs" / item.name).mkdir(parents=True, exist_ok=True)
                    shutil.copy(item / fname, OUT / "runs" / item.name / fname)
        elif item.suffix in (".log", ".json", ".tsv"):
            shutil.copy(item, OUT / item.name)
log("artifacts copied")
