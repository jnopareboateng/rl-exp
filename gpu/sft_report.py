#!/usr/bin/env python3
"""Runner helpers. Reports contain aggregate numbers and module/version names.

CPU tests cannot prove GPU, kernel or memory behaviour. Colab installation,
training and vLLM loading are UNTESTED here. This helper uses stdlib only.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from sft_lora import PackError, atomic_json, checkpoint_path, print_failure, runtime_versions


def report(scratch, arm, run, merged, destination, exit_code):
    steps = []
    if (scratch / "steps.tsv").exists():
        for line in (scratch / "steps.tsv").read_text(encoding="utf-8").splitlines():
            name, code, seconds = line.split("\t")
            steps.append(dict(step=name, exit_code=int(code), seconds=int(seconds)))
    result = dict(arm=arm, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  exit_code=exit_code, status="failed" if exit_code else "partial", steps=steps,
                  versions={}, epochs=[], training=None, merge=None,
                  gpu_execution="UNTESTED here; inspect this run's status and step codes",
                  vllm_load="UNTESTED", metric_scope="completion tokens only; teacher-forced token accuracy")
    inventory = scratch / "versions.json"
    if inventory.exists():
        result["versions"] = json.loads(inventory.read_text(encoding="utf-8"))
    manifest = run / "checkpoint.json"
    if manifest.exists():
        try:
            pointer = json.loads(manifest.read_text(encoding="utf-8"))
            state = json.loads((checkpoint_path(run, pointer["latest"]) / "state.json").read_text(encoding="utf-8"))
            seconds = state["training_seconds"]
            result["epochs"] = state["history"]
            result["versions"] = state.get("versions", state["identity"].get("versions"))
            result["configuration"] = state["identity"]
            result["git_revision"] = state.get("git_revision")
            result["command"] = state.get("command")
            result["first_session"] = state.get("first_session")      # the code and versions that started the run
            result["sessions"] = state.get("sessions", [])            # every session, in order; git_revision above is the last one's
            result["training"] = {key: state[key] for key in (
                "status", "epoch", "global_step", "train", "val", "adapter_modules", "trainable_parameters",
                "tokens", "completion_tokens", "training_seconds", "gpu", "kernels", "peak_memory_bytes", "best_val_loss")}
            result["training"].update(tokens_per_second=state["tokens"] / seconds if seconds else None,
                                       executed_this_session=any(step["step"] == "train" for step in steps),
                                       timing_scope="cumulative successful training steps; excludes startup, validation and saving")
        except (OSError, ValueError, KeyError, TypeError):
            result["training"] = {"status": "unreadable"}
    # A prior merge must never count as an export from this invocation.
    if any(step["step"] in ("merge", "merge-reused") and step["exit_code"] == 0 for step in steps):
        try:
            result["merge"] = json.loads((merged / "merge_report.json").read_text(encoding="utf-8"))
            result["merge"]["reused"] = any(step["step"] == "merge-reused" for step in steps)
        except (OSError, ValueError):
            result["merge"] = {"status": "unreadable"}
    if exit_code == 0 and result["training"] and result["training"]["status"] == "complete" and result["merge"] and "parameters" in result["merge"]:
        result["status"] = "complete"
    atomic_json(destination, result)
    print(f"Report status={result['status']}; epochs={len(result['epochs'])}; steps={len(steps)}")


def preflight(seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise PackError("Time limit must be positive and finite")
    query = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap,driver_version,memory.total", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True)
    devices = [line.split(",") for line in query.stdout.splitlines() if line.strip()]
    if not devices or any(float(row[0]) < 7.5 or int(row[1].strip().split(".")[0]) < 570 for row in devices):
        raise PackError("Need compute capability 7.5+ and driver R570+ for the CUDA 12.8 stack")
    print(json.dumps(dict(gpu_count=len(devices), memory_mib=[int(row[2]) for row in devices],
                         hard_training_seconds=math.ceil(seconds) + 600)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("versions")
    check = commands.add_parser("preflight")
    check.add_argument("seconds", type=float)
    best = commands.add_parser("best")
    best.add_argument("run", type=Path)
    final = commands.add_parser("report")
    final.add_argument("scratch", type=Path)
    final.add_argument("arm", choices=("A1", "A2"))
    final.add_argument("run", type=Path)
    final.add_argument("merged", type=Path)
    final.add_argument("destination", type=Path)
    final.add_argument("exit_code", type=int)
    args = parser.parse_args(argv)
    if args.action == "versions":
        print(json.dumps(runtime_versions()))
    elif args.action == "preflight":
        preflight(args.seconds)
    elif args.action == "best":
        manifest = args.run / "checkpoint.json"
        best = json.loads(manifest.read_text(encoding="utf-8")).get("best") if manifest.exists() else None
        if best is None:
            sys.exit(3)
        print(checkpoint_path(args.run, best))
    else:
        report(args.scratch, args.arm, args.run, args.merged, args.destination, args.exit_code)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print_failure("SFT helper", error)
        sys.exit(1)
