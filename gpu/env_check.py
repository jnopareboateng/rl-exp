#!/usr/bin/env python3
"""Print local runtime facts as JSON. GPU detection on Colab/Kaggle is UNTESTED.

Use only the standard library. Do not import CUDA libraries or make network calls.
GPU entries come from nvidia-smi; CUDA_VISIBLE_DEVICES is reported separately.
"""

import ast
import csv
import importlib.metadata as metadata
import json
import os
import platform
import re
import subprocess


def environment():
    versions = {}
    for name in ("torch", "transformers", "vllm", "peft", "trl", "numpy",
                 "flash-linear-attention", "fla-core", "causal-conv1d"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": platform.python_version(), "library_versions": versions,
              "cuda": {"torch_build": None, "driver_supported": None}, "gpus": [],
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        source = metadata.distribution("torch").locate_file("torch/version.py").read_text()
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "cuda" for t in node.targets
            ):
                result["cuda"]["torch_build"] = ast.literal_eval(node.value)
            elif isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "cuda":
                result["cuda"]["torch_build"] = ast.literal_eval(node.value)
    except (metadata.PackageNotFoundError, OSError, ValueError, SyntaxError):
        pass
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
        )
        if proc.returncode:
            raise OSError("GPU inventory unavailable")
        for index, uuid, name, memory, driver in csv.reader(proc.stdout.splitlines()):
            result["gpus"].append({"index": int(index), "uuid": uuid.strip(),
                                   "name": name.strip(), "memory_mib": float(memory),
                                   "driver_version": driver.strip()})
        proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        match = re.search(r"CUDA Version:\s*([\d.]+)", proc.stdout)
        result["cuda"]["driver_supported"] = match.group(1) if match else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        result["gpu_inventory_status"] = "unavailable"
    return result


if __name__ == "__main__":
    print(json.dumps(environment(), indent=2))
