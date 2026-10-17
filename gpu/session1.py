#!/usr/bin/env python3
"""CPU helpers for run_session1.sh. Real notebook/GPU sessions are UNTESTED.

Read and copy data without printing text. The report contains aggregate facts
only. Chemistry scoring and the k=8 study gates remain separate checks.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

from sample_vllm import load_rows, write_meta


def preflight(scratch):
    env = json.loads((scratch / "before.json").read_text(encoding="utf-8"))
    gpus = env.get("gpus", [])
    if not gpus:
        raise SystemExit("No GPU detected. Select a GPU runtime in Kaggle or Colab, then restart this command.")
    for gpu in gpus:
        print(f"GPU: {gpu['name']}; driver: {gpu['driver_version']}; capability: {gpu['compute_capability']}")
        if gpu["compute_capability"] < [7, 5]:
            raise SystemExit("Compute capability < 7.5. Select T4, L4 or A100; P100 cannot run this pack.")
        if int(gpu["driver_version"].split(".")[0]) < 575:
            raise SystemExit("Driver older than R575. Select a newer runtime image or ask the host to update its driver. A notebook cannot update the host driver.")
    visible = env.get("cuda_visible_devices")
    if visible is not None and (not visible.strip() or visible.strip() == "-1"):
        raise SystemExit("CUDA_VISIBLE_DEVICES hides the GPUs. Enable the runtime GPU, then restart this command.")


def prepare(scratch):
    source = Path("data/processed")
    # Keep all Val rows and every view of 40 randomly selected Train drugs.
    train = load_rows(SimpleNamespace(data=source, splits="train", limit_drugs=40, prompt_subset_seed=0))
    val = load_rows(SimpleNamespace(data=source, splits="val", limit_drugs=None, prompt_subset_seed=0))
    smoke, probe = scratch / "smoke-data", scratch / "probe-data"
    archive = scratch / "archive" / str(time.time_ns())
    for name in ("smoke-short", "smoke-long"):
        old = scratch / name
        if old.exists():
            archive.mkdir(parents=True, exist_ok=True)
            old.rename(archive / name)
    smoke.mkdir(exist_ok=True)
    probe.mkdir(exist_ok=True)
    (smoke / "val.jsonl").write_text(json.dumps(val[0], ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copyfile(source / "val.jsonl", probe / "val.jsonl")
    with (probe / "train.jsonl").open("w", encoding="utf-8") as stream:
        for row in train:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Prepared Val rows={len(val)}; Train rows={len(train)}; Train drugs={len({r['group'] for r in train})}; subset seed=0")


def summarize_run(directory):
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        return {"status": "not_started", "completed_prompts": 0}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    splits = {}
    path = directory / "samples.jsonl"
    k = meta["configuration"]["k"]
    batch = []
    with path.open("rb") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError):
                break
            if (not line.endswith(b"\n") or row["sample_idx"] != len(batch)
                    or (batch and row["id"] != batch[0]["id"])):
                break
            batch.append(row)
            if len(batch) != k:
                continue
            split = splits.setdefault(row["split"], dict(lengths=[], closed=0, truncated=0, completed_prompts=0))
            split["completed_prompts"] += 1
            split["lengths"].extend(s["n_tokens"] for s in batch)
            split["closed"] += sum("</think>" in s["text"] for s in batch)
            split["truncated"] += sum(s["finish_reason"] == "length" for s in batch)
            batch = []
    all_lengths = []
    for entry in splits.values():
        lengths = sorted(entry.pop("lengths"))
        all_lengths.extend(lengths)
        entry.update(samples=len(lengths), closed_think_rate=(entry["closed"] / len(lengths)
                     if meta["configuration"]["mode"] == "think" else None),
                     saved_text_closed_rate=entry["closed"] / len(lengths),
                     truncation_rate=entry["truncated"] / len(lengths),
                     mean_generated_tokens=sum(lengths) / len(lengths),
                     p90_generated_tokens=lengths[math.ceil(0.9 * len(lengths)) - 1])
    all_lengths.sort()
    samples = len(all_lengths)
    return dict(status=meta["status"], completed_prompts=sum(s["completed_prompts"] for s in splits.values()),
                samples=samples, tokens_per_second=meta.get("tokens_per_second"),
                timing_scope=meta.get("timing_scope"), calls=meta.get("calls", []),
                calls_scope=meta.get("calls_scope"), prompts_per_batch=meta.get("prompts_per_batch"),
                auto_batch_estimate=meta.get("auto_batch_estimate"), engine=meta.get("engine"),
                closed_think_rate=(sum(s["closed"] for s in splits.values()) / samples
                                   if samples and meta["configuration"]["mode"] == "think" else None),
                truncation_rate=sum(s["truncated"] for s in splits.values()) / samples if samples else None,
                mean_generated_tokens=sum(all_lengths) / samples if samples else None,
                p90_generated_tokens=all_lengths[math.ceil(0.9 * samples) - 1] if samples else None,
                per_split=splits)


def report(scratch, exit_code):
    env_path = scratch / "after.json"
    if not env_path.exists():
        env_path = scratch / "before.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
    steps = []
    timing = scratch / "steps.tsv"
    if timing.exists():
        for line in timing.read_text(encoding="utf-8").splitlines():
            name, code, seconds = line.split("\t")
            steps.append(dict(step=name, exit_code=int(code), seconds=int(seconds)))
    runs = {}
    for name in ("smoke-short", "smoke-long", "nothink-val", "nothink-train", "think-val", "think-train"):
        try:
            runs[name] = summarize_run(scratch / name)
        except (OSError, ValueError, KeyError, TypeError) as error:
            runs[name] = dict(status="unreadable", error_type=type(error).__name__)
    failures = {s["step"]: s["exit_code"] for s in steps if s["exit_code"]}
    for name, run in runs.items():
        run["executed_this_session"] = any(s["step"] == name for s in steps)
        if name in failures:
            run.update(sampler_status=run["status"], status="failed", exit_code=failures[name])
    if "smoke-resume" in failures:
        runs["smoke-short"].update(status="failed_resume", resume_exit_code=failures["smoke-resume"])
    result = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  status="failed" if exit_code else "partial" if failures or any(
                      r["status"] == "unreadable" for r in runs.values()) else "complete",
                  gpu_execution="UNTESTED" if not env.get("gpus") else "see per-step exit codes and run statuses",
                  versions={"python": env.get("python"), **env.get("library_versions", {})},
                  gpus=env.get("gpus", []), cuda=env.get("cuda"), steps=steps,
                  install_timings_s={s["step"]: s["seconds"] for s in steps if s["step"] in
                                     {"uv-download", "uv-install", "venv", "stack-install", "pip-check"}},
                  runs=runs, metric_scope="sample-weighted; counts include only complete durable prompts",
                  gates={"G0": "UNTESTED: probe k=4 cannot measure pass@8; score a later k=8 run",
                         "G1": "UNTESTED: closure alone cannot establish chemical validity or T5"})
    target = Path("results/session1_report.json")
    target.parent.mkdir(exist_ok=True)
    write_meta(target, result)
    print(f"Report: {target}; status={result['status']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "prepare", "report"))
    parser.add_argument("scratch", type=Path)
    parser.add_argument("exit_code", type=int, nargs="?", default=0)
    args = parser.parse_args()
    if args.action == "report":
        report(args.scratch, args.exit_code)
    else:
        globals()[args.action](args.scratch)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Session helper failed ({type(error).__name__}). Data values are omitted.", file=sys.stderr)
        sys.exit(1)
