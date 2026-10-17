#!/usr/bin/env python3
"""Regenerate the Val baseline table from samples (docs/plan.md section 5): score and report every run with the current
rlexp.metrics, then write results/val_baselines.json and print a markdown table. Aggregates only.

  .venv/bin/python scripts/val_table.py [--runs runs/colab-s1] [--out results/val_baselines.json] [--boot 2000]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

# (directory under --runs, label, decoding note). Edit here when a run is added; the order is the table order.
RUNS = [
    ("val-control_constant", "C0 constant answer (best Train reference)", "none"),
    ("val-control_copy_nearest", "C1 copy the structure of the nearest Train profile (TF-IDF)", "none"),
    ("val-control_constant_scaffold", "C2 constant scaffold answer", "none"),
    ("val-control_constant_pool", "C3 reward-optimal constant (chosen on Train from Train + ChEMBL pool)", "none"),
    ("a-direct", "4B untrained", "T=1.0, k=8"),
    ("a4-icl", "4B + 3 retrieved examples (A4)", "T=1.0, k=8"),
    ("rlfull-s20-t0.6", "4B RL-full step 20", "T=0.6, k=8, think mode"),
    ("rlfull-s20-t1.0", "4B RL-full step 20", "T=1.0, k=8, think mode"),
    ("rlfull-s30-t0.6", "4B RL-full step 30", "T=0.6, k=8, think mode"),
    ("rlfull-s30-t1.0", "4B RL-full step 30", "T=1.0, k=8, think mode"),
    ("rlfull-s40-t0.6", "4B RL-full step 40", "T=0.6, k=8, think mode"),
    ("rlfull-s40-t1.0", "4B RL-full step 40", "T=1.0, k=8, think mode"),
    ("a2-val", "4B A2 (answer-only SFT, epoch 1)", "T=1.0, k=8, think mode"),
    ("a2-t0.6", "4B A2", "T=0.6, k=8, think mode"),
    ("a2-t0.3", "4B A2", "T=0.3, k=8, think mode"),
    ("a1-val", "4B A1 (trace SFT, epoch 2)", "T=1.0, k=4, think mode, 8192 tokens"),
    ("b4-t0.6", "4B untrained", "T=0.6, k=8"),
    ("b4-t0.3", "4B untrained", "T=0.3, k=8"),
    ("b9-direct", "9B untrained", "T=1.0, k=8"),
    ("b9-t0.6", "9B untrained", "T=0.6, k=8"),
    ("b9-t0.3", "9B untrained", "T=0.3, k=8"),
    ("b27-direct", "Qwen3.5-27B-FP8 untrained", "T=1.0, k=8"),
    ("b38-direct", "Qwen3.8-27B-FP8 untrained", "T=1.0, k=4"),
    ("name-4b", "4B told the drug name", "T=1.0, k=8"),
    ("name-9b", "9B told the drug name", "T=1.0, k=8"),
    ("name-27b", "Qwen3.5-27B-FP8 told the drug name", "T=1.0, k=2"),
    ("name-38", "Qwen3.8-27B-FP8 told the drug name", "T=1.0, k=2"),
    ("g-a2", "4B A2 near-greedy", "T=0.01, k=1, think mode"),
    ("g-9b-direct", "9B untrained near-greedy", "T=0.01, k=1"),
    ("g-4b-direct", "4B untrained near-greedy", "T=0.01, k=1"),
]
COLUMNS = ("valid_closed", "exact@1", "core@1", "mean_tanimoto", "fv_tanimoto", "fv_samples", "fv_lift", "lift", "tanimoto_valid", "core@8", "fv_rate", "mean_reward")


def value(metrics, name):
    m = metrics.get(name)
    return None if m is None else (m["value"] if isinstance(m, dict) else m)


def run_cli(*args):
    done = subprocess.run([sys.executable, "-m", "rlexp.metrics", *args], capture_output=True, text=True)
    if done.returncode:
        sys.exit(f"rlexp.metrics {args[0]} failed: {done.stderr.strip()[-300:]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/colab-s1")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results/val_baselines.json")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    rows, table, report = [], [], None
    for name, label, note in RUNS:
        samples = Path(args.runs) / name / "samples.jsonl"
        if not samples.exists():
            continue
        scored, report = samples.with_name("scored.v3.jsonl"), samples.with_name("report.v3.json")
        run_cli("score", "--samples", str(samples), "--data", args.data, "--out", str(scored))
        run_cli("report", "--scored", str(scored), "--split", "val", "--data", args.data, "--boot", str(args.boot), "--json", str(report))
        split = json.loads(report.read_text())["splits"]["val"]
        metrics = split["metrics"]
        entry = {"run": name, "label": label, "decoding": note, "drugs": split["n_drugs"], "samples": split["n_samples"],
                 **{c: value(metrics, c) for c in COLUMNS}}
        lift = metrics.get("lift")
        if isinstance(lift, dict):
            entry["lift_ci"] = [lift.get("lo"), lift.get("hi")]
        rows.append(entry)
        cell = lambda v, spec="{:.3f}": "n/a" if v is None else spec.format(v)
        table.append(f"| {label} | {note} | " + " | ".join(cell(entry[c]) for c in COLUMNS[:-1]) + " |")
    bar = json.loads(report.read_text())["splits"]["val"].get("constant", {}) if report else {}
    out = {"split": "val", "columns": COLUMNS, "rows": rows,
           "constant": {k: value(bar.get("metrics", {}), k) for k in ("mean_tanimoto", "mean_reward", "lift")}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print("| arm | decoding | " + " | ".join(COLUMNS[:-1]) + " |\n|---|---|" + "---|" * (len(COLUMNS) - 1))
    print("\n".join(table))
    print(f"constant answer: {out['constant']}")


if __name__ == "__main__":
    main()
