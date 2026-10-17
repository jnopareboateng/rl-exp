#!/usr/bin/env python3
"""Compute budget of the study by stage, for the paper's reproducibility statement. Aggregates only; every number comes from a run record.

  .venv/bin/python scripts/compute_budget.py [--runs runs] [--sessions results/compute_sessions.json] [--out results/compute_budget.json] [--md docs/paper/compute_budget.md]

Training: SFT arms from their summary.json (`training_seconds`), RL runs from metrics.jsonl (sum of `seconds_step`: rollout, scoring and update) plus evals.jsonl (`seconds`: the Val
evaluations). Sampling: every `<runs>/colab-s1/*/meta.json` (Val) and `<runs>/final/*/meta.json` (locked Test): `wall_time_s` is the whole process of the last invocation (model load
included), `calls[].wall_time_s` the generation calls of every invocation. A run directory without meta.json (copied results, CPU controls) is counted and listed, never estimated.
Runtime sessions come from `results/compute_sessions.json` (hand-kept, from the agent session log: first and last call of each Colab runtime). Not included anywhere: weight downloads, package
installs, LoRA merges (about 80 s each on the A100 runtime, not logged per run), CPU-only stages (data preparation, controls, scoring) and the time of the agents that wrote the code.
"""
import argparse
import datetime
import json
import re
from pathlib import Path

SAMPLING_STAGES = [
    (r"^(ident|name)-", "Diagnostics: identification and name-given"),
    (r"shuffle|blank", "Diagnostics: shuffled and blank profile"),
    (r"^(rlfull|r1s|r3s)", "RL checkpoints: Val re-sampling"),
    (r"^a1", "A1 Val sampling"),
    (r"^(a2|g-a2)", "A2 Val sampling"),
    (r"^a4", "A4 Val sampling"),
    (r"^(b9|g-9b)", "9B untrained"),
    (r"^(b27|b38)", "27B untrained (Qwen3.5 and Qwen3.8)"),
]
TRAINING = [("SFT A2 (answer-only, 3 epochs)", "sft/A2"), ("SFT A1 (reasoning traces, 2 epochs)", "sft/A1"), ("RL-full pilot R1 (40 steps)", "rl/R1"),
            ("RL-contrastive R3 (cut by a runtime loss)", "rl/R3"), ("RL-full replicate R5 (seed 1)", "rl/R5"), ("RL-validity R2", "rl/R2"), ("RL-profilefree R4", "rl/R4")]


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def training_row(label, directory):
    """One training run: GPU seconds from its records, or None when the run has no record here."""
    d = Path(directory)
    if (d / "metrics.jsonl").is_file():                                   # an RL run
        steps = jsonl(d / "metrics.jsonl")
        evals = jsonl(d / "evals.jsonl") if (d / "evals.jsonl").is_file() else []
        status = json.loads((d / "summary.json").read_text(encoding="utf-8")).get("status") if (d / "summary.json").is_file() else None
        return {"label": label, "kind": "RL", "steps": len(steps), "status": status, "step_seconds": sum(r["seconds_step"] for r in steps),
                "eval_seconds": sum(e.get("seconds", 0.0) for e in evals), "seconds": sum(r["seconds_step"] for r in steps) + sum(e.get("seconds", 0.0) for e in evals)}
    if (d / "summary.json").is_file():                                    # an SFT run
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        return {"label": label, "kind": "SFT", "steps": s.get("global_step"), "status": s.get("status"), "seconds": float(s["training_seconds"])}
    return None


def sampling_rows(runs):
    """{stage: {runs, wall_seconds, generation_seconds}}, and the run directories that have no meta.json."""
    stages, no_meta = {}, []
    for root, forced in ((Path(runs) / "colab-s1", None), (Path(runs) / "final", "Locked Test evaluation")):
        for d in sorted(p for p in root.glob("*") if p.is_dir()):
            meta = d / "meta.json"
            if not meta.is_file():
                no_meta.append(f"{root.name}/{d.name}")
                continue
            m = json.loads(meta.read_text(encoding="utf-8"))
            stage = forced or next((label for pattern, label in SAMPLING_STAGES if re.search(pattern, d.name)), "4B untrained and prompt probes")
            row = stages.setdefault(stage, {"runs": 0, "wall_seconds": 0.0, "generation_seconds": 0.0})
            row["runs"] += 1
            row["wall_seconds"] += float(m.get("wall_time_s") or 0.0)
            row["generation_seconds"] += sum(float(c.get("wall_time_s", 0.0)) for c in m.get("calls") or [])
    return stages, no_meta


def session_hours(sessions):
    out = []
    as_of = datetime.datetime.fromisoformat(sessions["as_of"].replace("Z", "+00:00"))
    for s in sessions["runtimes"]:
        if "hours" in s:                                                  # a quota figure with no clock times (a Kaggle probe)
            out.append(dict(s))
            continue
        a = datetime.datetime.fromisoformat(s["start"].replace("Z", "+00:00"))
        b = datetime.datetime.fromisoformat(s["end"].replace("Z", "+00:00")) if s.get("end") else as_of
        out.append({**s, "hours": (b - a).total_seconds() / 3600})
    return out


def build(runs, sessions_path):
    train = [r for label, sub in TRAINING if (r := training_row(label, Path(runs) / sub))]
    sampling, no_meta = sampling_rows(runs)
    result = {"training": train, "sampling": sampling, "runs_without_meta": no_meta}
    if Path(sessions_path).is_file():
        sessions = json.loads(Path(sessions_path).read_text(encoding="utf-8"))
        result["runtime_sessions"] = {"as_of": sessions["as_of"], "runtimes": session_hours(sessions), "note": sessions.get("note", "")}
    result["totals_hours"] = {"training": sum(r["seconds"] for r in train) / 3600, "sampling_wall": sum(v["wall_seconds"] for v in sampling.values()) / 3600,
                              "sampling_generation": sum(v["generation_seconds"] for v in sampling.values()) / 3600}
    return result


def markdown(result):
    lines = ["| Stage | Runs or steps | GPU hours | Note |", "|---|---|---|---|"]
    for r in result["training"]:
        note = f"{r['kind']}; status {r['status']}" + (f"; includes {r['eval_seconds'] / 60:.0f} min of Val evaluations" if r["kind"] == "RL" else "")
        lines.append(f"| {r['label']} | {r['steps']} steps | {r['seconds'] / 3600:.2f} | {note} |")
    for stage, v in sorted(result["sampling"].items()):
        lines.append(f"| {stage} | {v['runs']} runs | {v['wall_seconds'] / 3600:.2f} | vLLM; {v['generation_seconds'] / 3600:.2f} h of it generation |")
    t = result["totals_hours"]
    lines.append(f"| **Total logged** | | **{t['training'] + t['sampling_wall']:.2f}** | training {t['training']:.2f} h, sampling {t['sampling_wall']:.2f} h |")
    if "runtime_sessions" in result:
        lines += ["", f"Colab runtime sessions (as of {result['runtime_sessions']['as_of']}):", "", "| Runtime | GPU | Hours | How it ended |", "|---|---|---|---|"]
        lines += [f"| {s.get('start', 'quota')} to {s.get('end') or ('running' if 'start' in s else 'quota')} | {s['gpu']} | {s['hours']:.2f} | {s.get('note', '')} |" for s in result["runtime_sessions"]["runtimes"]]
    if result["runs_without_meta"]:
        lines += ["", f"Run directories without a meta.json, not counted ({len(result['runs_without_meta'])}): " + ", ".join(result["runs_without_meta"])]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--sessions", default="results/compute_sessions.json")
    ap.add_argument("--out", default="results/compute_budget.json")
    ap.add_argument("--md", default="docs/paper/compute_budget.md")
    args = ap.parse_args(argv)
    result = build(args.runs, args.sessions)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
