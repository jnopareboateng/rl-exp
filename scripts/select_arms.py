#!/usr/bin/env python3
"""Pick, for every arm, the decoding (temperature) and RL checkpoint with the best Val fv_tanimoto, then the confirmatory pair
(docs/prereg-v1.md sections 4 and 5). Reads the per-run Val reports written by scripts/val_table.py (runs/<root>/<run>/report.v3.json).

  .venv/bin/python scripts/select_arms.py --candidates results/arm_candidates.json --runs runs/colab-s1 --out results/selection.json

candidates file: {"arms": {id: {"label", "candidates": [{"run", "temperature", "checkpoint"?}]}}, "comparator_set": [ids], "rl_set": [ids]}.
Selection key: Val fv_tanimoto; CPU controls (one sample per row, no fv) use mean_tanimoto, which equals it for a deterministic answer. Ties go to the
lower temperature, then the earlier checkpoint, then the run name. The comparator is the arm of comparator_set with the best selected score; the
RL arm is the best of rl_set. Nothing here reads Test.
"""
import argparse
import json
import sys
from pathlib import Path


def val_score(report):
    """Val fv_tanimoto of a metrics report (dict from report.v3.json), else mean_tanimoto; None if neither exists."""
    metrics = report["splits"]["val"]["metrics"]
    for key in ("fv_tanimoto", "mean_tanimoto"):
        m = metrics.get(key)
        value = m["value"] if isinstance(m, dict) else m
        if value is not None and value == value:
            return float(value), key
    return None, None


def step_of(candidate):
    checkpoint = candidate.get("checkpoint") or ""
    digits = "".join(ch for ch in checkpoint if ch.isdigit())
    return int(digits) if digits else 0


def pick(arm, load_report):
    """The best candidate of one arm: dict with run, temperature, checkpoint, score, metric, and every candidate's score for the record."""
    scored = []
    for c in arm["candidates"]:
        score, metric = val_score(load_report(c["run"]))
        if score is not None:
            scored.append((-score, c.get("temperature") if c.get("temperature") is not None else 0.0, step_of(c), c["run"], c, score, metric))
    if not scored:
        raise ValueError("an arm has no candidate with a Val score")
    scored.sort(key=lambda t: t[:4])
    best = scored[0]
    return {"run": best[4]["run"], "temperature": best[4].get("temperature"), "checkpoint": best[4].get("checkpoint"), "score": best[5], "metric": best[6],
            "candidates": [{"run": t[4]["run"], "temperature": t[4].get("temperature"), "checkpoint": t[4].get("checkpoint"), "score": t[5]} for t in scored]}


def select(spec, load_report):
    chosen = {arm_id: {"label": arm.get("label", arm_id), **pick(arm, load_report)} for arm_id, arm in spec["arms"].items()}
    for family in ("comparator_set", "rl_set"):
        missing = [a for a in spec[family] if a not in chosen]
        if missing:
            raise ValueError(f"{family} names arms without candidates: {missing}")
    best = lambda ids: max(ids, key=lambda a: (chosen[a]["score"], a))
    return {"arms": chosen, "comparator": best(spec["comparator_set"]), "rl": best(spec["rl_set"]),
            "comparator_set": spec["comparator_set"], "rl_set": spec["rl_set"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("runs/colab-s1"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    spec = json.loads(args.candidates.read_text(encoding="utf-8"))

    def load_report(run):
        return json.loads((args.runs / run / "report.v3.json").read_text(encoding="utf-8"))

    try:
        result = select(spec, load_report)
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"error: {type(e).__name__}: {e}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"comparator": result["comparator"], "rl": result["rl"],
                      "scores": {a: round(v["score"], 4) for a, v in result["arms"].items()}}))


if __name__ == "__main__":
    main()
