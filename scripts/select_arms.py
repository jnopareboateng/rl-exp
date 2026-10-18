#!/usr/bin/env python3
"""Pick, for every arm, the decoding (temperature) and RL checkpoint with the best Val fv_tanimoto, then the confirmatory pair
(docs/prereg-v1.md sections 4 and 5). Reads the per-run Val reports written by scripts/val_table.py (runs/<root>/<run>/report.v3.json).

  .venv/bin/python scripts/select_arms.py --candidates results/arm_candidates.json --runs runs/colab-s1 --out results/selection.json

candidates file: {"arms": {id: {"label", "spec"?: {"model", "mode", "max_tokens", "data", "system", "quantization"?, "gpu_memory_utilization"?},
"candidates": [{"run", "temperature", "checkpoint"?, "adapter_sha256"?}]}}, "comparator_set": [ids], "rl_set": [ids]}. An arm's optional "spec" block (what its Test run must
repeat: gpu/final_eval.py checks the frozen spec against it) is validated and copied into the selection. Its model is the hub id of an untrained arm (no adapter_sha256) and
final_models/<id> for an arm with a checkpoint. The samples of the winning run (runs/<root>/<run>/samples.jsonl) must agree: their mode, max_tokens and temperature are the block's
and the candidate's, and for an untrained arm their model is the block's (an adapter arm's Val run sampled a merged path of its own, which is not compared).
Selection key: Val fv_tanimoto and nothing else. A candidate whose report has no fv_tanimoto, or whose Val block was not computed from at least 8 samples for
every Val prompt row (a k<8 run, a partial or resumed run, a report made without --data), is an error: never a fallback to another metric and never a
silent omission. The CPU controls (one sample per row) enter as their k=8 replications (scripts/replicate_samples.py), where fv_tanimoto equals mean_tanimoto.
Within an arm, ties go to the lower temperature, then the earlier checkpoint (`step-<n>` or `s<n>`), then the run name. Between arms, for the comparator
(best of comparator_set) and the RL arm (best of rl_set): the highest score, a tie going to the SMALLER arm id.
A candidate's adapter_sha256 (sha256 of its adapter_model.safetensors) is carried into its arm's record: gpu/final_eval.py checks the frozen spec
against it. Nothing here reads Test.
"""
import argparse
import json
import re
import sys
from pathlib import Path

SPEC_KEYS = ("model", "mode", "max_tokens", "data", "system")
SPEC_OPTIONAL = ("quantization", "gpu_memory_utilization")
RUN_FIELDS = ("model", "mode", "temperature", "max_tokens")      # sampler settings that every sample row of a run repeats
RUN = re.compile(r"[A-Za-z0-9._-]+")                     # a run directory name: no separator (and, below, no "..", no ".")
STEP = re.compile(r"(?:step-|s)(\d+)")
SHA256 = re.compile(r"[0-9a-f]{64}")


def val_score(report):
    """Val fv_tanimoto of a metrics report (dict from report.v3.json). ValueError when the report has none, or when it does not cover every Val prompt
    row with at least 8 samples (a k<8 or partial run), or cannot say so (no coverage block: the report was made without --data)."""
    val = report["splits"]["val"]
    m = val["metrics"].get("fv_tanimoto") or {}
    value = m.get("value")
    if value is None or value != value:
        raise ValueError("the Val report has no fv_tanimoto (fewer than 8 samples per row, or no valid row)")
    cover = val.get("coverage")
    if not cover:
        raise ValueError("the Val report has no coverage block (made without --data), so it cannot show 8 samples for every Val row")
    if (m.get("n_rows"), m.get("n_drugs")) != (cover["rows"], cover["drugs"]):
        raise ValueError(f"fv_tanimoto covers {m.get('n_rows')} of {cover['rows']} Val rows and {m.get('n_drugs')} of {cover['drugs']} drugs: "
                         "fewer than 8 samples per row, or a partial run")
    return float(value)


def step_of(candidate):
    """The step of a candidate's checkpoint, `step-<n>` or `s<n>` as a whole string; 0 for a candidate without a checkpoint; anything else is an error."""
    checkpoint = candidate.get("checkpoint")
    if checkpoint is None or checkpoint == "":
        return 0
    match = isinstance(checkpoint, str) and STEP.fullmatch(checkpoint)
    if not match:
        raise ValueError(f"checkpoint {checkpoint!r} is neither step-<n> nor s<n>")
    return int(match.group(1))


def pick(arm, load_report):
    """The best candidate of one arm: dict with run, temperature, checkpoint, score, metric, every candidate's score for the record and, if the winning candidate
    names one, its adapter_sha256. A candidate without a usable Val score is an error, not a candidate that drops out."""
    scored = []
    for c in arm["candidates"]:
        run = c["run"]
        if not (isinstance(run, str) and RUN.fullmatch(run) and ".." not in run and run != "."):
            raise ValueError(f"run name {run!r} is not a plain run directory name")
        if c.get("adapter_sha256") is not None and not (isinstance(c["adapter_sha256"], str) and SHA256.fullmatch(c["adapter_sha256"])):
            raise ValueError(f"candidate {run}: adapter_sha256 is not a lowercase sha256 hex string")
        try:
            score = val_score(load_report(run))
        except ValueError as e:
            raise ValueError(f"candidate {run}: {e}") from None
        scored.append((-score, c.get("temperature") if c.get("temperature") is not None else 0.0, step_of(c), run, c, score))
    if not scored:
        raise ValueError("an arm has no candidates")
    scored.sort(key=lambda t: t[:4])
    best = scored[0][4]
    record = {"run": best["run"], "temperature": best.get("temperature"), "checkpoint": best.get("checkpoint"), "score": scored[0][5], "metric": "fv_tanimoto",
              "candidates": [{"run": t[4]["run"], "temperature": t[4].get("temperature"), "checkpoint": t[4].get("checkpoint"), "score": t[5]} for t in scored]}
    if best.get("adapter_sha256") is not None:
        record["adapter_sha256"] = best["adapter_sha256"]
    return record


def arm_spec(arm_id, arm):
    """The arm's "spec" block, checked: model, mode (think|nothink), max_tokens (positive integer), data and system (names), and optionally quantization (a name) and
    gpu_memory_utilization (a fraction); nothing else. {} when the arm has no block."""
    block = arm.get("spec")
    if block is None:
        return {}
    good = (isinstance(block, dict) and set(SPEC_KEYS) <= set(block) <= {*SPEC_KEYS, *SPEC_OPTIONAL} and block["mode"] in ("think", "nothink")
            and type(block["max_tokens"]) is int and block["max_tokens"] >= 1 and all(isinstance(block[k], str) and block[k] for k in ("model", "data", "system", "quantization") if k in block)
            and (block.get("gpu_memory_utilization") is None or (type(block["gpu_memory_utilization"]) in (int, float) and 0 < block["gpu_memory_utilization"] <= 1)))
    if not good:
        raise ValueError(f"arm {arm_id}: spec must be exactly {list(SPEC_KEYS)} (mode think or nothink, max_tokens a positive integer, the others names), "
                         f"and may add {list(SPEC_OPTIONAL)}")
    return {"spec": dict(block)}


def check_run(arm_id, best, block, run_settings):
    """The winning run was sampled the way the candidates file says: every sample row's mode and max_tokens are the block's, its temperature the candidate's, and for an
    untrained arm (no adapter_sha256) its model the block's. A field the rows do not carry, or a run without samples, is not compared. ValueError names the run."""
    seen = run_settings(best["run"])
    wanted = {"mode": block["mode"], "max_tokens": block["max_tokens"], "temperature": best["temperature"]}
    if "adapter_sha256" not in best:
        wanted["model"] = block["model"]
    for field in RUN_FIELDS:
        if field in wanted and field in seen and seen[field] != {wanted[field]}:
            raise ValueError(f"arm {arm_id}: run {best['run']} was sampled with {field} {sorted(seen[field], key=str)} but the candidates file says {wanted[field]!r}")


def run_settings(root, run):
    """{field: set of the distinct values} of model, mode, temperature and max_tokens over <root>/<run>/samples.jsonl; {} without the file."""
    path, seen = Path(root) / run / "samples.jsonl", {}
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    for field in RUN_FIELDS:
                        if field in row:
                            seen.setdefault(field, set()).add(row[field])
    return seen


def select(spec, load_report, load_settings=None):
    """The selection of spec (the candidates file). load_settings(run) -> {field: set of values} (run_settings) enables the comparison of each winning run's sample rows with
    the arm's spec block; without it nothing is compared."""
    chosen = {arm_id: {"label": arm.get("label", arm_id), **arm_spec(arm_id, arm), **pick(arm, load_report)} for arm_id, arm in spec["arms"].items()}
    for arm_id, record in chosen.items():
        if load_settings is not None and "spec" in record:
            check_run(arm_id, record, record["spec"], load_settings)
    for family in ("comparator_set", "rl_set"):
        if not spec[family]:
            raise ValueError(f"{family} is empty")
        missing = [a for a in spec[family] if a not in chosen]
        if missing:
            raise ValueError(f"{family} names arms without candidates: {missing}")
    best = lambda ids: min(ids, key=lambda a: (-chosen[a]["score"], a))        # highest score; a tie goes to the smaller arm id
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
        result = select(spec, load_report, lambda run: run_settings(args.runs, run))
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"error: {type(e).__name__}: {e}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"comparator": result["comparator"], "rl": result["rl"],
                      "scores": {a: round(v["score"], 4) for a, v in result["arms"].items()}}))


if __name__ == "__main__":
    main()
