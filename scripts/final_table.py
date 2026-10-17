#!/usr/bin/env python3
"""The arms table of a split (docs/prereg-v1.md sections 3 and 4): one row per (arm, split) from the samples of the locked evaluation, with the columns and the
descriptive component-bootstrap intervals of scripts/val_table.py. Aggregates only: no prompt, sample, answer, molecule string or drug name is written.

  .venv/bin/python scripts/final_table.py --final runs/final --splits test_t,test_r --out results/test_baselines.json --allow-test \
      [--spec results/frozen_arms.json] [--md results/test_table.md] [--boot 2000] [--data data/processed]

For every <final>/<id>/ that has a samples.jsonl: `rlexp.metrics score` (only when scored.v3.jsonl is missing or older than the samples), then ONE
`rlexp.metrics report --split <splits> --seed 0 --json <final>/<id>/report.v3.json` (the same numbers as one call per split: the bootstrap is seeded per split).
An arm with no rows on a requested split (a 27B on test_t only) gets rows only for the splits it has. Blank-profile arms (id with `blank`, or spec data "blank")
show every first-valid column as "n/a (shared draws)" in the markdown (every row has the same prompt, so the draws repeat; read mean similarity and validity)
and keep the numbers in the JSON (`shared_draws`). Identification arms (`ident-*`, or spec data "ident") are scored by `rlexp.ident score` (name match, pass@k,
answered, drugs), never by SMILES. With --spec the label and decoding of an arm come from it (label, temperature, k, mode, splits) and its arms that have no
samples, or none on a split it should cover, are listed under `gaps`; without it the label is the id. The header holds the constant-answer bar and C3's value per split.
Table order: controls, 4B arms, references, diagnostics; inside a family the order of the spec (the run list), then the id with numbers read as numbers (9B before 27B). A Test split needs --allow-test (the locked evaluation, after the prereg-v1 tag), like make_figures.py.
The scored files are only re-made when older than the samples: delete them to score again after a change to rlexp.metrics. Each arm's full report is kept as report.v3.json beside its
samples. The output is deterministic (no timestamps); the splits are named in every row, table and heading.
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

SPLIT_NAMES = {"val": "Val", "test_r": "Test-R", "test_t": "Test-T"}
TEST_SPLITS = ("test_r", "test_t")
# copied from scripts/val_table.py (loaded scripts are not edited): the columns, and value() below
COLUMNS = ("valid_closed", "exact@1", "core@1", "mean_tanimoto", "fv_tanimoto", "fv_samples", "fv_tokens", "fv_lift", "lift", "tanimoto_valid", "core@8", "fv_rate",
           "mean_reward")
CI_METRICS = ("lift", "fv_tanimoto", "mean_tanimoto", "fv_lift")          # descriptive component-bootstrap intervals, drug-level
FIRST_VALID = ("fv_tanimoto", "fv_samples", "fv_tokens", "fv_lift", "fv_rate")     # undefined for a run whose prompts are all the same
BAR = ("fv_tanimoto", "mean_tanimoto", "mean_reward", "lift", "exact@1", "core@1")      # the constant-answer bar; its molecule is never copied
FAMILIES = ("Control", "4B arm", "Reference", "Diagnostic")               # table order
FAMILY_OF = ((0, r"^C\d+$|control"), (3, r"(^|-)(shuffle|blank)(-|$)|^(name|ident)(-|$)"), (2, r"^\d+B(-|$)"))      # first match wins; anything else is a 4B arm
# (key, header, decimals, signed): the headers of make_figures.py. prompt_tokens stays in the JSON only: it is the original prompt's length, which a retrieval arm exceeds
MD_COLUMNS = (("valid_closed", "Valid answers", 3, False), ("fv_tanimoto", "Delivered similarity (95% interval)", 3, False), ("fv_samples", "Samples per answer", 2, False),
              ("fv_tokens", "Tokens per answer", 1, False), ("fv_lift", "Delivered lift", 3, True), ("lift", "Lift, all samples", 3, True),
              ("mean_tanimoto", "Mean similarity, all samples", 3, False), ("core@1", "Core match", 3, False), ("exact@1", "Exact match", 3, False))
SHARED = "n/a (shared draws)"
NOTES = {"prompt_tokens": "mean tokens of the ORIGINAL prompt of the rows (arm files keep that count): too many for the blank-profile arms, too few for the retrieval, name-given and "
                          "identification arms; fv_tokens + prompt_tokens is a cost only for arms that read the original prompt",
         "constant_bar": "the best constant Train reference scored on the rows of the run named in `from`; the molecule is not copied"}
PLAIN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")       # a run directory name: no separator, no leading dot


# ------------------------------------------------------------------------------------------------- shared with paired_table.py and exposure_terciles.py
def value(metrics, name):
    m = metrics.get(name)
    return None if m is None else (m["value"] if isinstance(m, dict) else m)


def clean(text):
    """A CLI error without the lists some messages quote: they hold row ids, which carry the drug name."""
    return re.sub(r"\[[^\]]*\]", "[...]", text)


def run_cli(module, *args):
    """stdout of `python -m <module> <args>`; exit with a short cleaned message when it fails."""
    done = subprocess.run([sys.executable, "-m", module, *args], capture_output=True, text=True, check=False)
    if done.returncode:
        sys.exit(f"error: {module} {args[0]} failed: {clean(done.stderr.strip())[-300:]}")
    return done.stdout


def check_splits(splits, allow_test):
    """Exit unless every split is known; a Test split also needs --allow-test (make_figures.py's rule)."""
    unknown = [s for s in splits if s not in SPLIT_NAMES]
    if not splits or unknown:
        sys.exit(f"error: splits must come from {list(SPLIT_NAMES)}, got {unknown or 'none'}")
    tests = [s for s in splits if s in TEST_SPLITS]
    if tests and not allow_test:
        sys.exit(f"refused: {', '.join(tests)} is a Test split; --allow-test is for the locked evaluation after the prereg-v1 tag only")


def plain(arm):
    """True for an arm id that is a plain run directory name (it is joined to --final, so no path may hide in it)."""
    return PLAIN_ID.fullmatch(arm) is not None and ".." not in arm


def cell(v, decimals, signed):
    """A table cell (make_figures.py's): n/a for a missing or non-finite value, -0.0 printed as 0.0."""
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
        return "n/a"
    v = round(v, decimals) + 0.0
    return f"{v:+.{decimals}f}" if signed else f"{v:.{decimals}f}"


# ------------------------------------------------------------------------------------------------- arms
def natural(text):
    """Sort key that reads digits as numbers: 9B before 27B-3.5."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def load_spec(path):
    """({arm id: spec arm}, the spec's splits) of a frozen spec, ({}, None) without one."""
    if path is None:
        return {}, None
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
        return {a["id"]: a for a in spec["arms"]}, spec.get("splits")
    except (OSError, ValueError, KeyError, TypeError) as e:
        sys.exit(f"error: cannot read the spec {path}: {type(e).__name__}")


def family(run, arm):
    """Index into FAMILIES: the spec's role when the arm is in it (a selected arm has none), else the pattern of its id."""
    if arm:
        return {"reference": 2, "diagnostic": 3, "probe": 3}.get(arm.get("role"), 1)
    return next((i for i, pattern in FAMILY_OF if re.search(pattern, run, re.IGNORECASE)), 1)


def decoding(arm, fam):
    if arm is None:
        return "deterministic, copied to eight samples" if fam == 0 else "n/a"
    parts = [f"T={arm['temperature']}", f"k={arm['k']}", {"think": "think mode", "nothink": "no-think"}.get(arm["mode"], str(arm["mode"]))]
    if arm.get("splits"):
        parts.append(" and ".join(SPLIT_NAMES.get(s, s) for s in arm["splits"]) + " only")
    return ", ".join(parts)


def is_kind(run, arm, name):
    return (arm or {}).get("data") == name or re.search(rf"(^|-){name}(-|$)", run, re.IGNORECASE) is not None


def ensure_scored(samples, data):
    """scored.v3.jsonl beside the samples, made by `rlexp.metrics score` when it is missing or older than them."""
    scored = samples.with_name("scored.v3.jsonl")
    if not scored.is_file() or scored.stat().st_mtime < samples.stat().st_mtime:
        run_cli("rlexp.metrics", "score", "--samples", str(samples), "--data", str(data), "--out", str(scored))
    return scored


def metric_row(sp):
    m = sp["metrics"]
    row = {"drugs": sp["n_drugs"], "samples": sp["n_samples"], "prompt_tokens": sp.get("prompt_tokens"), **{c: value(m, c) for c in COLUMNS}}
    for name in CI_METRICS:
        e = m.get(name)
        if isinstance(e, dict) and e.get("lo") is not None:
            row[f"{name}_ci"] = [e["lo"], e["hi"]]
    return row


def ident_rows(samples, data, curated, splits):
    """{split: row} of `rlexp.ident score` (one JSON line per split the samples name): name match, pass@k with its k, answered and drugs."""
    rows = {}
    for line in run_cli("rlexp.ident", "score", "--samples", str(samples), "--data", str(data), "--curated", str(curated)).splitlines():
        r = json.loads(line) if line.strip() else None
        if r and r["split"] in splits:
            rows[r["split"]] = {"drugs": r["drugs"], "samples": r["samples"], "name_match@1": r["name_match@1"], "pass_k": r["pass_k"],
                                "name_pass": r[f"name_pass@{r['pass_k']}"], "answered": r["answered"]}
    return rows


def arm_rows(d, arm, args, splits):
    """({split: row without the arm's identity}, {split: constant-answer bar block}) of one run directory."""
    if is_kind(d.name, arm, "ident"):
        return {s: {"kind": "identification", **r} for s, r in ident_rows(d / "samples.jsonl", args.data, args.curated, splits).items()}, {}
    scored, report = ensure_scored(d / "samples.jsonl", args.data), d / "report.v3.json"
    run_cli("rlexp.metrics", "report", "--scored", str(scored), "--split", ",".join(splits), "--data", str(args.data), "--boot", str(args.boot), "--seed", "0",
            "--json", str(report))
    blocks = {s: b for s, b in json.loads(report.read_text(encoding="utf-8"))["splits"].items() if b.get("n_rows")}
    return {s: {"kind": "generation", **metric_row(b)} for s, b in blocks.items()}, {s: b["constant"] for s, b in blocks.items() if "constant" in b}


def pick_bar(candidates):
    """The constant-answer bar of a split from the first run (table order) whose bar has a first-valid value, else the first bar; its molecule is not copied."""
    chosen = next((c for c in candidates if value(c[1]["metrics"], "fv_tanimoto") is not None), candidates[0] if candidates else None)
    return None if chosen is None else {"from": chosen[0], **{k: value(chosen[1]["metrics"], k) for k in BAR}}


def collect(args, splits, spec, spec_splits):
    """(rows, bars, gaps): rows sorted by split, family and id; bars {split: bar}; gaps the arms or splits the spec expects and the runs lack."""
    final = args.final
    if not final.is_dir():
        sys.exit(f"error: {final} is not a directory")
    dirs = sorted((d for d in final.iterdir() if (d / "samples.jsonl").is_file()), key=lambda d: natural(d.name))
    if not dirs:
        sys.exit(f"error: no <id>/samples.jsonl under {final}")
    rows, constants, gaps = [], {}, [{"run": i, "split": None} for i in sorted(set(spec) - {d.name for d in dirs}, key=natural)]
    for d in dirs:
        arm = spec.get(d.name)
        fam = family(d.name, arm)
        found, bar = arm_rows(d, arm, args, splits)
        print(f"{d.name}: {', '.join(s for s in splits if s in found) or 'no requested split'}", file=sys.stderr, flush=True)
        expected = (arm or {}).get("splits") or spec_splits or splits
        gaps += [{"run": d.name, "split": s} for s in splits if arm and s in expected and s not in found]
        constants.update({(d.name, s): c for s, c in bar.items()})
        rows += [{"run": d.name, "label": (arm or {}).get("label", d.name), "decoding": decoding(arm, fam), "split": s, "family": FAMILIES[fam],
                  "shared_draws": is_kind(d.name, arm, "blank"), **r} for s, r in found.items()]
    order, listed = {s: i for i, s in enumerate(splits)}, {a: i for i, a in enumerate(spec)}      # within a family: the spec's order (the run list), then the id
    rows.sort(key=lambda r: (order[r["split"]], FAMILIES.index(r["family"]), listed.get(r["run"], len(listed)), natural(r["run"])))
    bars = {s: pick_bar([(r["run"], constants[r["run"], s]) for r in rows if r["split"] == s and (r["run"], s) in constants]) for s in splits}
    return rows, bars, gaps


# ------------------------------------------------------------------------------------------------- markdown
def md_cell(row, key, decimals, signed):
    if row["shared_draws"] and key in FIRST_VALID:
        return SHARED
    text, ci = cell(row.get(key), decimals, signed), row.get(f"{key}_ci") if key == "fv_tanimoto" else None        # the interval of the endpoint only
    return f"{text} [{cell(ci[0], decimals, signed)}, {cell(ci[1], decimals, signed)}]" if ci and text != "n/a" else text


def split_md(split, rows, bar, c3, gaps):
    name = SPLIT_NAMES[split]
    mine = [r for r in rows if r["split"] == split]
    gen, ident = [r for r in mine if r["kind"] == "generation"], [r for r in mine if r["kind"] == "identification"]
    out = [f"## {name}", ""]
    if gen:
        out += ["| Arm | Group | Decoding | " + " | ".join(h for _, h, _, _ in MD_COLUMNS) + " |", "|---|---|---|" + "---:|" * len(MD_COLUMNS)]
        out += [f"| {r['label'].replace('|', '/')} | {r['family']} | {r['decoding'].replace('|', '/')} | "
                + " | ".join(md_cell(r, k, d, sg) for k, _, d, sg in MD_COLUMNS) + " |" for r in gen]
        out += ["", f"{name} split, {max(r['drugs'] for r in gen)} drugs. Similarity is the Tanimoto similarity (Morgan count vectors) to the reference structure. "
                "Delivered similarity and lift are those of the first valid, finished answer among up to eight samples of each prompt (0 if none), with its cost in "
                "samples and generated tokens per answer (prompts are not counted, and the arm with retrieved examples reads a longer one than the others). The bracket is a 95% interval from resampling groups "
                "of related drugs: descriptive, not a test. Lift is the similarity to the drug's own structure minus the mean similarity to the other drugs' structures. "
                "Core match and exact match are the share of samples with the reference's scaffold or connectivity (core) or its connectivity (exact), averaged over "
                "drugs. n/a: fewer than eight samples per prompt, so no first-valid value."]
        if any(r["shared_draws"] for r in gen):
            out += [f"{SHARED}: a run with the profile removed shows every prompt the same text, so its draws repeat across prompts and its first valid answer is "
                    "one shared molecule; read its mean similarity and valid answers."]
    else:
        out += ["No arm with generated structures has samples on this split."]
    if bar:
        out += ["", f"Best constant answer (one fixed structure chosen from the training drugs): mean similarity {cell(bar['mean_tanimoto'], 3, False)}, "
                f"delivered similarity {cell(bar['fv_tanimoto'], 3, False)}."]
    if c3:
        out += [f"C3, the reward-optimal constant (what a policy that ignores the profile can at best learn): delivered similarity {cell(c3['fv_tanimoto'], 3, False)}, "
                f"mean similarity {cell(c3['mean_tanimoto'], 3, False)}."]
    if ident:
        out += ["", "Identification (profile in, drug name out; a sample counts when its first answer line equals the drug's name):", "",
                "| Arm | Decoding | Name match | Pass at k | k | Answered | Drugs |", "|---|---|---:|---:|---:|---:|---:|"]
        out += [f"| {r['label'].replace('|', '/')} | {r['decoding'].replace('|', '/')} | {cell(r['name_match@1'], 3, False)} | {cell(r['name_pass'], 3, False)} | "
                f"{r['pass_k']} | {cell(r['answered'], 3, False)} | {r['drugs']} |" for r in ident]
    missing = [g["run"] + (f" on {SPLIT_NAMES[g['split']]}" if g["split"] else "") for g in gaps if g["split"] in (split, None)]
    return out + (["", "Planned but not run: " + ", ".join(missing) + "."] if missing else []) + [""]


def markdown(out):
    lines = ["# Arms table", ""]
    for s in out["splits"]:
        lines += split_md(s, out["rows"], out["constant_bar"][s], out["c3"][s], out["gaps"])
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", type=Path, default=Path("runs/final"))
    ap.add_argument("--splits", default="test_t,test_r")
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--curated", type=Path, default=Path("data/curated/chembl_refs.json"), help="drug names for the identification arms")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spec", type=Path, help="results/frozen_arms.json: labels, decoding and the arms and splits that should exist")
    ap.add_argument("--md", type=Path)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--allow-test", action="store_true", help="accept a Test split (locked evaluation, after the prereg-v1 tag)")
    args = ap.parse_args(argv)
    splits = list(dict.fromkeys(s for s in args.splits.split(",") if s))
    check_splits(splits, args.allow_test)
    spec, spec_splits = load_spec(args.spec)
    rows, bars, gaps = collect(args, splits, spec, spec_splits)
    if not rows:
        sys.exit(f"error: no arm has rows on {', '.join(splits)}")
    c3 = {s: next(({k: r[k] for k in ("fv_tanimoto", "mean_tanimoto")} for r in rows if r["run"] == "C3" and r["split"] == s and r["kind"] == "generation"), None)
          for s in splits}
    out = {"splits": splits, "boot": args.boot, "seed": 0, "columns": COLUMNS, "notes": NOTES, "constant_bar": bars, "c3": c3, "gaps": gaps, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    text = markdown(out)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {args.out}: {len(rows)} rows, {len({r['run'] for r in rows})} arms, splits {', '.join(splits)}")


if __name__ == "__main__":
    main()
