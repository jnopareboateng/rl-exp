#!/usr/bin/env python3
"""Paired comparisons of two arms on ONE split of the locked evaluation (docs/prereg-v1.md sections 4 and 7), every number from `rlexp.metrics report --vs`.

  .venv/bin/python scripts/paired_table.py --final runs/final --split test_t --pairs RL-full:C3,RL-full:A2 --out results/paired_test_t.json --allow-test \
      [--selection results/selection.json] [--auto] [--md results/paired_test_t.md] [--boot 10000] [--data data/processed]

For each A:B this runs `rlexp.metrics report --scored <final>/A/scored.v3.jsonl --vs <final>/B/scored.v3.jsonl --split S --data <data> --boot N --seed 0` and keeps the paired
block of fv_tanimoto (the confirmatory endpoint, first everywhere), mean_tanimoto, tanimoto_valid, fv_lift, lift and valid_closed: A, B, A minus B (the sign convention
everywhere), the descriptive interval, wins, losses and ties (drugs), the cluster-level sign-flip p-value and its method, the studentized line, drugs and rows. Metrics the
report does not test (tanimoto_valid, valid_closed) have no wins, losses or p-value. Two runs that cover different drugs are paired on the prompt rows they share, exactly as the
CLI does; n is reported with the size of the split's cohort and `partial` when the pair covers less. A pair with no rows in common is listed with n = 0.
A pair is labelled "confirmatory" only for (selection rl, selection comparator) on test_t, every other pair is "exploratory" (no adjustment for multiplicity); and only when the sha256
of --selection equals the selection_sha256 in <final>/manifest.json (the selection the run was frozen against): otherwise a warning is printed and no pair is confirmatory.
--auto (needs --selection) adds: the selected RL arm against every arm of the selection's comparator_set and against each reference arm present (ids like 9B, 27B-3.5), and the
selected RL arm and A2 each against their own `-shuffle` run (the profile test of section 7: real minus shuffled); nothing else. A pair is refused, with the missing files named,
unless both scored files exist. A Test split needs --allow-test (like make_figures.py); the split is named in every block.
"""
import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts/ is not a package
from final_table import SPLIT_NAMES, cell, check_splits, plain, run_cli      # noqa: E402

SCORED = "scored.v3.jsonl"
CONFIRMATORY_SPLIT = "test_t"
SIGN = "A minus B: a positive difference means the first arm of the pair is higher"
# (key, plain-English name, decimals, signed), fv_tanimoto first
METRICS = (("fv_tanimoto", "Delivered similarity (the confirmatory endpoint)", 3, False), ("mean_tanimoto", "Mean similarity, all samples", 3, False),
           ("tanimoto_valid", "Similarity of valid answers", 3, False), ("fv_lift", "Delivered lift", 3, True), ("lift", "Lift, all samples", 3, True),
           ("valid_closed", "Valid answers", 3, False))
FIELDS = ("a", "b", "diff", "wins", "losses", "ties", "p_value", "p_method", "p_unit", "n_clusters", "studentized", "n_drugs", "n_rows")
REFERENCE = re.compile(r"\d+B(-|$)")                       # 9B, 27B-3.5, 27B-3.8


def parse_pairs(text):
    """[(A, B)] from 'A:B,A:C'; an id is a plain run name (no path), and a pair needs two different arms."""
    pairs = []
    for item in filter(None, text.split(",")):
        a, sep, b = item.partition(":")
        if not (sep and plain(a) and plain(b)) or a == b:
            sys.exit(f"error: pair {item!r} is not A:B with two different plain arm ids")
        pairs.append((a, b))
    return pairs


def auto_pairs(selection, present):
    """The standard exploratory set (module docstring): the selected RL arm against its comparator_set and the references present, and the profile tests."""
    try:
        rl, comparators = selection["rl"], selection["comparator_set"]
    except KeyError as e:
        sys.exit(f"error: the selection lacks {e}")
    return ([(rl, c) for c in comparators if c != rl] + [(rl, r) for r in present if REFERENCE.match(r)]
            + [(a, f"{a}-shuffle") for a in dict.fromkeys((rl, "A2"))])


def selection_is_the_runs(selection_path, final):
    """(True, None) when the sha256 of the selection file equals selection_sha256 of <final>/manifest.json (written by gpu/final_eval.py), else (False, why)."""
    manifest = Path(final) / "manifest.json"
    if not manifest.is_file():
        return False, f"{manifest} does not exist"
    try:
        recorded = json.loads(manifest.read_text(encoding="utf-8")).get("selection_sha256")
    except ValueError:
        return False, f"{manifest} is not JSON"
    if recorded != hashlib.sha256(Path(selection_path).read_bytes()).hexdigest():
        return False, f"its sha256 is not the selection_sha256 of {manifest}"
    return True, None


def label(pair, split, selection):
    return "confirmatory" if selection and split == CONFIRMATORY_SPLIT and pair == (selection.get("rl"), selection.get("comparator")) else "exploratory"


def paired_record(final, pair, split, data, boot, tmp, selection):
    """One pair: its label, the common n against the split's cohort, and the paired block of each METRICS entry (None when the runs share no rows)."""
    a, b = pair
    out = Path(tmp) / "pair.json"
    run_cli("rlexp.metrics", "report", "--scored", str(final / a / SCORED), "--vs", str(final / b / SCORED), "--split", split, "--data", str(data), "--boot", str(boot),
            "--seed", "0", "--json", str(out))
    block = json.loads(out.read_text(encoding="utf-8"))["splits"][split]
    paired, cohort = block.get("paired") or {"n_drugs": 0, "n_rows": 0, "metrics": None}, block.get("coverage")
    metrics = None if not paired["metrics"] else {
        m: {**{k: paired["metrics"][m].get(k) for k in FIELDS}, "ci": None if paired["metrics"][m].get("lo") is None else [paired["metrics"][m]["lo"], paired["metrics"][m]["hi"]]}
        for m, *_ in METRICS}
    return {"a": a, "b": b, "split": split, "label": label(pair, split, selection), "n_drugs": paired["n_drugs"], "n_rows": paired["n_rows"], "cohort": cohort,
            "partial": None if cohort is None else paired["n_drugs"] < cohort["drugs"] or paired["n_rows"] < cohort["rows"], "metrics": metrics}


def p_text(p):
    return "n/a" if p is None else f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def pair_md(rec):
    name = SPLIT_NAMES[rec["split"]]
    out = [f"### {rec['a']} minus {rec['b']} ({rec['label']}, {name})", ""]
    if not rec["metrics"]:
        return out + [f"The two runs share no prompt rows on {name}: nothing to compare.", ""]
    size = f"{rec['n_drugs']} common drugs, {rec['n_rows']} prompt rows"
    if rec["cohort"]:
        size += f"; {name} has {rec['cohort']['drugs']} drugs and {rec['cohort']['rows']} rows" + (" (PARTIAL: the pair covers less)" if rec["partial"] else "")
    out += [size + ".", "", f"| Measure | {rec['a']} | {rec['b']} | Difference | 95% interval | Wins / losses / ties | p (sign-flip) | Drugs |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, text, d, signed in METRICS:
        e = rec["metrics"][key]
        ci = "n/a" if e["ci"] is None else f"[{cell(e['ci'][0], d, True)}, {cell(e['ci'][1], d, True)}]"
        wlt = "n/a" if e["wins"] is None or not e["n_drugs"] else f"{e['wins']} / {e['losses']} / {e['ties']}"
        out.append(f"| {text} | {cell(e['a'], d, False)} | {cell(e['b'], d, False)} | {cell(e['diff'], d, True)} | {ci} | {wlt} | {p_text(e['p_value'])} | {e['n_drugs']} |")
    return out + [""]


def markdown(res):
    name = SPLIT_NAMES[res["split"]]
    out = [f"# Paired comparisons on {name}", "", "Differences are A minus B: a positive difference means the first arm of the pair is higher. Wins, losses and ties count drugs. "
           "p is the two-sided sign-flip test on whole groups of related drugs; the interval is a descriptive bootstrap, not a test. Only a pair marked confirmatory "
           "(the Val-selected RL arm against the Val-selected comparator on Test-T) is a registered test, and for it only the delivered similarity; every other number is "
           "exploratory, with no adjustment for multiplicity.", ""]
    for rec in res["pairs"]:
        out += pair_md(rec)
    if res.get("selection_is_the_runs") is False:
        out += ["The selection file given is not the one the run was frozen against (its sha256 differs from the run manifest's): no pair above is registered, every one is exploratory."]
    if any(rec["b"].endswith("-shuffle") for rec in res["pairs"]):
        out += ["A pair whose second arm is a shuffled-profile run is the profile test of the protocol: the real profile minus another drug's profile, on the same prompts."]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", type=Path, default=Path("runs/final"))
    ap.add_argument("--split", required=True)
    ap.add_argument("--pairs", default="", help="A:B,A:C: arm ids (run directories under --final), A minus B")
    ap.add_argument("--selection", type=Path, help="results/selection.json: marks the confirmatory pair; the arms --auto uses")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--allow-test", action="store_true", help="accept a Test split (locked evaluation, after the prereg-v1 tag)")
    args = ap.parse_args(argv)
    check_splits([args.split], args.allow_test)
    if args.auto and not args.selection:
        sys.exit("error: --auto needs --selection")
    selection = None
    if args.selection:
        try:
            selection = json.loads(args.selection.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            sys.exit(f"error: cannot read the selection {args.selection}: {type(e).__name__}")
        if not isinstance(selection, dict):
            sys.exit(f"error: the selection {args.selection} is not a JSON object")
    if not args.final.is_dir():
        sys.exit(f"error: {args.final} is not a directory")
    pairs = parse_pairs(args.pairs)
    if args.auto:
        pairs += auto_pairs(selection, sorted(d.name for d in args.final.iterdir() if (d / SCORED).is_file()))
    pairs = list(dict.fromkeys(pairs))
    if not pairs:
        sys.exit("error: no pairs: give --pairs A:B,... and/or --auto")
    missing = sorted({str(args.final / arm / SCORED) for pair in pairs for arm in pair if not (args.final / arm / SCORED).is_file()})
    if missing:
        sys.exit("error: missing scored file(s) (run final_table.py or `rlexp.metrics score` first): " + ", ".join(missing))
    registered = selection
    if selection is not None:
        ok, why = selection_is_the_runs(args.selection, args.final)
        if not ok:
            registered = None
            print(f"warning: --selection cannot be shown to be the selection the run was frozen against ({why}): no pair is labelled confirmatory", file=sys.stderr, flush=True)
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        for pair in pairs:
            rec = paired_record(args.final, pair, args.split, args.data, args.boot, tmp, registered)
            records.append(rec)
            e = rec["metrics"]["fv_tanimoto"] if rec["metrics"] else None
            print(f"[{args.split}] {rec['a']} minus {rec['b']} ({rec['label']}): " + ("no rows in common" if e is None else
                  "fv_tanimoto undefined (fewer than 8 samples per row in a run)" if not e["n_drugs"] else
                  f"fv_tanimoto {cell(e['diff'], 4, True)} (A {cell(e['a'], 4, False)}, B {cell(e['b'], 4, False)}), wins {e['wins']} losses {e['losses']} ties {e['ties']}, "
                  f"p {p_text(e['p_value'])}, {rec['n_drugs']} drugs"), flush=True)
    res = {"split": args.split, "sign": SIGN, "boot": args.boot, "seed": 0, "selection_is_the_runs": None if selection is None else registered is not None,
           "selection": None if selection is None else {k: selection.get(k) for k in ("rl", "comparator")}, "pairs": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(markdown(res), encoding="utf-8")
    print(f"wrote {args.out}: {len(records)} pairs on {args.split}")


if __name__ == "__main__":
    main()
