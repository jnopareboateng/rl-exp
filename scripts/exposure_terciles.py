#!/usr/bin/env python3
"""EXPLORATORY: does a paired contrast depend on how public a drug already was? The per-drug fv_tanimoto difference A minus B of two runs of ONE split, summarised in
terciles of two exposure covariates (docs/plan.md section 3, docs/paper/README.md H2). No multiplicity adjustment; every number is labelled exploratory.

  .venv/bin/python scripts/exposure_terciles.py --final runs/final --split test_t --a RL-full --b C3 --out results/exposure_test_t.json --allow-test \
      [--md results/exposure_test_t.md] [--refs data/processed/refs.jsonl] [--exposure data/curated/exposure.json]

Covariates per drug, from the refs rows of the split (refs.jsonl; data/curated/exposure.json fills a value the refs lack): the PubMed article count, and the record age =
approval year minus the PubChem record's creation year (the years the structure was public before first approval). Terciles are by rank of (covariate, drug id), lowest first,
sizes rank * 3 // n (14, 14, 13 for 41 drugs). A drug without a covariate is counted in `missing`, never dropped silently; fewer than three drugs with one is an error.
Numbers come from `rlexp.metrics.report` run on the two runs restricted to the drugs of a tercile (boot 0, seed 0: no resampling): the mean difference is the report's
drug-level mean, wins/losses/ties count drugs, and p is the report's sign-flip p-value on the clusters of those drugs (each Test-T drug is its own cluster; drugs of one component
that land in different terciles are separate units in each). The contrast of all common drugs is `overall` and equals paired_table.py's fv_tanimoto block. The output holds counts
and covariate ranges only: no drug name, id, sample or molecule.
A Test split needs --allow-test (like make_figures.py).
"""
import argparse
import json
import sys
from pathlib import Path

from rlexp import metrics

sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts/ is not a package
from final_table import SPLIT_NAMES, cell, check_splits, clean, plain      # noqa: E402

SCORED = "scored.v3.jsonl"
TERCILES = ("low", "middle", "high")
COVARIATES = (("pubmed_count", "PubMed articles"), ("record_age", "Years the structure was public before approval"))
SIGN = "A minus B: a positive difference means run A is higher"
STAT = ("n_drugs", "n_clusters", "wins", "losses", "ties", "p_value", "p_method", "p_unit")
DEFINITIONS = {"pubmed_count": "PubMed article count of the drug (refs.jsonl; data/curated/exposure.json fills a missing value)",
               "record_age": "years the structure was public before first approval: approval year minus the creation year of its PubChem record",
               "terciles": "rank * 3 // n over the drugs that have the covariate, ordered by (covariate, drug id), lowest first"}


def number(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def terciles(values):
    """{drug: covariate} -> three lists of drugs, lowest third first, by rank of (covariate, drug id): the id breaks ties, so the cut is deterministic."""
    order = sorted(values, key=lambda g: (values[g], g))
    bins = [[], [], []]
    for rank, g in enumerate(order):
        bins[rank * 3 // len(order)].append(g)
    return bins


def covariates(refs, exposure, split, drugs):
    """({drug: PubMed count}, {drug: record age}, missing counts) of the drugs: the refs rows of `split` first, exposure.json for a value they lack."""
    mine = {r["group"]: r for r in refs if r.get("split") == split}

    def pick(g, key):
        v = number(mine.get(g, {}).get(key))
        return v if v is not None else number(exposure.get(g, {}).get(key))

    got = {g: {k: pick(g, k) for k in ("pubmed_count", "pubchem_create_year", "approval_year")} for g in drugs}
    pubmed = {g: v["pubmed_count"] for g, v in got.items() if v["pubmed_count"] is not None}
    age = {g: v["approval_year"] - v["pubchem_create_year"] for g, v in got.items() if v["approval_year"] is not None and v["pubchem_create_year"] is not None}
    missing = {"pubmed_count": len(drugs) - len(pubmed), "pubchem_create_year": sum(v["pubchem_create_year"] is None for v in got.values()),
               "approval_year": sum(v["approval_year"] is None for v in got.values()), "record_age": len(drugs) - len(age)}
    return pubmed, age, missing


def contrast(ra, rb, split):
    """The report's paired fv_tanimoto block of two lists of scored rows (None when they share no rows)."""
    try:
        paired = metrics.report(ra, rb, [split], boot=0, seed=0)["splits"][split]["paired"]
    except metrics.InputError as e:
        sys.exit(f"error: {clean(str(e))}")
    return (paired["metrics"] or {}).get("fv_tanimoto")


def stats(e):
    return {"mean_a": e["a"], "mean_b": e["b"], "mean_diff": e["diff"], **{k: e[k] for k in STAT}}


def tercile_record(name, drugs, values, ra, rb, split):
    members = set(drugs)
    e = contrast([r for r in ra if r["group"] in members], [r for r in rb if r["group"] in members], split)
    return {"tercile": name, "range": [min(values[g] for g in drugs), max(values[g] for g in drugs)], **stats(e), "label": "exploratory"}


def analyse(ra, rb, refs, exposure, split):
    overall = contrast(ra, rb, split)                                    # first: the metrics module checks the rows (a clean error, not a KeyError)
    ids = {r["id"] for r in ra} & {r["id"] for r in rb}
    drugs = sorted({r["group"] for r in ra if r["id"] in ids})
    if not drugs or overall is None or overall["n_drugs"] != len(drugs):
        sys.exit(f"error: fv_tanimoto is defined for {0 if overall is None else overall['n_drugs']} of {len(drugs)} common drugs "
                 "(fewer than 8 samples per prompt row in a run): the contrast cannot be split")
    pubmed, age, missing = covariates(refs, exposure, split, drugs)
    out = {"split": split, "metric": "fv_tanimoto", "sign": SIGN, "label": "exploratory", "definitions": DEFINITIONS, "n_drugs": len(drugs), "overall": stats(overall), "missing": missing, "covariates": {}}
    for key, _ in COVARIATES:
        values = pubmed if key == "pubmed_count" else age
        if len(values) < 3:
            sys.exit(f"error: {len(values)} of {len(drugs)} drugs have {key}: terciles need three")
        out["covariates"][key] = {"label": "exploratory", "n_drugs": len(values), "missing": len(drugs) - len(values),
                                  "terciles": [tercile_record(n, d, values, ra, rb, split) for n, d in zip(TERCILES, terciles(values))]}
    return out


def p_text(p):
    return "n/a" if p is None else f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def markdown(res):
    name, a, b = SPLIT_NAMES[res["split"]], res["a"], res["b"]
    sizes = [t["n_drugs"] for c in res["covariates"].values() for t in c["terciles"]]
    holds = f"{min(sizes)} drugs" if min(sizes) == max(sizes) else f"{min(sizes)} to {max(sizes)} drugs"
    out = [f"# Exposure terciles on {name} (exploratory)", "",
           f"Delivered similarity of {a} minus {b} on the {res['n_drugs']} common {name} drugs, by how public the structure already was. A positive difference means {a} is "
           f"higher. Wins, losses and ties count drugs; p is the two-sided sign-flip test within the tercile. Exploratory: no adjustment for multiplicity, and a tercile holds "
           f"only {holds}, so a difference between terciles is not a test of one.", ""]
    for key, text in COVARIATES:
        c = res["covariates"][key]
        out += [f"## {text} (lowest to highest third)", "", f"| Tercile | Range | Drugs | {a} | {b} | Difference | Wins / losses / ties | p (sign-flip) |",
                "|---|---|---:|---:|---:|---:|---:|---:|"]
        out += [f"| {t['tercile']} | {cell(t['range'][0], 0, False)} to {cell(t['range'][1], 0, False)} | {t['n_drugs']} | {cell(t['mean_a'], 3, False)} | "
                f"{cell(t['mean_b'], 3, False)} | {cell(t['mean_diff'], 3, True)} | {t['wins']} / {t['losses']} / {t['ties']} | {p_text(t['p_value'])} |" for t in c["terciles"]]
        out += ["", f"{c['n_drugs']} drugs have this covariate; {c['missing']} do not and are left out of these terciles.", ""]
    o = res["overall"]
    return "\n".join(out + [f"All {o['n_drugs']} common drugs: {cell(o['mean_a'], 3, False)} against {cell(o['mean_b'], 3, False)}, difference {cell(o['mean_diff'], 3, True)}, "
                            f"wins / losses / ties {o['wins']} / {o['losses']} / {o['ties']}, p {p_text(o['p_value'])}.", ""])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", type=Path, default=Path("runs/final"))
    ap.add_argument("--split", required=True)
    ap.add_argument("--a", required=True, help="arm id of A (a run directory under --final)")
    ap.add_argument("--b", required=True, help="arm id of B")
    ap.add_argument("--refs", type=Path, default=Path("data/processed/refs.jsonl"))
    ap.add_argument("--exposure", type=Path, default=Path("data/curated/exposure.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--allow-test", action="store_true", help="accept a Test split (locked evaluation, after the prereg-v1 tag)")
    args = ap.parse_args(argv)
    check_splits([args.split], args.allow_test)
    if args.a == args.b or not (plain(args.a) and plain(args.b)):
        sys.exit("error: --a and --b must be two different plain arm ids")
    scored = [args.final / arm / SCORED for arm in (args.a, args.b)]
    if missing := [str(f) for f in (*scored, args.refs, args.exposure) if not f.is_file()]:
        sys.exit("error: missing input file(s): " + ", ".join(missing))
    try:
        ra, rb = [[r for r in metrics.read_jsonl(f, drop=("text",)) if r.get("split") == args.split] for f in scored]
        refs, exposure = metrics.read_jsonl(args.refs), json.loads(args.exposure.read_text(encoding="utf-8"))
    except (metrics.InputError, OSError, ValueError) as e:
        sys.exit(f"error: {clean(str(e))}")
    if not isinstance(exposure, dict) or not ra or not rb:
        sys.exit(f"error: {args.exposure} is not a drug-to-covariates object" if not isinstance(exposure, dict) else f"error: {args.a if not ra else args.b} has no rows on {args.split}")
    res = {"a": args.a, "b": args.b, **analyse(ra, rb, refs, exposure, args.split)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    text = markdown(res)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {args.out}: {res['n_drugs']} drugs on {args.split}, missing {res['missing']}")


if __name__ == "__main__":
    main()
