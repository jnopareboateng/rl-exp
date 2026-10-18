#!/usr/bin/env python3
"""Rank readout: how well a delivered answer picks its own drug out of the split's own references (docs/prereg-v1-addendum1.md, S1 to S3).

  .venv/bin/python scripts/rank_readout.py --final runs/colab-s1 --split val --arms a2-val,rlfull5-s50-t0.3,val8-control_constant_pool --out results/rank_val.json [--md results/rank_val.md]
  .venv/bin/python scripts/rank_readout.py --final runs/final --split test_t --a RL-full --b RL-full-shuffle --out results/rank_test_t_rl_shuffle.json --allow-test

The candidate set is the N distinct drugs of the split, each by its reference structure, read from <data>/<split>.jsonl: it is fixed by the split, not by the arm, so
arms are comparable. For every prompt row we take the FIRST valid, closed, finished answer among the first 8 samples (the endpoint's own policy, `rlexp.metrics` FV),
rank the N references by count-vector Morgan Tanimoto to it (`rlexp.chem.tanimoto`, the one definition of the similarity) and record where the row's own drug lands.
Ties share their mid-rank, so hit@1 needs a strictly nearest reference. A row with no valid answer among its first 8 samples scores the worst rank, and the share of
rows that did have one is reported beside the result as fv_rate: read the two together, because an arm can look good here by failing on the rows it would score badly.
Per drug we average over its rows, then over drugs, so drugs count equally (`rlexp.metrics`' unit).

Why this readout exists: any policy whose output does not depend on the drug gives every row the SAME ranking, so its hit@1 is exactly 1/N and its mean percentile
rank is exactly (N+1)/2N, whatever molecule it emits. A constant answer therefore cannot score above chance here, which the delivered-similarity endpoint allows it to
do. Both chance values are printed beside every result. Nothing here is confirmatory (addendum 1, section 5).

Contrasts (--a/--b, or --pairs A:B,...): the per-drug difference A minus B, summed within each component (cluster_id) and tested by the cluster-level sign-flip of
`rlexp.metrics.sign_flip_p`, the same exchangeable units and the same test as the endpoint. Wins, losses and ties count drugs. A Test split needs --allow-test.
No answer text, prompt or structure is printed or written.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts/ is not a package
from final_table import SPLIT_NAMES, cell, check_splits, plain      # noqa: E402

from rlexp import chem, metrics      # noqa: E402

SCORED = "scored.v3.jsonl"
HIT = "hit@1"
PCT = "pct_rank"


def candidates(data_rows, split):
    """(groups, mols) of the split's distinct drugs, in a fixed order. Raises InputError on a drug whose reference will not parse."""
    refs = {}
    for r in data_rows:
        if r.get("split") == split and r.get("group") and r.get("ref_smiles"):
            refs.setdefault(r["group"], r["ref_smiles"])
    if not refs:
        raise metrics.InputError(f"no {split} rows with a group and a reference structure")
    groups, mols = [], []
    for group in sorted(refs):
        mol = chem.standardize(refs[group])
        if mol is None:
            raise metrics.InputError(f"the reference of one {split} drug does not parse: the candidate set would be incomplete")
        groups.append(group)
        mols.append(mol)
    return groups, mols


def first_valid(scored_rows, split, k=metrics.K):
    """{(group, row id): answer or None} using the first valid, closed, finished answer among the first k samples of each prompt row."""
    rows = {}
    for r in scored_rows:
        if r.get("split") == split:
            rows.setdefault((r["group"], r["id"]), []).append(r)
    out = {}
    for key, samples in rows.items():
        out[key] = None
        for s in sorted(samples, key=lambda s: s["sample_idx"])[:k]:
            if s.get("valid") and s.get("closed") and metrics.finished(s):
                out[key] = s.get("answer")
                break
    return out


def rank_of(answer, group, groups, mols):
    """(mid-rank of the drug's own reference, 1 to N) for this answer, or None when there is no usable answer."""
    mol = chem.standardize(answer) if answer else None
    if mol is None:
        return None
    sims = [chem.tanimoto(mol, m) for m in mols]
    own = sims[groups.index(group)]
    better = sum(1 for s in sims if s > own + metrics.TIE)
    tied = sum(1 for s in sims if abs(s - own) <= metrics.TIE) - 1
    return better + 1 + tied / 2


def per_drug(scored_rows, split, groups, mols):
    """{group: {hit@1, pct_rank, fv_rate, rows}}: the row mean of each, over the rows of that drug. A row with no usable answer scores the worst rank (pct 1.0)."""
    n = len(groups)
    out = {}
    for (group, _id), answer in first_valid(scored_rows, split).items():
        if group not in out:
            out[group] = {HIT: [], PCT: [], "fv_rate": []}
        rank = rank_of(answer, group, groups, mols) if group in groups else None
        out[group][HIT].append(0.0 if rank is None else float(rank <= 1))
        out[group][PCT].append(1.0 if rank is None else rank / n)
        out[group]["fv_rate"].append(float(rank is not None))
    return {g: {**{k: float(np.mean(v)) for k, v in d.items()}, "rows": len(d[PCT])} for g, d in out.items()}


def chance(n):
    return {HIT: 1 / n, PCT: (n + 1) / (2 * n)}


def arm_record(final, arm, split, groups, mols):
    drugs = per_drug(metrics.read_jsonl(Path(final) / arm / SCORED), split, groups, mols)
    if not drugs:
        return {"arm": arm, "n_drugs": 0, HIT: None, PCT: None, "fv_rate": None, "n_rows": 0}
    return {"arm": arm, "n_drugs": len(drugs), "n_rows": sum(d["rows"] for d in drugs.values()),
            **{k: float(np.mean([d[k] for d in drugs.values()])) for k in (HIT, PCT, "fv_rate")}}


def clusters(data_rows, split):
    """{group: cluster_id} for the split; a drug with no cluster_id is its own unit (`rlexp.metrics`' rule)."""
    out = {}
    for r in data_rows:
        if r.get("split") == split and r.get("group"):
            out.setdefault(r["group"], r.get("cluster_id", r["group"]))
    return out


def contrast(final, pair, split, groups, mols, cluster_of, flips=metrics.FLIPS):
    """A minus B on the drugs the two arms share: the per-drug mean difference of each measure, and for pct_rank and hit@1 the cluster sign-flip p-value."""
    a, b = pair
    da = per_drug(metrics.read_jsonl(Path(final) / a / SCORED), split, groups, mols)
    db = per_drug(metrics.read_jsonl(Path(final) / b / SCORED), split, groups, mols)
    common = sorted(set(da) & set(db))
    rec = {"a": a, "b": b, "split": split, "n_drugs": len(common), "measures": {}}
    if not common:
        return rec
    rng = np.random.default_rng(0)
    for key in (PCT, HIT):
        diffs = {g: da[g][key] - db[g][key] for g in common}
        totals = {}
        for g, d in diffs.items():
            totals[cluster_of.get(g, g)] = totals.get(cluster_of.get(g, g), 0.0) + d
        p, method, non_tied = metrics.sign_flip_p(np.array(list(totals.values()), dtype=float), rng, flips)
        wins = sum(1 for d in diffs.values() if d < -metrics.TIE)          # lower pct_rank is better, so a win is a negative difference
        losses = sum(1 for d in diffs.values() if d > metrics.TIE)
        if key == HIT:                                                     # higher hit@1 is better: a win is a positive difference
            wins, losses = losses, wins
        rec["measures"][key] = {"a": float(np.mean([da[g][key] for g in common])), "b": float(np.mean([db[g][key] for g in common])),
                                "diff": float(np.mean(list(diffs.values()))), "wins": wins, "losses": losses, "ties": len(common) - wins - losses,
                                "n_clusters": len(totals), "non_tied": non_tied, "p_value": p, "p_method": method}
    return rec


def parse_pairs(text):
    pairs = []
    for item in filter(None, text.split(",")):
        a, sep, b = item.partition(":")
        if not (sep and plain(a) and plain(b)) or a == b:
            sys.exit(f"error: pair {item!r} is not A:B with two different plain arm ids")
        pairs.append((a, b))
    return pairs


def markdown(res):
    name, n = SPLIT_NAMES[res["split"]], res["n_candidates"]
    out = [f"# Rank readout on {name}", "",
           f"The answer ranks the {n} {name} drugs by similarity to itself; the table shows where the row's own drug lands. Lower percentile rank is better. "
           f"Any policy that does not depend on the drug scores exactly {res['chance'][HIT]:.3f} on top-1 and {res['chance'][PCT]:.3f} on percentile rank, "
           "whatever it answers, so a constant answer cannot beat chance here. Rows with no valid answer among their first eight samples take the worst rank and "
           "lower the valid-answer share beside it. Nothing in this table is confirmatory.", ""]
    if res["arms"]:
        out += ["| Arm | Top-1 | Mean percentile rank | Valid answers | Drugs |", "|---|---:|---:|---:|---:|"]
        for r in res["arms"]:
            out.append(f"| {r['arm']} | {cell(r[HIT], 3, False)} | {cell(r[PCT], 3, False)} | {cell(r['fv_rate'], 3, False)} | {r['n_drugs']} |")
        out += [f"| chance (any drug-independent policy) | {res['chance'][HIT]:.3f} | {res['chance'][PCT]:.3f} | | {n} |", ""]
    for c in res["contrasts"]:
        out += [f"### {c['a']} minus {c['b']} ({name})", ""]
        if not c["measures"]:
            out += ["The two arms share no drug on this split: nothing to compare.", ""]
            continue
        out += [f"{c['n_drugs']} common drugs. Wins count drugs for which the first arm is better.", "",
                "| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |", "|---|---:|---:|---:|---:|---:|---:|"]
        for key, text in ((PCT, "Mean percentile rank"), (HIT, "Top-1")):
            m = c["measures"][key]
            out.append(f"| {text} | {cell(m['a'], 3, False)} | {cell(m['b'], 3, False)} | {cell(m['diff'], 3, True)} | "
                       f"{m['wins']} / {m['losses']} / {m['ties']} | {m['p_value']:.4f} | {m['n_clusters']} |")
        out += [""]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", type=Path, default=Path("runs/final"))
    ap.add_argument("--split", required=True)
    ap.add_argument("--arms", default="", help="comma-separated arm ids (run directories under --final) for the table")
    ap.add_argument("--a", help="first arm of a contrast (A minus B)")
    ap.add_argument("--b", help="second arm of a contrast")
    ap.add_argument("--pairs", default="", help="A:B,A:C: further contrasts")
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--allow-test", action="store_true", help="accept a Test split (locked evaluation, after the prereg-v1 tag)")
    args = ap.parse_args(argv)
    check_splits([args.split], args.allow_test)
    if (args.a is None) != (args.b is None):
        sys.exit("error: --a and --b come together")
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if not plain(a)]
    if bad:
        sys.exit(f"error: not plain arm ids: {bad}")
    pairs = parse_pairs(args.pairs) + ([(args.a, args.b)] if args.a else [])
    pairs = list(dict.fromkeys(pairs))
    if not arms and not pairs:
        sys.exit("error: nothing to do: give --arms and/or --a/--b or --pairs")
    if not args.final.is_dir():
        sys.exit(f"error: {args.final} is not a directory")
    missing = sorted({str(args.final / arm / SCORED) for arm in arms + [x for p in pairs for x in p] if not (args.final / arm / SCORED).is_file()})
    if missing:
        sys.exit("error: missing scored file(s) (run `rlexp.metrics score` first): " + ", ".join(missing))
    data_rows = metrics.load_data(args.data, splits=(args.split,))
    groups, mols = candidates(data_rows, args.split)
    cluster_of = clusters(data_rows, args.split)
    n = len(groups)
    print(f"[{args.split}] candidate set {n} drugs; chance top-1 {1/n:.4f}, chance mean percentile rank {(n+1)/(2*n):.4f}", flush=True)
    records = []
    for arm in arms:
        r = arm_record(args.final, arm, args.split, groups, mols)
        records.append(r)
        print(f"  {arm}: top-1 {cell(r[HIT], 4, False)}, percentile rank {cell(r[PCT], 4, False)}, valid answers {cell(r['fv_rate'], 4, False)}, {r['n_drugs']} drugs", flush=True)
    contrasts = []
    for pair in pairs:
        c = contrast(args.final, pair, args.split, groups, mols, cluster_of)
        contrasts.append(c)
        m = c["measures"].get(PCT)
        print(f"  {c['a']} minus {c['b']}: " + ("no drugs in common" if m is None else
              f"percentile rank {cell(m['diff'], 4, True)} (A {cell(m['a'], 4, False)}, B {cell(m['b'], 4, False)}), "
              f"wins {m['wins']} losses {m['losses']} ties {m['ties']}, p {m['p_value']:.4f} ({m['p_method']})"), flush=True)
    res = {"split": args.split, "n_candidates": n, "chance": chance(n), "confirmatory": False,
           "sign": "A minus B; for the percentile rank a lower value is better, so a win is a negative difference", "arms": records, "contrasts": contrasts}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(markdown(res), encoding="utf-8")
    print(f"wrote {args.out}: {len(records)} arms and {len(contrasts)} contrasts on {args.split}")


if __name__ == "__main__":
    main()
