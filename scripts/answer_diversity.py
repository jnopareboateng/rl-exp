#!/usr/bin/env python3
"""How concentrated are a run's usable answers, and how close are they to the constant control's answer? Aggregates only.

  .venv/bin/python scripts/answer_diversity.py --runs runs/colab-s1 --split val --constant val8-control_constant_pool \
      --names a2-val,rlfull-s40-t0.6 [--json results/answer_diversity.json]

A usable answer is valid, closed and finished (rlexp.metrics._ok). Per run: how many distinct structures (connectivity key) the usable answers use, the share of the most
frequent one, the share of drugs whose own most frequent answer is the run's most frequent one, the mean count-vector Tanimoto of the answers to the constant control's
answer, and the mean Tanimoto between answers given to DIFFERENT drugs (1.0 means one answer for everything). No answer text is printed or written.
"""
import argparse
import itertools
import json
import statistics
from collections import Counter
from pathlib import Path

from rlexp import chem, metrics


def usable_answers(path, split):
    """[(drug, answer smiles)] of the usable samples of one split of a scored file."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") == split and metrics._ok(row) and row.get("answer"):
                out.append((row["group"], row["answer"]))
    return out


def constant_answer(path, split):
    answers = {smiles for _, smiles in usable_answers(path, split)}
    if len(answers) != 1:
        raise ValueError(f"{path}: expected one constant answer on {split}, found {len(answers)}")
    return answers.pop()


def diversity(answers, constant):
    """The aggregates of [(drug, smiles)] against a constant answer (smiles); None entries when there is nothing to average."""
    mols = {}
    for _, smiles in answers:
        if smiles not in mols:
            mols[smiles] = chem.standardize(smiles)
    keys = {s: (chem.conn(m) if m is not None else None) for s, m in mols.items()}
    usable = [(d, s) for d, s in answers if keys.get(s)]
    if not usable:
        return {"n_usable": 0}
    by_key = Counter(keys[s] for _, s in usable)
    top_key, top_n = by_key.most_common(1)[0]
    per_drug = {}
    for d, s in usable:
        per_drug.setdefault(d, Counter())[keys[s]] += 1
    modal = {d: c.most_common(1)[0][0] for d, c in per_drug.items()}
    ref = chem.standardize(constant)
    to_const = [chem.tanimoto(mols[s], ref) for _, s in usable]
    # one representative per (drug, structure) keeps the cross-drug pair count small and weights every drug's distinct answers once
    reps = {}
    for d, s in usable:
        reps.setdefault((d, keys[s]), s)
    drugs = sorted({d for d, _ in reps})
    cross = []
    by_drug = {d: [s for (dd, _), s in reps.items() if dd == d] for d in drugs}
    for a, b in itertools.combinations(drugs, 2):
        cross.append(statistics.fmean(chem.tanimoto(mols[x], mols[y]) for x in by_drug[a] for y in by_drug[b]))
    return {"n_usable": len(usable), "n_drugs": len(per_drug), "distinct_structures": len(by_key), "top_structure_share": top_n / len(usable),
            "drugs_with_top_structure_as_mode": sum(1 for d in modal if modal[d] == top_key) / len(modal),
            "mean_tanimoto_to_constant": statistics.fmean(to_const),
            "mean_tanimoto_between_drugs": statistics.fmean(cross) if cross else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs/colab-s1"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--constant", required=True, help="run directory (under --runs) of the constant control whose answer is the reference point")
    ap.add_argument("--names", required=True, help="comma-separated run directories under --runs")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    constant = constant_answer(args.runs / args.constant / "scored.v3.jsonl", args.split)
    result = {"split": args.split, "constant_run": args.constant, "runs": {}}
    for name in args.names.split(","):
        result["runs"][name] = diversity(usable_answers(args.runs / name / "scored.v3.jsonl", args.split), constant)
    cols = ("n_usable", "distinct_structures", "top_structure_share", "drugs_with_top_structure_as_mode", "mean_tanimoto_to_constant", "mean_tanimoto_between_drugs")
    print("run".ljust(26) + "".join(c[:14].rjust(16) for c in cols))
    for name, agg in result["runs"].items():
        print(name.ljust(26) + "".join(("n/a" if agg.get(c) is None else f"{agg[c]:.3f}" if isinstance(agg[c], float) else str(agg[c])).rjust(16) for c in cols))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
