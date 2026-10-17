#!/usr/bin/env python3
"""Class-information ceiling: does a drug's structured description predict its structure better than a constant?

    .venv/bin/python scripts/class_ceiling.py [--cache data/raw/chembl_pool] [--boot 10000] [--seed 0] [--json out.json]

Population: the kept pool (pool.jsonl) plus the Train drugs ChEMBL knows (train_facts.jsonl), restricted to drugs with at
least one mechanism or ATC code, one per connectivity, in ChEMBL-id order. Each drug has a binary fact set (features()):
ATC levels 3 and 4, mechanism action types, target ids and types, indication MeSH headings. Similarity between drugs is the
Jaccard index of the sets; ties go to the lowest ChEMBL id. Leave-one-out predictions of a drug's structure, each scored
with chem.tanimoto (Morgan count-vector Tanimoto, the reward's) against the true structure:
  (a) facts-kNN     the structure of the most similar other drug
  (b) facts-medoid  of the 5 most similar other drugs, the one with the highest mean Tanimoto to the other four
  (c) constant      the population structure with the highest mean Tanimoto to the others (for that one drug itself,
                    the runner-up: a drug is never answered with its own structure)
  (d) random        a random other drug, as the exact expectation (mean over all other drugs)
  (e) oracle        the most similar structure among the other drugs (upper bound)
Drug-balanced means; 95% percentile bootstrap over drugs (one set of resamples per table, so differences are paired).
Two neighbour rules are reported. "any": every other drug may be a neighbour (the specification). "split-matched": only
drugs the split rule would keep apart (bit Tanimoto < data_prep.SIM and no shared non-empty Murcko scaffold, as between
Train and Val/Test-R/Test-T), which is what a held-out drug sees. Subsets restrict the target drugs only (the neighbours
and the constant stay the whole population): all drugs, at least 3 facts, first approved 2015 or later (ChEMBL's
first_approval), and, beyond the brief, Train drugs only (the drugs most like the held-out ones).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import DataStructs

from rlexp import chem
from rlexp.data_prep import SIM

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"knn": "(a) facts-kNN k=1", "medoid": "(b) facts-medoid k=5", "constant": "(c) constant",
           "random": "(d) random other drug", "oracle": "(e) oracle"}
THRESHOLDS = (0.4, 0.3)


def num(chembl_id):
    return int(chembl_id.removeprefix("CHEMBL"))


def features(row):
    """Binary fact set of a drug: ATC levels 3 and 4 (first 4 and 5 characters of each code), mechanism action types,
    target ids and target types, and indication MeSH headings, each prefixed by its kind."""
    out = set()
    for code in row["atc"]:
        out |= {f"atc{n}:{code[:n + 1]}" for n in (3, 4) if len(code) > n}
    for m in row["mechanisms"]:
        out |= {f"{kind}:{m[key]}" for kind, key in (("action", "action_type"), ("target", "target_chembl_id"),
                                                      ("ttype", "target_type")) if m.get(key)}
    out |= {f"mesh:{i['mesh_heading']}" for i in row["indications"] if i.get("mesh_heading")}
    return frozenset(out)


def jaccard_matrix(sets):
    """Jaccard index of every pair of fact sets (0 when both are empty)."""
    vocab = {f: j for j, f in enumerate(sorted(set().union(*sets)))}
    X = np.zeros((len(sets), len(vocab)))
    for i, s in enumerate(sets):
        X[i, [vocab[f] for f in s]] = 1.0
    inter, size = X @ X.T, X.sum(1)
    union = size[:, None] + size[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def tanimoto_matrix(mols, bits=False):
    """All-pairs Tanimoto, count vectors (= chem.tanimoto) or 2048-bit vectors (= chem.tanimoto_bits), from the fingerprint
    generator of rlexp.chem (chem._FP) built once per molecule; tests/test_pool.py pins both to the public functions."""
    make = chem._FP.GetFingerprint if bits else chem._FP.GetCountFingerprint  # ponytail: private chem._FP, ~50x faster than N^2 chem.tanimoto calls; a public bulk helper in chem.py would remove it
    fps = [make(m) for m in mols]
    return np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])  # ponytail: dense N x N (about 25 MB per 1,800 drugs); chunk the rows past ~20,000 drugs


def predictions(J, T, ok, k=5):
    """(knn, medoid): for every drug the index of the drug whose structure answers it, among the drugs j with ok[i, j].
    knn is the most similar (J) one; medoid is, of the k most similar, the one with the highest mean T to the other k - 1.
    Rows must be in ChEMBL-id order: ties in J and in the medoid go to the earlier row."""
    S = np.where(ok, J, -1.0)
    top = np.argsort(-S, axis=1, kind="stable")[:, :k]
    valid = np.take_along_axis(S, top, 1) >= 0.0
    sub = T[top[:, :, None], top[:, None, :]]  # (n, k, k): T among the k neighbours
    rest = (sub * valid[:, None, :]).sum(2) - np.einsum("nkk->nk", sub)  # to the other valid ones
    mean = np.where(valid, rest / np.maximum(valid.sum(1) - 1, 1)[:, None], -np.inf)
    return top[:, 0], top[np.arange(len(top)), mean.argmax(1)]


def scores(J, T, ok):
    """{method: per-drug Tanimoto of the prediction to the true structure} for the five predictors, using only the pairs ok."""
    n = len(J)
    assert ok.sum(1).min() >= 1, "a drug with no eligible neighbour"
    knn, med = predictions(J, T, ok)
    order = np.argsort(-(T.sum(0) - np.diag(T)) / (n - 1), kind="stable")  # constants, most central first
    const = order[ok[:, order].argmax(1)]  # the most central one that is eligible for the drug
    i = np.arange(n)
    return {"knn": T[i, knn], "medoid": T[i, med], "constant": T[i, const], "random": (T * ok).sum(1) / ok.sum(1),
            "oracle": np.where(ok, T, -1.0).max(1)}


def bootstrap(V, boot=10000, seed=0):
    """(mean, lo, hi): the mean over drugs (columns) of every row of V with its 95% percentile bootstrap interval. All rows
    share the resampled drugs, so the interval of a difference of rows is the paired one."""
    V = np.asarray(V, float)
    n = V.shape[1]
    if n == 0:
        return (np.full(len(V), np.nan),) * 3
    rng, means = np.random.default_rng(seed), np.empty((boot, len(V)))
    for s in range(0, boot, 1000):
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=min(1000, boot - s))
        means[s:s + len(counts)] = counts @ V.T / n
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    return V.mean(1), lo, hi


def summarize(S, mask, boot, seed):
    """{quantity: [mean, lo, hi]} over the target drugs in mask: per method the mean Tanimoto and the shares at THRESHOLDS,
    plus the paired differences facts-kNN and facts-medoid minus the constant."""
    rows = {f"mean:{m}": S[m][mask] for m in METHODS}
    rows |= {f"share{t}:{m}": (S[m][mask] >= t).astype(float) for t in THRESHOLDS for m in METHODS}
    rows |= {f"diff:{m}": S[m][mask] - S["constant"][mask] for m in ("knn", "medoid")}
    mean, lo, hi = bootstrap(list(rows.values()), boot, seed)
    return {"n": int(mask.sum()), **{k: [float(a), float(b), float(c)] for k, a, b, c in zip(rows, mean, lo, hi)}}


def verdict(lo, hi):
    """How a paired interval (kNN minus constant) reads: its position relative to 0."""
    return "beats" if lo > 0 else "falls below" if hi < 0 else "cannot be told apart from"


def gap(r):
    """Share of the distance from the constant to the oracle that the facts-kNN covers (a summarize() result)."""
    c, o = r["mean:constant"][0], r["mean:oracle"][0]
    return (r["mean:knn"][0] - c) / (o - c) if o > c else float("nan")


def load_population(cache):
    """Rows of pool.jsonl and train_facts.jsonl with at least one mechanism or ATC code, one per connectivity (Train rows
    win), in ChEMBL-id order, each with a `source` of "pool" or "train"."""
    rows = []
    for name, source in (("train_facts.jsonl", "train"), ("pool.jsonl", "pool")):
        with open(cache / name, encoding="utf-8") as f:
            rows += [{**json.loads(line), "source": source} for line in f if line.strip()]
    seen, pop = set(), []
    for r in rows:
        if (r["mechanisms"] or r["atc"]) and r["conn"] not in seen:
            seen.add(r["conn"])
            pop.append(r)
    return sorted(pop, key=lambda r: num(r["chembl_id"])), len(rows)


def run(pop, boot=10000, seed=0):
    """All tables: {neighbour rule: {subset: summarize(...)}} plus facts about the population."""
    sets = [features(r) for r in pop]
    mols = [chem.standardize(r["smiles"]) for r in pop]
    J, T, B = jaccard_matrix(sets), tanimoto_matrix(mols), tanimoto_matrix(mols, bits=True)
    n = len(pop)
    code = {}
    scaf = np.array([code.setdefault(r["scaffold"], len(code)) if r["scaffold"] else -1 - i for i, r in enumerate(pop)])
    any_ = ~np.eye(n, dtype=bool)
    rules = {"any": any_, "split-matched": any_ & (B < SIM) & (scaf[:, None] != scaf[None, :])}
    n_facts = np.array([len(s) for s in sets])
    year = np.array([r["first_approval"] if r["first_approval"] is not None else -1 for r in pop])
    subsets = {"all drugs": np.ones(n, bool), "at least 3 facts": n_facts >= 3, "first approved 2015 or later": year >= 2015,
               "Train drugs only": np.array([r["source"] == "train" for r in pop])}
    out = {"population": {"drugs": n, "pool": sum(r["source"] == "pool" for r in pop), "train": sum(r["source"] == "train" for r in pop),
                          "median_facts": float(np.median(n_facts)), "at_least_3_facts": int((n_facts >= 3).sum()),
                          "first_approval_2015_or_later": int((year >= 2015).sum()), "unknown_year": int((year < 0).sum()),
                          "no_fact_in_common_with_any_other": int((np.where(any_, J, -1.0).max(1) == 0).sum()),
                          "eligible_neighbours_median": {k: float(np.median(v.sum(1))) for k, v in rules.items()}},
           "rules": {}}
    for rule, ok in rules.items():
        S = scores(J, T, ok)
        out["rules"][rule] = {name: summarize(S, mask, boot, seed) for name, mask in subsets.items()}
    return out


def fmt(x, pct=False):
    return f"{100 * x:5.1f}%" if pct else f"{x:.3f}"


def report(res):
    """Plain-text tables and the short reading."""
    p = res["population"]
    lines = [f"population {p['drugs']} drugs (pool {p['pool']}, Train {p['train']}); median {p['median_facts']:.0f} facts per drug; "
             f"{p['at_least_3_facts']} with at least 3 facts; {p['first_approval_2015_or_later']} first approved 2015 or later "
             f"({p['unknown_year']} without a year); {p['no_fact_in_common_with_any_other']} share no fact with any other drug"]
    for rule, subsets in res["rules"].items():
        lines.append(f"\nneighbours: {rule} (median {p['eligible_neighbours_median'][rule]:.0f} eligible per drug)")
        for name, r in subsets.items():
            lines.append(f"  targets: {name} (n={r['n']})        mean Tanimoto [95% CI]       share >= 0.4 [CI]           share >= 0.3 [CI]")
            for m, label in METHODS.items():
                cells = [f"{fmt(r[f'mean:{m}'][0])} [{fmt(r[f'mean:{m}'][1])}, {fmt(r[f'mean:{m}'][2])}]"]
                cells += [f"{fmt(r[f'share{t}:{m}'][0], True)} [{fmt(r[f'share{t}:{m}'][1], True)},{fmt(r[f'share{t}:{m}'][2], True)}]"
                          for t in THRESHOLDS]
                lines.append(f"    {label:24}{cells[0]:28}{cells[1]:28}{cells[2]}")
            lines.append("    paired, minus constant: " + "; ".join(
                f"{label} {r[f'diff:{m}'][0]:+.3f} [{r[f'diff:{m}'][1]:+.3f}, {r[f'diff:{m}'][2]:+.3f}]"
                for m, label in (("knn", "(a)"), ("medoid", "(b)"))))
    lines.append("\nreading")
    for rule, subsets in res["rules"].items():
        r = subsets["all drugs"]
        d, lo, hi = r["diff:knn"]
        lines.append(f"  {rule}: the facts-kNN {verdict(lo, hi)} the constant ({fmt(r['mean:knn'][0])} vs {fmt(r['mean:constant'][0])}, "
                     f"paired {d:+.3f} [{lo:+.3f}, {hi:+.3f}]) and covers {100 * gap(r):.0f}% of the way to the oracle "
                     f"({fmt(r['mean:oracle'][0])}); the 5-medoid {verdict(*r['diff:medoid'][1:])} it; "
                     f"{fmt(r['share0.4:knn'][0], True).strip()} of kNN answers reach Tanimoto 0.4 and "
                     f"{fmt(r['share0.3:knn'][0], True).strip()} reach 0.3.")
    wins = {rule: verdict(*subsets["all drugs"]["diff:knn"][1:]) == "beats" for rule, subsets in res["rules"].items()}
    sm = res["rules"]["split-matched"]["all drugs"]
    held = f"by {sm['diff:knn'][0]:+.3f} Tanimoto ({100 * gap(sm):.0f}% of the way to the oracle)"
    lines.append("  in short: " + (
        f"class facts beat a constant even with close analogues ruled out, but {held}." if all(wins.values()) else
        "class facts beat the constant only through close analogues, which the split rule removes from a held-out drug's neighbours."
        if wins["any"] else "class facts do not beat the constant." if not any(wins.values()) else "mixed, see the tables."))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "data/raw/chembl_pool")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, help="also write the numbers here")
    a = ap.parse_args(argv)
    try:
        pop, n_rows = load_population(a.cache)
    except OSError as e:
        sys.exit(f"error: {e} (run scripts/build_pool.py first)")
    print(f"{n_rows} rows read, {len(pop)} drugs with a mechanism or an ATC code")
    res = run(pop, a.boot, a.seed)
    print(report(res))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
