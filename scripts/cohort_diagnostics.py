#!/usr/bin/env python3
"""EXPLORATORY, read-only diagnostics of why the profile dependence seen on Val and Test-R is not detected on Test-T (advisor round 2, checks 1 to 4). Aggregates only.

  .venv/bin/python scripts/cohort_diagnostics.py --final runs/final --out results/exploratory_cohort_diagnostics.json [--md results/exploratory_cohort_diagnostics.md] --allow-test

1. heterogeneity: the difference between the Test-T and the Test-R effect (real minus shuffled profile, delivered similarity and percentile rank, for RL-full and A2), bootstrapped
   over components, beside the paired per-drug SD and the minimum detectable difference (2.8 SD / sqrt(units)) of each split.
2. headroom: per drug, the best similarity to any Train reference minus the C3 molecule's similarity to the drug (what a policy that could pick the best Train structure would gain
   over the constant); per-split quartiles and the share at or below zero.
3. crowding: mean pairwise similarity among a split's own references, the share of drugs whose nearest other reference exceeds 0.25, and the rank readout of Test-T recomputed on
   random 30-drug candidate subsets (Test-R's N), so the two splits are comparable at equal N.
4. composition: per split quartiles of the prompt tokens, rows per task type, heavy atoms, PubChem record year, PubMed count and nearest-Train similarity, with the Test-T minus
   Test-R difference of medians and a drug-level permutation p-value.
Nothing here chooses an arm, a setting or a comparator, and every number is exploratory (no multiplicity adjustment). No prompt, answer or structure is printed or written.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts/ is not a package
import rank_readout as rr      # noqa: E402
from final_table import check_splits      # noqa: E402

from rlexp import chem, metrics      # noqa: E402

PAIRS = {"RL-full": ("RL-full", "RL-full-shuffle"), "A2": ("A2", "A2-shuffle")}


def quartiles(values):
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], float)
    return {"n": int(len(v)), "q1": float(np.percentile(v, 25)), "median": float(np.median(v)), "q3": float(np.percentile(v, 75)), "mean": float(v.mean())} if len(v) else {"n": 0}


def fv_per_drug(scored_rows, split):
    """{group: mean over its rows of the endpoint's row value}: the Tanimoto of the first usable answer among the first K samples, else 0."""
    rows = {}
    for r in scored_rows:
        if r.get("split") == split:
            rows.setdefault((r["group"], r["id"]), []).append(r)
    per = {}
    for (group, _), samples in rows.items():
        value = 0.0
        for s in sorted(samples, key=lambda s: s["sample_idx"])[:metrics.K]:
            if metrics._ok(s):
                value = float(s["tanimoto"])
                break
        per.setdefault(group, []).append(value)
    return {g: float(np.mean(v)) for g, v in per.items()}


def effect_by_cluster(diffs, cluster_of):
    """[(sum of the per-drug differences, number of drugs)] per component."""
    out = {}
    for g, d in diffs.items():
        s, n = out.get(cluster_of.get(g, g), (0.0, 0))
        out[cluster_of.get(g, g)] = (s + d, n + 1)
    return np.array(list(out.values()), float)


def heterogeneity(diffs_t, diffs_r, cluster_t, cluster_r, boot=10_000, seed=0):
    """Test-T effect minus Test-R effect (drug-weighted means) with a component bootstrap interval, plus the paired SD and the minimum detectable difference of each split."""
    a, b = effect_by_cluster(diffs_t, cluster_t), effect_by_cluster(diffs_r, cluster_r)
    rng = np.random.default_rng(seed)

    def mean_of(units, idx):
        picked = units[idx]
        return picked[:, 0].sum() / picked[:, 1].sum()

    draws = np.array([mean_of(a, rng.integers(0, len(a), len(a))) - mean_of(b, rng.integers(0, len(b), len(b))) for _ in range(boot)])
    point = a[:, 0].sum() / a[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum()
    out = {"effect_t": float(a[:, 0].sum() / a[:, 1].sum()), "effect_r": float(b[:, 0].sum() / b[:, 1].sum()), "difference": float(point),
           "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], "share_of_draws_at_or_below_zero": float((draws <= 0).mean())}
    for name, diffs, units in (("t", diffs_t, a), ("r", diffs_r, b)):
        sd = float(np.std(list(diffs.values()), ddof=1))
        out[f"sd_{name}"], out[f"units_{name}"] = sd, int(len(units))
        out[f"mdd_{name}"] = float(2.8 * sd / np.sqrt(len(units)))
    return out


def headroom(drug_mols, train_mols, c3_mol):
    """Per drug: the best similarity to any Train reference minus C3's similarity to the drug."""
    gaps = [max(chem.tanimoto(m, t) for t in train_mols) - chem.tanimoto(c3_mol, m) for m in drug_mols]
    q = quartiles(gaps)
    q["share_at_or_below_zero"] = float(np.mean(np.asarray(gaps) <= 0))
    return q


def crowding(mols):
    """Mean pairwise similarity among a split's references and the share of drugs whose nearest other reference exceeds 0.25."""
    n = len(mols)
    sims = np.array([[chem.tanimoto(mols[i], mols[j]) if i != j else np.nan for j in range(n)] for i in range(n)])
    return {"n": n, "mean_pairwise": float(np.nanmean(sims)), "share_nearest_above_0.25": float(np.mean(np.nanmax(sims, axis=1) > 0.25)),
            "median_nearest": float(np.median(np.nanmax(sims, axis=1)))}


def subset_readout(rows, split, groups, mols, size, draws=300, seed=0):
    """Mean percentile rank of each arm's rows when the candidate set is the own drug plus size-1 random other drugs (the Test-R N), over `draws` draws.
    A row with no usable answer scores 1.0, as in rank_readout. Returns the per-draw means."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    per_row = {}
    for (group, rid), answer in rr.first_valid(rows, split).items():
        mol = chem.standardize(answer) if answer else None
        per_row[(group, rid)] = None if mol is None else np.array([chem.tanimoto(mol, m) for m in mols])
    by_drug = {}
    for (group, _), sims in per_row.items():
        by_drug.setdefault(group, []).append(sims)
    means = np.zeros(draws)
    for d in range(draws):
        vals = []
        for group, rowsims in by_drug.items():
            own = groups.index(group)
            others = np.delete(np.arange(n), own)
            chosen = rng.choice(others, size - 1, replace=False)
            cand = np.concatenate([[own], chosen])
            row_values = []
            for sims in rowsims:
                if sims is None:
                    row_values.append(1.0)
                    continue
                s = sims[cand]
                better = np.sum(s > s[0] + metrics.TIE)
                tied = np.sum(np.abs(s - s[0]) <= metrics.TIE) - 1
                row_values.append((better + 1 + tied / 2) / size)
            vals.append(np.mean(row_values))
        means[d] = np.mean(vals)
    return means


def perm_p(a, b, draws=10_000, seed=0):
    """Two-sided permutation p-value of the difference of medians of two samples of drugs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b])
    obs = abs(np.median(a) - np.median(b))
    hits = 0
    for _ in range(draws):
        rng.shuffle(pooled)
        hits += abs(np.median(pooled[:len(a)]) - np.median(pooled[len(a):])) >= obs - 1e-12
    return float((hits + 1) / (draws + 1))


def composition(per_split):
    """per_split: {split: {covariate: [values per drug or row]}}. Quartiles per split and the Test-T minus Test-R difference of medians with a permutation p."""
    out = {}
    for cov in sorted({c for d in per_split.values() for c in d}):
        out[cov] = {s: quartiles(d[cov]) for s, d in per_split.items() if cov in d}
        if "test_t" in per_split and "test_r" in per_split and cov in per_split["test_t"] and cov in per_split["test_r"]:
            t, r = per_split["test_t"][cov], per_split["test_r"][cov]
            out[cov]["test_t_minus_test_r_median"] = float(np.median(t) - np.median(r))
            out[cov]["permutation_p"] = perm_p(t, r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", type=Path, default=Path("runs/final"))
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--boot", type=int, default=10_000)
    args = ap.parse_args(argv)
    check_splits(["test_t", "test_r"], args.allow_test)
    refs = [json.loads(line) for line in (args.data / "refs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_split = {s: [r for r in refs if r["split"] == s] for s in ("train", "val", "test_r", "test_t")}
    mols = {s: [chem.standardize(r["ref_smiles"]) for r in rs] for s, rs in by_split.items()}
    c3_answer = next(r["answer"] for r in metrics.read_jsonl(args.final / "C3" / rr.SCORED) if r.get("answer"))
    c3 = chem.standardize(c3_answer)
    result = {"exploratory": True, "heterogeneity": {}, "headroom": {}, "crowding": {}, "subset_readout": {}, "composition": {}}

    # 1. heterogeneity
    clusters = {s: {r["group"]: r["cluster_id"] if r.get("cluster_id") is not None else r["group"] for r in by_split[s]} for s in ("test_t", "test_r")}
    data = {s: metrics.load_data(args.data, splits=(s,)) for s in ("val", "test_t", "test_r")}
    cand = {s: rr.candidates(data[s], s) for s in ("test_t", "test_r")}
    scored = {}
    for arm in {x for p in PAIRS.values() for x in p} | {"C1"}:
        scored[arm] = metrics.read_jsonl(args.final / arm / rr.SCORED)
    for name, (real, shuf) in PAIRS.items():
        for measure in ("delivered_similarity", "percentile_rank"):
            diffs = {}
            for s in ("test_t", "test_r"):
                if measure == "delivered_similarity":
                    a, b = fv_per_drug(scored[real], s), fv_per_drug(scored[shuf], s)
                else:
                    groups, mol = cand[s]
                    a = {g: v[rr.PCT] for g, v in rr.per_drug(scored[real], s, groups, mol).items()}
                    b = {g: v[rr.PCT] for g, v in rr.per_drug(scored[shuf], s, groups, mol).items()}
                diffs[s] = {g: a[g] - b[g] for g in a if g in b}
            result["heterogeneity"][f"{name}:{measure}"] = heterogeneity(diffs["test_t"], diffs["test_r"], clusters["test_t"], clusters["test_r"], args.boot)
    # 2 and 3. headroom and crowding
    for s in ("val", "test_r", "test_t"):
        result["headroom"][s] = headroom(mols[s], mols["train"], c3)
        result["crowding"][s] = crowding(mols[s])
    groups_t, mols_t = cand["test_t"]
    n_r = len(cand["test_r"][0])
    for arm in ("RL-full", "RL-full-shuffle", "A2", "A2-shuffle", "C1"):
        if arm not in scored:
            scored[arm] = metrics.read_jsonl(args.final / arm / rr.SCORED)
    sub = {arm: subset_readout(scored[arm], "test_t", groups_t, mols_t, n_r) for arm in ("RL-full", "RL-full-shuffle", "A2", "A2-shuffle", "C1")}
    for arm, v in sub.items():
        result["subset_readout"][arm] = {"mean_percentile_rank": float(v.mean()), "q2.5": float(np.percentile(v, 2.5)), "q97.5": float(np.percentile(v, 97.5)), "candidate_set_size": n_r}
    for name, (real, shuf) in PAIRS.items():
        d = sub[real] - sub[shuf]
        result["subset_readout"][f"{name}_real_minus_shuffled"] = {"mean": float(d.mean()), "q2.5": float(np.percentile(d, 2.5)), "q97.5": float(np.percentile(d, 97.5))}
    # 4. composition
    cov = {}
    for s in ("val", "test_r", "test_t"):
        rows, rs = data[s], by_split[s]
        near = [max(chem.tanimoto(m, t) for t in mols["train"]) for m in mols[s]]
        task_counts = {}
        for r in rows:
            task_counts[r.get("task_type")] = task_counts.get(r.get("task_type"), 0) + 1
        cov[s] = {"prompt_tokens_per_row": [r["n_prompt_tokens"] for r in rows], "heavy_atoms": [r["n_heavy"] for r in rs],
                  "pubchem_create_year": [r["pubchem_create_year"] for r in rs if r.get("pubchem_create_year") is not None],
                  "pubmed_count": [r["pubmed_count"] for r in rs if r.get("pubmed_count") is not None], "nearest_train_similarity": near}
        result["composition"][f"rows_per_task_type_{s}"] = {str(k): v for k, v in sorted(task_counts.items(), key=lambda kv: str(kv[0]))}
    result["composition"]["covariates"] = composition(cov)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
