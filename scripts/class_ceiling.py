#!/usr/bin/env python3
"""Do a drug's structured facts predict its structure better than a constant? A one-nearest-neighbour and a 5-medoid check.

    .venv/bin/python scripts/class_ceiling.py [--cache data/raw/chembl_pool] [--boot 10000] [--seed 0] [--json results/class_ceiling.json]

Scope. Only two retrieval rules over coarsened facts are tried. The result does not bound learned predictors, generated
structures, richer facts or free-text profiles, and an interval that includes 0 is not evidence of equivalence. Intervals are
descriptive for this fixed population and these fixed rules (nothing is refit per resample).

Population: the kept pool (pool.jsonl) plus the Train drugs ChEMBL knows (train_facts.jsonl), restricted to drugs with at
least one mechanism or ATC code, one per connectivity, in ChEMBL-id order. Each drug has a binary fact set (features()): ATC
levels 3 and 4, mechanism action types, target ids and types, indication MeSH headings. Similarity between drugs is the
Jaccard index of the sets; ties go to the lowest ChEMBL id. Leave-one-out predictions of a drug's structure, each scored with
chem.tanimoto (Morgan count-vector Tanimoto, the reward's) against the true structure, from the drugs that are eligible for it:
  (a) facts-kNN     the structure of the most similar eligible drug
  (b) facts-medoid  of the 5 most similar eligible drugs, the one with the highest mean Tanimoto to the other four
  (c) constant      the population structure with the highest mean Tanimoto to the other population drugs (as briefed)
  (c') constant, Train-fixed: candidates ranked by mean Tanimoto to the Train references only, so the ranking does not depend
                    on which drugs are targets; a Train target never helps rank its own constant
                    (for (c) and (c') a drug takes the highest-ranked eligible candidate, never its own structure)
  (d) random        a random eligible drug, as the exact expectation (mean over all eligible drugs)
  (e) oracle        the most similar eligible structure (upper bound)
Neighbour rules (who is eligible for a target drug):
  component-matched  (primary) the drugs outside the target's connected component of the graph rlexp.data_prep splits by: an
                     edge when the bit Tanimoto is >= data_prep.SIM or the Murcko scaffold is identical and non-empty, edges
                     chain. The real split sends whole components to one split, so Train never holds a Val/Test drug's component.
  pairwise-matched   (secondary) only direct pairs are removed (bit Tanimoto < SIM and no shared scaffold)
  any                every other drug (the specification)
Drug-balanced means. Primary 95% interval: percentile bootstrap that resamples the connected components of the target drugs
(a component's drugs stay together); secondary: resamples drugs. --boot resamples each, seeded by --seed; one set of resamples
per table, so differences are paired. Subsets restrict the target drugs only (neighbours and constants stay the whole
population): all drugs, at least 3 facts, first approved 2015 or later (ChEMBL's first_approval) and, beyond the brief, Train
drugs only. The JSON holds the numbers, the printed reading and the provenance (inputs, versions, revision, command line).
"""
import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import rdkit
from rdkit import DataStructs

from rlexp import chem
from rlexp.data_prep import SIM

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"knn": "(a) facts-kNN k=1", "medoid": "(b) facts-medoid k=5", "constant": "(c) constant, central",
           "constant_train": "(c') constant, Train-fixed", "random": "(d) random other drug", "oracle": "(e) oracle"}
SHORT = {"knn": "(a)", "medoid": "(b)", "constant": "(c)", "constant_train": "(c')"}
DIFFS = (("knn", "constant"), ("knn", "constant_train"), ("medoid", "constant"), ("medoid", "constant_train"))
THRESHOLDS = (0.4, 0.3)
PRIMARY = "component-matched"
SCOPE = ("scope: only a one-nearest-neighbour rule and a 5-medoid rule over coarsened ATC, mechanism, target and indication facts "
         "were tried. This does not bound learned predictors, generated structures, richer facts or free-text profiles, and "
         "failing to reject zero is not equivalence: an interval shows which gains the data rule out, not that there is none. "
         "Intervals are descriptive for this fixed population and these fixed rules (nothing is refit per resample).")


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


def components(B, scaffolds):
    """Connected-component label (0, 1, ...) of every drug in the graph rlexp.data_prep.assign_splits splits by: an edge when
    the bit Tanimoto B[i, j] is >= SIM or the two drugs have an identical non-empty Murcko scaffold (scaffolds[i], "" for none).
    The same construction as data_prep's (tests/test_pool.py compares the two partitions)."""
    parent = list(range(len(B)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*(a.tolist() for a in np.nonzero(np.triu(B >= SIM, 1)))):
        parent[find(i)] = find(j)
    first = {}
    for i, s in enumerate(scaffolds):
        if s:  # acyclic molecules have no scaffold and link to nothing
            parent[find(first.setdefault(s, i))] = find(i)
    labels = {}
    return np.array([labels.setdefault(find(i), len(labels)) for i in range(len(B))])


def rule_masks(B, scaffolds):
    """({rule: ok}, comp): ok[i, j] says whether drug j may answer drug i under each neighbour rule; comp = components()."""
    n = len(B)
    comp = components(B, scaffolds)
    code = {}
    scaf = np.array([code.setdefault(s, len(code)) if s else -1 - i for i, s in enumerate(scaffolds)])  # acyclic: unique, never equal
    other = ~np.eye(n, dtype=bool)
    return {PRIMARY: other & (comp[:, None] != comp[None, :]),
            "pairwise-matched": other & (B < SIM) & (scaf[:, None] != scaf[None, :]),
            "any": other}, comp


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


def scores(J, T, ok, train):
    """({method: per-drug Tanimoto of the prediction to the true structure}, {method: index of the answering drug}) for the
    predictors of the module docstring, using only the pairs ok. train: boolean mask of the Train drugs."""
    n = len(J)
    assert ok.sum(1).min() >= 1, "a drug with no eligible neighbour"
    knn, med = predictions(J, T, ok)
    order = np.argsort(-(T.sum(0) - np.diag(T)) / (n - 1), kind="stable")  # (c): most central structure first
    const = order[ok[:, order].argmax(1)]  # the most central one that is eligible for the drug
    t = train.astype(float)  # (c'): rank by the mean T to the Train references other than the candidate and the target
    refs = (t.sum() - t)[None, :] - t[:, None]  # [target, candidate]: how many references enter that mean
    mean = ((t @ T - t * np.diag(T))[None, :] - t[:, None] * T) / np.maximum(refs, 1)
    valid = ok & (refs > 0)
    assert valid.any(1).all(), "no Train reference to rank the Train-fixed constant"
    const_t = np.where(valid, mean, -np.inf).argmax(1)
    i = np.arange(n)
    picks = {"knn": knn, "medoid": med, "constant": const, "constant_train": const_t}
    S = {m: T[i, p] for m, p in picks.items()}
    S |= {"random": (T * ok).sum(1) / ok.sum(1), "oracle": np.where(ok, T, -1.0).max(1)}
    return S, picks


def bootstrap(V, boot=10000, seed=0, groups=None):
    """(mean, lo, hi): the mean over drugs (columns) of every row of V with its 95% percentile bootstrap interval. groups
    labels the unit of each drug (default: every drug is its own unit); a resample draws as many units as there are, with
    replacement, and brings all drugs of a drawn unit along, so the mean of a resample is over the drugs it contains. All rows
    share the resamples, so the interval of a difference of rows is the paired one."""
    V = np.asarray(V, float)
    n = V.shape[1]
    if n == 0:
        return (np.full(len(V), np.nan),) * 3
    unit = np.unique(np.arange(n) if groups is None else np.asarray(groups), return_inverse=True)[1].ravel()
    u = unit.max() + 1
    rng, means = np.random.default_rng(seed), np.empty((boot, len(V)))
    for s in range(0, boot, 1000):
        counts = rng.multinomial(u, np.full(u, 1.0 / u), size=min(1000, boot - s))[:, unit]  # multiplicity of every drug
        means[s:s + len(counts)] = counts @ V.T / counts.sum(1, keepdims=True)
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    return V.mean(1), lo, hi


def summarize(S, mask, comp, boot, seed):
    """{quantity: {"mean", "ci" (resampling components), "ci_drug" (resampling drugs)}} over the target drugs in mask: per
    method the mean Tanimoto and the shares at THRESHOLDS, plus the paired differences of DIFFS."""
    rows = {f"mean:{m}": S[m][mask] for m in METHODS}
    rows |= {f"share{t}:{m}": (S[m][mask] >= t).astype(float) for t in THRESHOLDS for m in METHODS}
    rows |= {f"diff:{a}-{b}": S[a][mask] - S[b][mask] for a, b in DIFFS}
    V = np.array(list(rows.values()))
    (mean, lo, hi), (_, dlo, dhi) = bootstrap(V, boot, seed, comp[mask]), bootstrap(V, boot, seed)
    return {"n": int(mask.sum()), "n_components": len(set(comp[mask].tolist())),
            **{k: {"mean": float(mean[j]), "ci": [float(lo[j]), float(hi[j])], "ci_drug": [float(dlo[j]), float(dhi[j])]}
               for j, k in enumerate(rows)}}


def verdict(lo, hi):
    """How a paired interval (kNN minus constant) reads: its position relative to 0."""
    return "beats" if lo > 0 else "falls below" if hi < 0 else "cannot be told apart from"


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
    """All tables: {"population": facts about it, "rules": {neighbour rule: {"subsets": {name: summarize(...)}, ...}}}."""
    sets = [features(r) for r in pop]
    mols = [chem.standardize(r["smiles"]) for r in pop]
    J, T, B = jaccard_matrix(sets), tanimoto_matrix(mols), tanimoto_matrix(mols, bits=True)
    n = len(pop)
    rules, comp = rule_masks(B, [r["scaffold"] for r in pop])
    train = np.array([r["source"] == "train" for r in pop])
    n_facts = np.array([len(s) for s in sets])
    year = np.array([r["first_approval"] if r["first_approval"] is not None else -1 for r in pop])
    subsets = {"all drugs": np.ones(n, bool), "at least 3 facts": n_facts >= 3, "first approved 2015 or later": year >= 2015,
               "Train drugs only": train}
    size = np.bincount(comp)
    out = {"population": {"drugs": n, "pool": int((~train).sum()), "train": int(train.sum()),
                          "median_facts": float(np.median(n_facts)), "at_least_3_facts": int((n_facts >= 3).sum()),
                          "first_approval_2015_or_later": int((year >= 2015).sum()), "unknown_year": int((year < 0).sum()),
                          "no_fact_in_common_with_any_other": int((np.where(rules["any"], J, -1.0).max(1) == 0).sum()),
                          "components": {"n": len(size), "largest": int(size.max()), "singletons": int((size == 1).sum())}},
           "rules": {}}
    for rule, ok in rules.items():
        S, picks = scores(J, T, ok, train)
        out["rules"][rule] = {"eligible_median": float(np.median(ok.sum(1))),
                              "distinct_constants": {m: len(set(picks[m].tolist())) for m in ("constant", "constant_train")},
                              "subsets": {name: summarize(S, mask, comp, boot, seed) for name, mask in subsets.items()}}
        if rule == "pairwise-matched":  # picks that the real split would not allow: the drug's own component
            same = comp[picks["knn"]] == comp
            out["population"]["pairwise_knn_pick_in_own_component"] = {"all": int(same.sum()), "train": int((same & train).sum())}
    return out


def _d(q):
    """'+0.012 [-0.003, +0.025]' of a paired quantity (mean and component interval)."""
    return f"{q['mean']:+.3f} [{q['ci'][0]:+.3f}, {q['ci'][1]:+.3f}]"


TAG = {"constant": "central", "constant_train": "Train-fixed"}


def _compare(r, m, c):
    """'beats the central constant 0.169 (paired +0.032 [+0.015, +0.051])'; when the interval includes 0 it also gives its upper end."""
    q = r[f"diff:{m}-{c}"]
    v = verdict(*q["ci"])
    tail = f"; compatible with gains up to {q['ci'][1]:+.3f}" if v.startswith("cannot") else ""
    return f"{v} the {TAG[c]} constant {r[f'mean:{c}']['mean']:.3f} (paired {_d(q)}{tail})"


def _summary(r):
    """How the facts-kNN stands against the two constants, in a few words."""
    v = [verdict(*r[f"diff:knn-{c}"]["ci"]) for c in TAG]
    return ("beats both constants" if v == ["beats"] * 2 else "beats one constant but not the other" if "beats" in v
            else "falls below both constants" if v == ["falls below"] * 2 else "does not beat either constant")


def reading(res):
    """The plain-language reading as a list of lines (printed, and stored in the JSON). Every number comes from res."""
    lines = [SCOPE]
    plan = [(PRIMARY, ("all drugs", "Train drugs only", "first approved 2015 or later")),
            ("pairwise-matched", ("all drugs", "Train drugs only")), ("any", ("all drugs",))]
    for rule, names in plan:
        for name in names:
            r = res["rules"][rule]["subsets"][name]
            parts = [f"{label} {r[f'mean:{m}']['mean']:.3f} " + " and ".join(_compare(r, m, c) for c in TAG)
                     for m, label in (("knn", "facts-kNN"), ("medoid", "5-medoid"))]
            lines.append(f"{rule}, {name} (n={r['n']}, {r['n_components']} components): " + "; ".join(parts))
    s = res["rules"][PRIMARY]["subsets"]
    lines.append(f"in short, with whole components kept apart: over all drugs the facts-kNN {_summary(s['all drugs'])}, over Train drugs "
                 f"it {_summary(s['Train drugs only'])}; the other two rules are in the tables.")
    return lines


def _cell(q, pct=False):
    f = (lambda x: f"{100 * x:5.1f}%") if pct else (lambda x: f"{x:.3f}")
    return f"{f(q['mean'])} [{f(q['ci'][0])}, {f(q['ci'][1])}]"


def report(res):
    """Plain-text tables and the reading."""
    p = res["population"]
    pk = p.get("pairwise_knn_pick_in_own_component", {})
    lines = [f"population {p['drugs']} drugs (pool {p['pool']}, Train {p['train']}); median {p['median_facts']:.0f} facts per drug; "
             f"{p['at_least_3_facts']} with at least 3 facts; {p['first_approval_2015_or_later']} first approved 2015 or later "
             f"({p['unknown_year']} without a year); {p['no_fact_in_common_with_any_other']} share no fact with any other drug; "
             f"{p['components']['n']} connected components (largest {p['components']['largest']}, {p['components']['singletons']} "
             f"single); pairwise-matched kNN picks inside the target's component: {pk.get('all')} (Train targets {pk.get('train')})"]
    for rule, rr in res["rules"].items():
        c = rr["distinct_constants"]
        lines.append(f"\nneighbours: {rule}{' (primary)' if rule == PRIMARY else ''} (median {rr['eligible_median']:.0f} eligible per "
                     f"drug; distinct constants used: central {c['constant']}, Train-fixed {c['constant_train']})")
        for name, r in rr["subsets"].items():
            lines.append(f"  targets: {name} (n={r['n']}, {r['n_components']} components)      mean Tanimoto [95% CI over "
                         f"components]    share >= 0.4 [CI]           share >= 0.3 [CI]")
            for m, label in METHODS.items():
                cells = [_cell(r[f"mean:{m}"])] + [_cell(r[f"share{t}:{m}"], True) for t in THRESHOLDS]
                lines.append(f"    {label:28}{cells[0]:28}{cells[1]:28}{cells[2]}")
            for a, b in DIFFS:
                q = r[f"diff:{a}-{b}"]
                lines.append(f"    paired {SHORT[a]} minus {SHORT[b]}: {_d(q)} over components, "
                             f"[{q['ci_drug'][0]:+.3f}, {q['ci_drug'][1]:+.3f}] over drugs")
    lines += ["", "reading"] + [f"  {x}" for x in (res.get("reading") or reading(res))]
    return "\n".join(lines)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True, timeout=30).stdout.rstrip("\n")
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(cache, boot, seed, command):
    """Where a result came from: the command line, seed and resample count, the ChEMBL release, the sha256 of the manifest and
    of the two derived files (checked against the manifest's `derived` record when build_pool.py wrote one), the git revision
    with the paths this script depends on that differ from it, this script's own sha256, and library versions."""
    cache = Path(cache)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    files = {n: {"sha256": _sha256(cache / n), "rows": (cache / n).read_bytes().count(b"\n")} for n in ("pool.jsonl", "train_facts.jsonl")}
    listed = (manifest.get("derived") or {}).get("files")
    if listed is not None and any(listed.get(n, {}).get("sha256") != f["sha256"] for n, f in files.items()):
        raise ValueError("pool.jsonl or train_facts.jsonl differs from the manifest's derived record; run scripts/build_pool.py again")
    watched = ["scripts/class_ceiling.py", "src/rlexp/chem.py", "src/rlexp/data_prep.py"]
    return {"command": command, "seed": seed, "bootstrap_resamples": boot,
            "chembl_release": manifest.get("chembl_release"), "chembl_release_date": manifest.get("release_date"),
            "inputs": {"manifest.json": {"sha256": _sha256(cache / "manifest.json")}, **files},
            "manifest_records_derived_files": listed is not None,
            "git": {"revision": _git("rev-parse", "HEAD"), "differs_from_revision": (_git("status", "--porcelain", "--", *watched) or "").splitlines(),
                    "script_sha256": _sha256(__file__)},
            "versions": {"python": platform.python_version(), "rdkit": rdkit.__version__, "numpy": np.__version__}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "data/raw/chembl_pool")
    ap.add_argument("--boot", type=int, default=10000, help="resamples for each of the two bootstraps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, help="also write the numbers, the reading and the provenance here")
    a = ap.parse_args(argv)
    python = Path(sys.executable)
    python = python.relative_to(ROOT) if python.is_relative_to(ROOT) else python  # ".venv/bin/python" when run from the repo's venv
    command = shlex.join([str(python), str(Path(__file__)) if argv is not None else sys.argv[0], *(sys.argv[1:] if argv is None else argv)])
    try:
        pop, n_rows = load_population(a.cache)
        prov = provenance(a.cache, a.boot, a.seed, command)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e} (run scripts/fetch_chembl_pool.py and scripts/build_pool.py first)")
    print(f"{n_rows} rows read, {len(pop)} drugs with a mechanism or an ATC code; {prov['chembl_release']}, seed {a.seed}, "
          f"{a.boot} resamples, pool.jsonl {prov['inputs']['pool.jsonl']['sha256'][:12]}")
    res = run(pop, a.boot, a.seed)
    res["reading"], res["provenance"] = reading(res), prov
    print(report(res))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
