#!/usr/bin/env python3
"""Scoring and statistics layer (docs/contracts.md "metrics.py"). numpy only; rlexp.reward is imported lazily.

  python -m rlexp.metrics score  --samples runs/X/samples.jsonl --data data/processed --out runs/X/scored.jsonl
  python -m rlexp.metrics report --scored runs/X/scored.jsonl [--vs runs/Y/scored.jsonl] [--split test_t,test_r,val]
                                 [--boot 10000] [--seed 0] [--data data/processed] [--json PATH]

Definitions
  unit       the drug (`group`). A drug's value for a metric is the mean over its prompt rows of the mean over each
             row's samples; drugs then count equally. n_tokens_p90 is the nearest-rank 90th percentile of the same
             drug-balanced distribution (a sample weighs 1 / (rows * samples of its row) within its drug).
  @8         per prompt row with n >= 8 samples; rows with fewer are skipped and counted, and a drug with no usable
             row has no value. exact@8 and core@8 are pass@8, 1 - C(n-c,8)/C(n,8) over all n samples (n = 8 is "any
             of 8"): oracle coverage, which needs the answer to select and is never a deployable result.
             maj@8: the plurality connectivity among the first 8 answers (by sample_idx) is the reference or one of
             its ref_alt_smiles (all spellings of the reference add up); an answer that does not parse casts no vote
             and a tie for first place earns nothing.
  intervals  95% percentile bootstrap, descriptive. The resampled units are the cluster_id components when the rows
             carry cluster_id (all drugs of a drawn component come along; the drug stays the unit of analysis
             inside it), else the drugs. Seeded per split and block, vectorised. A drug without a value for a
             metric drops out of that metric's mean in every resample.
  --vs       paired A - B on the prompt rows both runs scored (so the same drugs). For exact@1 and core@1 also the
             wins, losses and ties per drug and the two-sided sign-flip permutation p-value, the decision statistic.
             Related drugs repeat one outcome, so the flipped unit is the cluster: the drug differences are SUMMED
             within each cluster_id (the numerator of the reported drug-level mean) and the cluster totals are flipped;
             without cluster_id every drug is its own unit. All 2^n sign patterns when at most 20 units are not tied,
             else 100,000 seeded random flips. Splits are reported separately and never pooled.
  bar        the constant-answer bar scores the Train reference with the highest mean 2048-bit Morgan Tanimoto (bit
             vectors, chem.tanimoto_bits; the scorer itself uses count vectors) to the other Train references against
             every prompt row the run scored (needs --data with a train split and rlexp.reward.score).
"""
import argparse
import functools
import itertools
import json
import os
import sys
import warnings
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

DATA_SPLITS = ("train", "val", "test_r", "test_t")        # <data dir>/<split>.jsonl
SPLIT_ORDER = ("test_t", "test_r", "val", "train")        # report order
REF_KEYS = ("ref_smiles", "ref_conn", "ref_scaffold")     # required in every data row
REF_OPTIONAL = ("ref_inchikey", "ref_alt_smiles")         # handed to the scorer when the data row has them
ROW_KEYS = ("cluster_id", "ref_alt_smiles")               # copied from the data row into each scored row when present
SCORE_KEYS = ("closed", "answer", "valid", "tanimoto", "scaffold_match", "conn_match", "stereo_match", "reward")
SCORED_KEYS = ("id", "group", "split", "sample_idx", "finish_reason", "n_tokens") + SCORE_KEYS
SAMPLE_METRICS = ("exact@1", "core@1", "stereo_exact@1", "mean_tanimoto", "mean_reward", "valid_closed",
                  "truncation", "n_tokens_mean")          # means over samples, in the column order of aggregate()
METRICS = SAMPLE_METRICS + ("n_tokens_p90", "exact@8", "core@8", "maj@8")
P90, K = len(SAMPLE_METRICS), 8     # P90: the one column that is not a drug mean. K: the k of the @8 columns after it
ORACLE = ("exact@8", "core@8")
ORACLE_LABEL = "oracle coverage (needs the answer to select; not deployable)"
TESTED = ("exact@1", "core@1")      # candidate primary metrics (docs/plan.md section 2): the only ones with a p-value
BAR = ("exact@1", "core@1", "mean_tanimoto", "mean_reward")
CI = (2.5, 97.5)
TIE = 1e-9                          # |per-drug difference| below this is a tie (guards float noise in drug means)
EXACT_MAX, FLIPS = 20, 100_000      # sign-flip test: enumerate up to 20 non-tied drugs, else this many random flips


class InputError(ValueError):
    """Bad or inconsistent input files."""


# ----------------------------------------------------------------------------------------------- io
def read_jsonl(path, drop=()):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    raise InputError(f"{path}:{n}: invalid JSON ({e.msg} at column {e.colno})") from None
                for k in drop:
                    r.pop(k, None)
                rows.append(r)
    return rows


def _atomic(path, chunks):
    """Write all chunks to path or leave it untouched: no half-written scored files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(chunks)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_jsonl(path, rows):
    _atomic(path, (json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def write_json(path, obj):
    _atomic(path, [json.dumps(obj, indent=2, allow_nan=False) + "\n"])


def load_data(data_dir, keep=()):
    """Rows of the <data dir>/<split>.jsonl files that exist; prompt and completion are dropped unless named in keep."""
    drop = tuple({"prompt", "completion"} - set(keep))
    rows = []
    for s in DATA_SPLITS:
        p = Path(data_dir) / f"{s}.jsonl"
        if p.exists():
            part = read_jsonl(p, drop)
            wrong = sum(r.get("split") != s for r in part)
            if wrong:
                raise InputError(f"{p}: {wrong} row(s) whose split field is not '{s}'")
            rows += part
    if not rows:
        raise FileNotFoundError(f"no {'/'.join(DATA_SPLITS)} .jsonl files with rows under {data_dir}")
    return rows


def _index(data):
    """id -> data row; ids must be unique across the data files."""
    index = {}
    for r in data:
        miss = [k for k in ("id", "group", "split") + REF_KEYS if k not in r]
        if miss:
            raise InputError(f"a data row lacks {miss}")
        if r["id"] in index:
            raise InputError("data has duplicate ids across its files")
        index[r["id"]] = r
    return index


def _check_rows(rows, required, what):
    seen = set()
    for i, r in enumerate(rows):
        miss = [k for k in required if k not in r]
        if miss:
            raise InputError(f"{what} row {i} lacks {miss}")
        key = (r["id"], r["sample_idx"])
        if key in seen:
            raise InputError(f"{what}: duplicate (id, sample_idx); was a resumed run appended twice?")
        seen.add(key)


def _known(rows, index, what):
    unknown = {r["id"] for r in rows}.difference(index)
    if unknown:
        raise InputError(f"{what}: {len(unknown)} unknown id(s) not in the data files, e.g. "
                         f"{[i[:60] for i in sorted(unknown)[:3]]}")
    bad = [r["id"] for r in rows if any(k in r and r[k] != index[r["id"]][k] for k in ("group", "split"))]
    if bad:
        raise InputError(f"{what}: {len(bad)} row(s) whose group or split disagree with the data, e.g. "
                         f"{[i[:60] for i in bad[:3]]}")


# -------------------------------------------------------------------------------------------- scoring
def _default_scorer():
    from rlexp import reward
    if not hasattr(reward, "score"):
        raise RuntimeError("rlexp.reward.score(solution_str, ref) not found; reward.py v2 (docs/contracts.md) is required")
    return reward.score


def score_samples(samples, data_rows, scorer=None):
    """Join samples to data rows by id and add scorer(text, ref) to each: closed, answer, valid, tanimoto,
    scaffold_match, conn_match, stereo_match, reward. group, split, cluster_id and ref_alt_smiles (when the data
    has them) come from the data row, and the scorer is also handed ref_alt_smiles and ref_inchikey. Unknown ids
    raise InputError."""
    _check_rows(samples, ("id", "sample_idx", "text"), "samples")
    index = _index(data_rows)
    _known(samples, index, "samples")
    scorer = scorer or _default_scorer()
    out = []
    for s in samples:
        d = index[s["id"]]
        res = scorer(s["text"], {k: d[k] for k in REF_KEYS + REF_OPTIONAL if k in d})
        miss = [k for k in SCORE_KEYS if k not in res]
        if miss:
            raise InputError(f"scorer result lacks {miss}")
        out.append({**s, "group": d["group"], "split": d["split"], **{k: d[k] for k in ROW_KEYS if k in d}, **res})
    return out


def answer_samples(rows, answers, model):
    """One sample per data row whose text is '</think>\\n' + answer: how docs/contracts.md writes controls."""
    return [{"id": r["id"], "group": r["group"], "split": r["split"], "sample_idx": 0, "text": "</think>\n" + a,
             "finish_reason": "stop", "n_tokens": 0, "model": model, "mode": "control",
             "temperature": None, "top_p": None, "max_tokens": None, "seed": None}
            for r, a in zip(rows, answers, strict=True)]


def best_constant(refs):
    """(smiles, mean bit Tanimoto to the others) of the reference with the highest mean 2048-bit Morgan Tanimoto
    (chem.tanimoto_bits) to the other references. Bit vectors on purpose: the scorer itself now uses count
    vectors, so this is a simple Train-only rule, not the best constant under the reward. refs: {group: ref_smiles},
    one per drug; ties go to the first. About 5 s for 240 references (all pairs through the public function)."""
    from rlexp import chem
    groups = list(refs)
    mols = [chem.standardize(refs[g]) for g in groups]
    if any(m is None for m in mols):
        raise InputError(f"{sum(m is None for m in mols)} reference(s) do not standardize")
    if len(mols) < 2:
        raise InputError("need at least 2 references to pick a central one")
    S = np.zeros((len(mols), len(mols)))
    for i, j in itertools.combinations(range(len(mols)), 2):
        S[i, j] = S[j, i] = chem.tanimoto_bits(mols[i], mols[j])
    mean = S.sum(1) / (len(mols) - 1)
    i = int(mean.argmax())
    return refs[groups[i]], float(mean[i])


# ------------------------------------------------------------------------------------------ statistics
def passk(n, c, k):
    """Unbiased pass@k = 1 - C(n-c,k)/C(n,k) as the stable product 1 - prod(1 - k/j), j = n-c+1..n
    (Chen et al. 2021). NaN when n < k."""
    if n < k:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


@functools.lru_cache(maxsize=100_000)
def _conn_of(answer):
    """chem.conn of an answer string, None if it does not standardise. Memoised: answers repeat a lot."""
    from rlexp import chem
    mol = chem.standardize(answer)
    return chem.conn(mol) if mol is not None else None


def _maj(first):
    """1.0 if the plurality connectivity among these scored answers is the reference, else 0.0. An answer with
    conn_match, or whose conn equals that of one of the row's ref_alt_smiles, votes for the reference (all spellings
    of the reference add up); another parseable answer votes for its own connectivity (so two spellings of one wrong
    molecule add up too); an unparseable answer casts no vote; a tie for first place earns nothing."""
    alts = {c for c in map(_conn_of, first[0].get("ref_alt_smiles") or ()) if c}
    votes = Counter()
    for r in first:
        if r["conn_match"]:
            votes[True] += 1
        elif r["valid"] and r["answer"]:
            c = _conn_of(r["answer"])
            if c:
                votes[True if c in alts else c] += 1
    top = votes.most_common(2)
    return float(bool(top) and top[0][0] is True and (len(top) == 1 or top[0][1] > top[1][1]))


def aggregate(rows, pass_ids=None, cluster=None):
    """Scored samples of ONE split -> drug-level arrays. pass_ids restricts which prompt rows may feed the @8
    metrics. cluster maps every group to its cluster_id (the bootstrap units); None makes each drug its own unit."""
    by_id, group = {}, {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
        if group.setdefault(r["id"], r["group"]) != r["group"]:
            raise InputError("a prompt id appears under two different groups")
    per_drug = {}
    for i, rs in by_id.items():
        rs.sort(key=lambda r: r["sample_idx"])
        X = np.array([[bool(r["conn_match"]), bool(r["conn_match"]) or bool(r["scaffold_match"]),
                       bool(r["stereo_match"]), r["tanimoto"], r["reward"], bool(r["closed"]) and bool(r["valid"]),
                       r["finish_reason"] == "length", r["n_tokens"]] for r in rs], float)
        per_drug.setdefault(group[i], []).append((i, X, rs))
    drugs = sorted(per_drug)
    V = np.full((len(drugs), len(METRICS)), np.nan)
    k_ids, tok, gi, w = set(), [], [], []
    for d, g in enumerate(drugs):
        R = per_drug[g]
        V[d, :P90] = np.mean([X.mean(0) for _, X, _ in R], axis=0)
        P = []
        for i, X, rs in R:
            if len(X) >= K and (pass_ids is None or i in pass_ids):
                k_ids.add(i)
                P.append([passk(len(X), int(X[:, c].sum()), K) for c in (0, 1)] + [_maj(rs[:K])])
            tok += X[:, -1].tolist()
            gi += [d] * len(X)
            w += [1.0 / (len(R) * len(X))] * len(X)
        if P:
            V[d, P90 + 1:] = np.mean(P, axis=0)
    codes = {}      # cluster index in order of first appearance, so one-drug clusters reproduce the drug bootstrap
    clu = np.array([codes.setdefault(g if cluster is None else cluster[g], len(codes)) for g in drugs], dtype=int)
    o = np.argsort(tok, kind="stable")
    return {"drugs": drugs, "V": V, "tok": np.array(tok)[o], "gi": np.array(gi)[o], "w": np.array(w)[o],
            "clu": clu, "ids": set(by_id), "k_ids": k_ids, "n_samples": len(tok)}


def _p90(a, counts):
    """Nearest-rank weighted 90th percentile of n_tokens, one per row of counts (B, D) of drug multiplicities."""
    out, step = [], max(1, 2_000_000 // len(a["tok"]))
    for i in range(0, len(counts), step):
        cum = np.cumsum(counts[i:i + step][:, a["gi"]] * a["w"], axis=1)
        out.append(a["tok"][(cum >= 0.9 * cum[:, -1:] - 1e-9).argmax(axis=1)])    # 1e-9: rounding of the cumsum
    return np.concatenate(out)


def _reps(a, counts, b=None):
    """(B, len(METRICS)) values of every metric for drug multiplicities counts (B, D): of a, or of a - b paired."""
    V = a["V"] if b is None else a["V"] - b["V"]
    ok = ~np.isnan(V)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (counts @ np.where(ok, V, 0.0)) / (counts @ ok)
    out[:, P90] = _p90(a, counts) if b is None else _p90(a, counts) - _p90(b, counts)
    return out


def _estimate(a, boot, rng, b=None):
    """Point values, 95% percentile interval (2, M) and the number of drugs behind each metric. The bootstrap
    draws clusters (all drugs of a drawn cluster together); one-drug clusters make that the drug bootstrap."""
    V = a["V"] if b is None else a["V"] - b["V"]
    D, clu = len(a["drugs"]), a["clu"]
    C = int(clu.max()) + 1 if D else 0
    point = _reps(a, np.ones((1, D)), b)[0]
    nd = (~np.isnan(V)).sum(0)
    nd[P90] = D
    ci = np.full((2, len(METRICS)), np.nan)
    if C >= 2 and boot >= 1:
        idx = rng.integers(0, C, size=(boot, C))
        per_cluster = np.bincount((idx + C * np.arange(boot)[:, None]).ravel(), minlength=boot * C).reshape(boot, C)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # metric undefined for every drug: all-NaN column
            ci = np.nanpercentile(_reps(a, per_cluster[:, clu].astype(float), b), CI, axis=0)
        ncl = np.array([len(np.unique(clu[~np.isnan(V[:, j])])) for j in range(len(METRICS))])
        ncl[P90] = C
        ci[:, ncl < 2] = np.nan                                 # no interval from fewer than two units
    return point, ci, nd


def sign_flip_p(d, rng, flips=FLIPS):
    """Two-sided sign-flip permutation p-value for H0: each unit's total difference is as likely + as -. d holds one
    total per unit (a cluster, or a drug when there are no clusters). The statistic is T = sum(d); tied units
    (|d| < TIE) never change it and are dropped. Up to EXACT_MAX non-tied units: every one of the 2^n sign patterns,
    p = share with |T*| >= |T|. More: `flips` seeded random patterns with p = (1 + #{|T*| >= |T|}) / (flips + 1),
    which is never 0. Returns (p, method, n non-tied units)."""
    d = d[np.abs(d) >= TIE]
    x, n = np.abs(d), len(d)
    if n == 0:
        return 1.0, "exact", 0
    t = abs(float(d.sum()))
    eps = 1e-9 * float(x.sum())                                 # |T*| == |T| up to float rounding counts as >=
    if n <= EXACT_MAX:
        T = np.zeros(1)
        for v in x:
            T = np.concatenate([T + v, T - v])                  # all subset sums with signs: 2^n values
        return float((np.abs(T) >= t - eps).mean()), "exact", n
    hits = 0
    for start in range(0, flips, 5000):
        s = rng.integers(0, 2, size=(min(5000, flips - start), n)) * 2.0 - 1.0
        hits += int((np.abs(s @ x) >= t - eps).sum())
    return (1 + hits) / (flips + 1), "monte_carlo", n


def _rng(seed, split, block):
    return np.random.default_rng([seed, zlib.crc32(split.encode()), block])


def _cell(x):
    return None if np.isnan(x) else float(x)


def _block(a, boot, rng, b=None):
    """Metric table for a, or for the paired difference a - b."""
    point, ci, nd = _estimate(a, boot, rng, b)
    ones = np.ones((1, len(a["drugs"])))
    pa, pb = _reps(a, ones)[0], (_reps(b, ones)[0] if b is not None else None)
    out = {}
    for j, m in enumerate(METRICS):
        n = len(a["k_ids"]) if j > P90 else len(a["ids"])
        lo, hi = _cell(ci[0, j]), _cell(ci[1, j])
        e = {"value": _cell(point[j]), "lo": lo, "hi": hi, "n_drugs": int(nd[j]), "n_rows": n,
             "skipped_rows": len(a["ids"]) - n, "degenerate": lo is not None and lo == hi}
        if m in ORACLE:
            e["label"] = ORACLE_LABEL
        if b is not None:
            e.update(a=_cell(pa[j]), b=_cell(pb[j]), diff=e.pop("value"),
                     excludes_zero=lo is not None and (lo > 0 or hi < 0))
        out[m] = e
    return out


def _paired(ra, rb, boot, seed, split, cluster):
    """A - B on the prompt rows both runs scored; the @8 metrics only on rows where both have n >= K."""
    ids = {r["id"] for r in ra} & {r["id"] for r in rb}
    if not ids:
        return {"n_drugs": 0, "n_rows": 0, "metrics": None}
    ra, rb = [r for r in ra if r["id"] in ids], [r for r in rb if r["id"] in ids]
    na, nb = Counter(r["id"] for r in ra), Counter(r["id"] for r in rb)
    usable = {i for i in ids if na[i] >= K and nb[i] >= K}
    a, b = aggregate(ra, usable, cluster), aggregate(rb, usable, cluster)
    if a["drugs"] != b["drugs"]:
        raise InputError("the two runs disagree on the group of some prompt ids")
    out = _block(a, boot, _rng(seed, split, 1), b)
    flips = _rng(seed, split, 3)
    for m in TESTED:
        d = a["V"][:, METRICS.index(m)] - b["V"][:, METRICS.index(m)]
        ok = ~np.isnan(d)
        d, clu = d[ok], a["clu"][ok]
        # related drugs repeat one outcome: sum the drug differences inside each cluster and flip the totals as units
        p, method, n = sign_flip_p(np.bincount(clu, weights=d, minlength=int(a["clu"].max()) + 1), flips)
        out[m].update(wins=int((d >= TIE).sum()), losses=int((d <= -TIE).sum()), ties=int((np.abs(d) < TIE).sum()),
                      n_clusters=len(np.unique(clu)), n_nontied=n, p_unit="cluster" if cluster else "drug",
                      p_value=p, p_method=method)
    return {"n_drugs": len(a["drugs"]), "n_rows": len(ids), "metrics": out}


def _by_split(rows):
    out = {}
    for r in rows:
        out.setdefault(r["split"], []).append(r)
    return out


def _fill(rows, index):
    """Scored rows with cluster_id and ref_alt_smiles taken from the data row where the scored row lacks them."""
    if index is None:
        return rows
    out = []
    for r in rows:
        d = index[r["id"]]
        add = {k: d[k] for k in ROW_KEYS if k not in r and k in d}
        out.append({**r, **add} if add else r)
    return out


def _clusters(rows):
    """group -> cluster_id from the scored rows; None if any drug lacks one (then drugs are the units)."""
    cl = {}
    for r in rows:
        c = r.get("cluster_id")
        if c is None:
            return None
        if cl.setdefault(r["group"], c) != c:
            raise InputError("a drug has rows in two different clusters")
    return cl


def report(scored, vs=None, splits=None, boot=10000, seed=0, data=None, scorer=None, names=("A", "B")):
    """Per-split drug-level metrics with bootstrap intervals; JSON-able dict (undefined numbers are None).
    vs: a second run for paired A - B. data: data rows, enabling coverage, cluster ids and alternate references
    for scored rows that lack them, and the constant-answer bar."""
    _check_rows(scored, SCORED_KEYS, "scored")
    if vs is not None:
        _check_rows(vs, SCORED_KEYS, "--vs")
    index = _index(data) if data is not None else None
    if index is not None:
        _known(scored, index, "scored")
        if vs is not None:
            _known(vs, index, "--vs")
    scored, vs = _fill(scored, index), (_fill(vs, index) if vs is not None else None)
    res = {"meta": {"a": names[0], "b": names[1] if vs is not None else None, "boot": boot, "seed": seed, "ci": 95},
           "notes": [], "splits": {}}
    by, byb = _by_split(scored), _by_split(vs or [])
    order = splits or [s for s in SPLIT_ORDER if s in by] + sorted(set(by) - set(SPLIT_ORDER))
    bar = None
    if data is not None:
        train = {r["group"]: r["ref_smiles"] for r in data if r["split"] == "train"}
        try:
            if len(train) < 2:
                raise RuntimeError("fewer than 2 train references in the data")
            bar = (best_constant(train), scorer or _default_scorer())
        except Exception as e:      # the bar is optional: say why it is missing, never fail the report over it
            res["notes"].append(f"constant-answer bar skipped: {type(e).__name__}: {e}")
    for s in order:
        rows = by.get(s)
        if not rows:
            res["splits"][s] = {"n_drugs": 0, "n_rows": 0, "n_samples": 0}
            continue
        cl = _clusters(rows)
        a = aggregate(rows, cluster=cl)
        out = res["splits"][s] = {"n_drugs": len(a["drugs"]), "n_rows": len(a["ids"]), "n_samples": a["n_samples"],
                                  "bootstrap": {"unit": "cluster" if cl else "drug", "n_units": int(a["clu"].max()) + 1},
                                  "metrics": _block(a, boot, _rng(seed, s, 0))}
        if index is not None:
            tot = [r for r in data if r["split"] == s]
            out["coverage"] = {"drugs": len({r["group"] for r in tot}), "rows": len(tot)}
        if vs is not None:
            out["paired"] = _paired(rows, byb.get(s, []), boot, seed, s, cl)
        if bar is not None:
            (smiles, centrality), sc = bar
            mine = [r for r in data if r["id"] in a["ids"]]
            ab = aggregate(score_samples(answer_samples(mine, [smiles] * len(mine), "constant"), mine, sc), cluster=cl)
            out["constant"] = {"smiles": smiles, "mean_bit_tanimoto_to_train": centrality,
                               "metrics": {m: e for m, e in _block(ab, boot, _rng(seed, s, 2)).items() if m in BAR}}
    return res


# ----------------------------------------------------------------------------------------------- text
def _v(x, name, signed=False):
    if x is None:
        return "n/a"
    return format(x, f"{'+' if signed else ''}.{1 if name == 'n_tokens_mean' else 0 if name == 'n_tokens_p90' else 3}f")


def _ci(e, name, signed=False):
    return "n/a" if e["lo"] is None else f"[{_v(e['lo'], name, signed)}, {_v(e['hi'], name, signed)}]"


def _note(e):
    return " ".join(x for x in ("*" if e.get("excludes_zero") else "", "deg" if e["degenerate"] else "",
                                f"skip {e['skipped_rows']}" if e["skipped_rows"] else "") if x)


def _name(k):
    return f"  {k}" if k in ORACLE else k      # the oracle rows sit one level in, under their label


def _table(entries, line):
    """Rows of a metric table; the pass@8 rows come under the oracle label."""
    out = []
    for k, e in entries.items():
        if k == ORACLE[0]:
            out.append(f"  {ORACLE_LABEL}:")
        out.append(line(k, e))
    return out


def _p(p):
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def format_report(res):
    m = res["meta"]
    L = [f"A: {m['a']}" + (f"\nB: {m['b']}" if m["b"] else ""),
         f"{m['ci']}% percentile bootstrap, descriptive: resamples cluster_id components when the rows carry "
         f"cluster_id, else drugs; {m['boot']} resamples, seed {m['seed']}"]
    small = []
    for s, sp in res["splits"].items():
        cov = sp.get("coverage")
        of = (lambda k: f" of {cov[k]}") if cov else (lambda k: "")
        part = "  PARTIAL" if cov and (sp["n_rows"] < cov["rows"] or sp["n_drugs"] < cov["drugs"]) else ""
        L += ["", f"split {s}   drugs {sp['n_drugs']}{of('drugs')}   rows {sp['n_rows']}{of('rows')}   "
                  f"samples {sp['n_samples']}{part}" + ("   (train: memorisation check, not a claim)" if s == "train" else "")]
        if not sp["n_rows"]:
            L.append("  no samples")
            continue
        small.append(sp["n_drugs"])
        bs = sp["bootstrap"]
        L.append(f"  bootstrap resamples {bs['n_units']} clusters of {sp['n_drugs']} drugs" if bs["unit"] == "cluster"
                 else f"  bootstrap resamples the {sp['n_drugs']} drugs (rows carry no cluster_id)")
        L.append(f"  {'metric':<16}{'value':>9}  {'95% CI':<24}{'drugs':>5}{'rows':>6}   note")
        row = lambda k, e: (f"  {_name(k):<16}{_v(e['value'], k):>9}  {_ci(e, k):<24}{e['n_drugs']:>5}"
                            f"{e['n_rows']:>6}   {_note(e)}")
        L += _table(sp["metrics"], row)
        p = sp.get("paired")
        if p is not None:
            if not p["metrics"]:
                L += ["", "  paired A - B: no prompt rows in common"]
            else:
                L += ["", f"  paired A - B on {p['n_drugs']} common drugs, {p['n_rows']} common rows",
                      f"  {'metric':<16}{'A':>9}{'B':>9}{'A-B':>10}  {'95% CI (descriptive)':<26}"
                      f"{'drugs':>5}{'rows':>6}   note"]
                prow = lambda k, e: (f"  {_name(k):<16}{_v(e['a'], k):>9}{_v(e['b'], k):>9}{_v(e['diff'], k, True):>10}  "
                                     f"{_ci(e, k, True):<26}{e['n_drugs']:>5}{e['n_rows']:>6}   {_note(e)}")
                L += _table(p["metrics"], prow)
                tested = {k: e for k, e in p["metrics"].items() if "p_value" in e}
                unit = next(iter(tested.values()))["p_unit"]
                L += ["", "  sign-flip permutation test, the decision statistic. "
                      + ("The drug differences are summed within each cluster and the cluster totals flipped as units"
                         if unit == "cluster" else "Rows carry no cluster_id, so the drug differences are flipped one by one")
                      + "; ties dropped; wins, losses and ties count drugs.",
                      f"  {'metric':<16}{'wins':>6}{'losses':>8}{'ties':>6}{'clusters' if unit == 'cluster' else 'drugs':>10}"
                      f"{'non-tied':>10}{'p (2-sided)':>13}   method"]
                L += [f"  {k:<16}{e['wins']:>6}{e['losses']:>8}{e['ties']:>6}{e['n_clusters']:>10}{e['n_nontied']:>10}"
                      f"{_p(e['p_value']):>13}   "
                      + (f"exact, 2^{e['n_nontied']} sign patterns" if e["p_method"] == "exact"
                         else f"Monte Carlo, {FLIPS} flips, p = (1 + hits) / (flips + 1)")
                      for k, e in tested.items()]
        c = sp.get("constant")
        if c is not None:
            L += ["", f"  constant-answer bar (Train reference with the highest mean bit Tanimoto to the other Train "
                      f"references, {c['mean_bit_tanimoto_to_train']:.3f}), same prompt rows"]
            L += [row(k, e) for k, e in c["metrics"].items()]
    L += [""] + [f"note: {n}" for n in res["notes"]]
    L += ["drugs = drugs with a value; rows = prompt rows used; skip = rows with fewer than "
          f"{K} samples (unusable for the @{K} metrics); deg = every unit has the same value, so the interval has zero",
          "  width and understates the uncertainty (a rate of 0 over n drugs is only bounded near 3/n); "
          "* = bootstrap interval excludes 0 (descriptive).",
          "exact@8 and core@8 are oracle coverage: they need the answer to pick the sample, never a deployable result.",
          "Intervals and p-values are per metric and not adjusted for multiplicity: only the primary metric declared in "
          "docs/plan.md section 2 is confirmatory,",
          "  and for it the sign-flip p-value, not the bootstrap interval, is the decision statistic."]
    if small and min(small) < 30:
        L.append(f"A split has {min(small)} drugs (< 30): percentile intervals run somewhat narrow at this size.")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score samples.jsonl into scored.jsonl")
    s.add_argument("--samples", required=True)
    s.add_argument("--data", default="data/processed")
    s.add_argument("--out", required=True)
    r = sub.add_parser("report", help="per-split metrics with bootstrap intervals")
    r.add_argument("--scored", required=True)
    r.add_argument("--vs", help="second scored.jsonl: paired A (--scored) minus B (--vs)")
    r.add_argument("--split", help="comma-separated; default every split in the file")
    r.add_argument("--boot", type=int, default=10000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--data", default="data/processed", help="enables coverage and the constant-answer bar")
    r.add_argument("--json", help="also write the machine-readable report here")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "score":
            samples = read_jsonl(a.samples)
            scored = score_samples(samples, load_data(a.data))
            write_jsonl(a.out, scored)
            rate = lambda k: sum(bool(r[k]) for r in scored) / (len(scored) or 1)
            print(f"scored {len(scored)} samples ({len({r['id'] for r in scored})} prompt rows, "
                  f"{len({r['group'] for r in scored})} drugs) -> {a.out}; closed {rate('closed'):.3f}, "
                  f"valid {rate('valid'):.3f}, conn_match {rate('conn_match'):.3f}")
            return
        scored = read_jsonl(a.scored, drop=("text",))
        vs = read_jsonl(a.vs, drop=("text",)) if a.vs else None
        try:
            data, note = load_data(a.data), None
        except FileNotFoundError as e:
            data, note = None, f"coverage and constant-answer bar skipped: {e}"
        res = report(scored, vs, a.split.split(",") if a.split else None, a.boot, a.seed, data,
                     names=(a.scored, a.vs))
        res["notes"] += [note] if note else []
        print(format_report(res))
        if a.json:
            write_json(a.json, res)
    except (InputError, OSError, RuntimeError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
