#!/usr/bin/env python3
"""Scoring and statistics layer (docs/contracts.md "metrics.py"). numpy only; rlexp.reward and rlexp.chem (RDKit) are
imported lazily.

  python -m rlexp.metrics score  --samples runs/X/samples.jsonl --data data/processed --out runs/X/scored.jsonl
  python -m rlexp.metrics report --scored runs/X/scored.jsonl [--vs runs/Y/scored.jsonl] [--split test_t,test_r,val]
                                 [--boot 10000] [--seed 0] [--data data/processed] [--json PATH]
Each command opens only the data files of the splits that its input file names (report also reads train, for the
constant-answer bar, when --data is given): a Val run never opens test_r.jsonl or test_t.jsonl.

Definitions (v3, 2026-10-17: finished, lift, tanimoto_valid, fv_*, the endpoint, the tested set and the input checks)
  endpoint   fv_tanimoto is the single confirmatory endpoint (decision 2026-10-17): the count-vector Tanimoto of the FIRST
             valid finished answer among the first 8 samples of each prompt row (0 when none qualifies), drug-balanced,
             with its cost fv_samples reported beside it. It is the deployable resample-until-RDKit-accepts policy: it
             removes the validity confound without conditioning on validity. Its cost is stated, not matched: only the
             cap of 8 is common, so it is always read with fv_samples (draws per answer) and fv_tokens (generated tokens up
             to the first valid answer), plus prompt_tokens for the prompt. Every other metric is a diagnostic.
  unit       the drug (`group`). A drug's value for a metric is the mean over its prompt rows of the mean over each
             row's samples; drugs then count equally. n_tokens_p90 is the nearest-rank 90th percentile of the same
             drug-balanced distribution (a sample weighs 1 / (rows * samples of its row) within its drug).
  finished   a sample whose finish_reason is "stop" (one without finish_reason counts as finished). "length" (cut by the cap)
             and "abort" are failures, as in dense RFT, which keeps only "stop". `score` hands finished to a scorer that accepts
             the keyword (rlexp.reward.score does): a deployed system cannot use text that never ended.
  lift       a diagnostic of profile association. Per sample of drug d on split S with a valid, closed and finished answer
             a: T(a, ref_d) minus the mean of T(a, ref_d') over the OTHER drugs d' of S, T = chem.tanimoto (count vectors),
             one reference per drug of S in the data files given to the scorer (so a run that sampled few drugs is still
             judged against all of S). Any other sample has lift 0.0, and so does every sample of a split with a single
             drug. Stored in the scored row and averaged like the other sample metrics. A constant answer that is valid
             and finished on every row has mean lift exactly 0 over a complete panel (the terms cancel). An answer that
             does not depend on the profile has expected lift 0 only if the completion process (valid, closed, finished)
             does not depend on the drug either. If it does, through a drug-dependent finish rate or a nuisance cue that
             correlates with the drug such as prompt length, the mean is the completion-weighted sum of the drugs' lifts
             and need not vanish (one fixed molecule scores clearly above 0 when only one drug finishes): lift measures
             the association between the drug and everything the policy does, not the content of the answer alone.
             fv_lift is the lift of the first valid answer (0 when none); a constant has fv_lift 0 when every row has a
             valid answer. mean_tanimoto is the Tanimoto itself over all attempts with invalid = 0.
  tanimoto_valid  the drug mean of each row's mean Tanimoto over its valid finished samples (a row without one has no
             value, a drug without one drops out). It conditions on validity, so it is read beside valid_closed.
  valid_closed    valid, closed and finished rate.
  @8         per prompt row with n >= 8 samples; rows with fewer are skipped and counted, and a drug with no usable
             row has no value. exact@8 and core@8 are pass@8, 1 - C(n-c,8)/C(n,8) over all n samples (n = 8 is "any
             of 8"): oracle coverage, which needs the answer to select and is never a deployable result.
             maj@8: the plurality connectivity among the first 8 answers (by sample_idx) is the reference or one of
             its ref_alt_smiles (all spellings of the reference add up); an answer that does not parse casts no vote
             and a tie for first place earns nothing.
  fv_*       the first-valid policy at 8, the deployable "resample until RDKit accepts it": per prompt row the first of
             the first 8 samples (by sample_idx) that is valid, closed and finished. fv_reward, fv_tanimoto and fv_lift
             are its values (0 when none of the 8 qualifies), fv_rate the share of rows with one, fv_samples the mean
             number of samples drawn up to and including it (8 when none): the cost per answer. fv_tokens is the mean number of
             generated tokens of those samples. prompt_tokens, per split, is the mean n_prompt_tokens of the run's prompt rows
             in the data files (drug-balanced; None without --data or when a row lacks it): the row's original prompt, which an
             arm file with a longer prompt (retrieval) does not update. Rows with fewer than 8 samples are skipped and counted
             like the @8 metrics.
  input      stored scores must be finite: a NaN, inf, bool or non-numeric reward, tanimoto, lift, n_tokens or sample_idx
             raises InputError naming the field and the number of rows (nanmean would otherwise drop the drug without a word),
             and so does a flag (valid, closed, conn_match, scaffold_match, stereo_match) that is not a real boolean or an
             identifier (id, group, split) that is not a string; no value type reaches a traceback.
             Component metadata is all or nothing: rows that carry cluster_id only in part, a drug with two cluster_ids, two
             runs that disagree on a drug's cluster_id, or a scored row that disagrees with the data raise InputError.
  intervals  95% percentile bootstrap, descriptive. The resampled units are the cluster_id components when the rows
             carry cluster_id (all drugs of a drawn component come along; the drug stays the unit of analysis
             inside it), else the drugs. Seeded per split and block, vectorised. A drug without a value for a
             metric drops out of that metric's mean in every resample.
  --vs       paired A - B on the prompt rows both runs scored (so the same drugs). For fv_tanimoto, mean_tanimoto, lift,
             fv_lift, core@1 and exact@1 (TESTED) also the wins, losses and ties per drug and the two-sided sign-flip
             permutation p-value, the decision statistic (None when no drug has a value in both runs). Related drugs
             repeat one outcome, so the flipped unit is the cluster: the drug differences are SUMMED within each
             cluster_id (the numerator of the reported drug-level mean) and the cluster totals are flipped; without
             cluster_id every drug is its own unit. All 2^n sign patterns when at most 20 units are not tied, else
             100,000 seeded random flips. The test needs independent components whose summed differences are symmetric
             about 0 under the null (exchangeability or a zero mean alone is not enough): it is not assumption-free (the
             report says so once). Beside it a studentized sensitivity: the paired mean difference over its
             component-bootstrap standard error, a descriptive size in bootstrap standard errors and not a calibrated
             test (`boot_se` and `studentized` in the JSON). Splits are reported separately and never pooled.
  bar        the constant-answer bar scores the Train reference with the highest mean 2048-bit Morgan Tanimoto (bit
             vectors, chem.tanimoto_bits; the scorer itself uses count vectors) to the other Train references against
             every prompt row the run scored (needs --data with a train split and rlexp.reward.score). Each row gets the
             constant 8 times so fv_* is defined, on the rows with 8 samples in the run (the run's own fv rows); being always
             valid, the bar's fv_tanimoto is the constant's Tanimoto.
             Its lift uses the same full reference panel as the samples (every drug of the split in the data), so it is 0
             only when the run covers every drug of the split.
"""
import argparse
import functools
import inspect
import itertools
import json
import math
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
SAMPLE_METRICS = ("exact@1", "core@1", "stereo_exact@1", "lift", "mean_tanimoto", "tanimoto_valid", "mean_reward",
                  "valid_closed", "truncation", "n_tokens_mean")     # drug means of per-sample values, X's column order
FV = ("fv_reward", "fv_tanimoto", "fv_lift", "fv_rate", "fv_samples", "fv_tokens")       # the first-valid policy at K
METRICS = SAMPLE_METRICS + ("n_tokens_p90", "exact@8", "core@8", "maj@8") + FV
P90, K = len(SAMPLE_METRICS), 8     # P90: the one column that is not a drug mean. K: the k of the per-row columns after it
ORACLE = ("exact@8", "core@8")
ORACLE_LABEL = "oracle coverage (needs the answer to select; not deployable)"
FV_LABEL = "deployable with resampling: costs fv_samples generations per answer"
LABELS = {ORACLE[0]: ORACLE_LABEL, FV[0]: FV_LABEL}     # a labelled block of rows starts at its first member
TESTED = ("fv_tanimoto", "mean_tanimoto", "lift", "fv_lift", "core@1", "exact@1")    # the metrics with a p-value
BAR = ("exact@1", "core@1", "lift", "mean_tanimoto", "mean_reward", "fv_tanimoto")
BOOL_FIELDS = ("closed", "valid", "scaffold_match", "conn_match", "stereo_match")      # flags of a scored row: real booleans
FINITE_SAMPLE = ("sample_idx", "n_tokens")                            # numbers the metrics use in a sample row
FINITE_SCORED = FINITE_SAMPLE + ("reward", "tanimoto", "lift")        # ... and in a scored row (lift is absent before v3)
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


def load_data(data_dir, keep=(), splits=DATA_SPLITS):
    """Rows of the <data dir>/<split>.jsonl files that exist, for the named splits only (default all four): a file that is
    not asked for is never opened. prompt and completion are dropped unless named in keep."""
    drop = tuple({"prompt", "completion"} - set(keep))
    rows = []
    for s in DATA_SPLITS:
        p = Path(data_dir) / f"{s}.jsonl"
        if s in splits and p.exists():
            part = read_jsonl(p, drop)
            wrong = sum(r.get("split") != s for r in part)
            if wrong:
                raise InputError(f"{p}: {wrong} row(s) whose split field is not '{s}'")
            rows += part
    if not rows:
        raise FileNotFoundError(f"no {'/'.join(s for s in DATA_SPLITS if s in splits)} .jsonl files with rows under {data_dir}")
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
    """The keys are there, the identifiers are strings and sample_idx is a finite number (so no key below is unhashable or NaN), and
    no (id, sample_idx) occurs twice. InputError otherwise, never a TypeError."""
    for i, r in enumerate(rows):
        miss = [k for k in required if k not in r]
        if miss:
            raise InputError(f"{what} row {i} lacks {miss}")
    _typed(rows, (("id", _is_str, "non-string"), ("group", _is_str, "non-string"), ("split", _is_str, "non-string"),
                  ("cluster_id", _is_id, "non-scalar"), ("finish_reason", _is_text_or_none, "non-string")), what)
    _finite(rows, ("sample_idx",), what)
    seen = set()
    for r in rows:
        key = (r["id"], r["sample_idx"])
        if key in seen:
            raise InputError(f"{what}: duplicate (id, sample_idx); was a resumed run appended twice?")
        seen.add(key)


def is_number(v):
    """A finite int or float; a bool is not a number here."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


_is_str = lambda v: isinstance(v, str)
_is_bool = lambda v: isinstance(v, bool)
_is_id = lambda v: v is None or isinstance(v, (str, int)) and not isinstance(v, bool)
_is_text_or_none = lambda v: v is None or isinstance(v, str)
_FLAGS = tuple((f, _is_bool, "non-boolean") for f in BOOL_FIELDS) + (("answer", _is_text_or_none, "non-string"),)     # a scored row


def _typed(rows, spec, what):
    """InputError naming the field and the number of rows whose value has the wrong type; spec: (field, test, adjective). A field a row
    lacks is skipped. It keeps a list or dict from reaching a dict key or a sort, and the text "false" from counting as true."""
    for f, ok, adjective in spec:
        bad = sum(f in r and not ok(r[f]) for r in rows)
        if bad:
            raise InputError(f"{what}: {bad} row(s) with a {adjective} {f}")


def _finite(rows, fields, what):
    """InputError naming the field and the number of rows whose value is NaN, inf, null, text or a bool. Such a value would
    otherwise be read as 'missing' by nanmean and silently remove a drug from the estimand. A field a row lacks is skipped."""
    for f in fields:
        bad = sum(f in r and not is_number(r[f]) for r in rows)
        if bad:
            raise InputError(f"{what}: {bad} row(s) with a non-finite or non-numeric {f}; stored scores must be finite numbers")


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


def finished(r):
    """True for a sample that ended by itself: finish_reason "stop", or none recorded. "length" (cut by the cap) and "abort" are
    failures, and so is any other value; dense RFT keeps only "stop" too. The one definition of the failure rule for samples that did
    not end (the scored rows here, rlexp.ident's name matches)."""
    return r.get("finish_reason", "stop") == "stop"


def _ok(r):
    """A usable answer, one a deployed system could act on: valid, closed and finished."""
    return bool(r["valid"]) and bool(r["closed"]) and finished(r)


def _takes_finished(scorer):
    """True if scorer(text, ref, finished=...) is a legal call: the signature names `finished` or takes **kwargs.
    Read from the signature, not by catching TypeError, so an error raised inside a scorer is never mistaken for a
    missing keyword (and nothing is scored twice)."""
    try:
        params = inspect.signature(scorer).parameters.values()
    except (TypeError, ValueError):             # a callable that has no introspectable signature
        return False
    return any(p.kind is p.VAR_KEYWORD or (p.name == "finished" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))
               for p in params)


@functools.lru_cache(maxsize=1)
def _generator():
    from rdkit.Chem import rdFingerprintGenerator
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)     # chem.tanimoto's settings, pinned by a test


@functools.lru_cache(maxsize=100_000)
def _fp_of(smiles):
    """Count fingerprint of the standardised molecule of an answer or a reference, built once per distinct string: the
    vector chem.tanimoto compares, kept here because chem.tanimoto takes molecules and would rebuild both fingerprints on
    every pair (a test pins the equality). None if the string does not standardise."""
    from rlexp import chem
    mol = chem.standardize(smiles)
    return None if mol is None else _generator().GetCountFingerprint(mol)


def _split_refs(data_rows, splits):
    """{split: {group: ref_smiles}} for the given splits, one reference per drug. The rows of a drug (its views) must
    share it; the first row's is used and a drug whose rows disagree raises InputError."""
    refs, bad = {s: {} for s in splits}, {}
    for d in data_rows:
        if d["split"] in refs and refs[d["split"]].setdefault(d["group"], d["ref_smiles"]) != d["ref_smiles"]:
            bad.setdefault(d["split"], set()).add(d["group"])
    if bad:
        s, groups = next(iter(bad.items()))
        raise InputError(f"{sum(map(len, bad.values()))} drug(s) have rows with different ref_smiles (lift needs one "
                         f"reference per drug), in split {s} e.g. {[g[:60] for g in sorted(groups)[:3]]}")
    return refs


def _add_lift(rows, data_rows):
    """Set row["lift"] on every scored row: T(answer, own reference) minus the mean of T(answer, reference) over the other
    drugs of the row's split, 0.0 unless the sample is ok (valid, closed, finished) with an answer that standardises, and
    for a split of one drug. The references are those of every drug of the split in data_rows. Each distinct string is
    standardised and fingerprinted once, and each distinct (split, answer) is compared with the D references of its split
    in one bulk call, so there are at most (samples x D) Tanimoto computations."""
    refs = _split_refs(data_rows, {r["split"] for r in rows})
    pos, ref_fps, sims_of = {}, {}, {}      # per split: drug -> position, reference fingerprints; per (split, answer): T to all
    for r in rows:
        r["lift"] = 0.0
        fp = _fp_of(r["answer"]) if _ok(r) and isinstance(r["answer"], str) else None
        if fp is None:
            continue
        s = r["split"]
        if s not in ref_fps:
            groups = list(refs[s])
            fps = [_fp_of(refs[s][g]) for g in groups]
            if (n := sum(f is None for f in fps)):
                raise InputError(f"{n} reference(s) of split {s} do not standardize: lift cannot be computed")
            pos[s], ref_fps[s] = {g: i for i, g in enumerate(groups)}, fps
        if len(ref_fps[s]) < 2:             # a single drug has no other drug to compare with: lift 0.0
            continue
        if (s, r["answer"]) not in sims_of:
            from rdkit import DataStructs
            sims_of[s, r["answer"]] = np.array(DataStructs.BulkTanimotoSimilarity(fp, ref_fps[s]))
        sims, i = sims_of[s, r["answer"]], pos[s][r["group"]]
        r["lift"] = float(sims[i] - np.delete(sims, i).mean())


def score_samples(samples, data_rows, scorer=None):
    """Join samples to data rows by id and add scorer(text, ref) to each: closed, answer, valid, tanimoto,
    scaffold_match, conn_match, stereo_match, reward. group, split, cluster_id and ref_alt_smiles (when the data
    has them) come from the data row, and the scorer is also handed ref_alt_smiles and ref_inchikey. A scorer that
    accepts `finished` (rlexp.reward.score does) is also handed finished = (finish_reason != "length"), so a sample cut
    by the length cap fails; one that does not is called as before. Then lift is added (_add_lift) from the references of
    data_rows. Unknown ids, and a non-finite sample_idx, n_tokens, tanimoto or reward, raise InputError."""
    _check_rows(samples, ("id", "sample_idx", "text"), "samples")
    _finite(samples, FINITE_SAMPLE, "samples")
    index = _index(data_rows)
    _known(samples, index, "samples")
    scorer = scorer or _default_scorer()
    takes_finished = _takes_finished(scorer)
    out = []
    for s in samples:
        d = index[s["id"]]
        ref = {k: d[k] for k in REF_KEYS + REF_OPTIONAL if k in d}
        res = scorer(s["text"], ref, finished=finished(s)) if takes_finished else scorer(s["text"], ref)
        miss = [k for k in SCORE_KEYS if k not in res]
        if miss:
            raise InputError(f"scorer result lacks {miss}")
        out.append({**s, "group": d["group"], "split": d["split"], **{k: d[k] for k in ROW_KEYS if k in d}, **res})
    _finite(out, ("tanimoto", "reward"), "scorer result")
    _typed(out, _FLAGS, "scorer result")
    _add_lift(out, data_rows)
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


def _first_valid(first):
    """[reward, tanimoto, lift, found, samples drawn, tokens generated] of the first ok sample (valid, closed, finished) among
    these scored samples, the first K of a row by sample_idx; the last two count up to and including it. A row with none scores 0
    on the first three, found 0 and K draws with all K samples' tokens: the cost of giving up. A scored row from before v3 has
    no lift: NaN."""
    for n, r in enumerate(first, 1):
        if _ok(r):
            return [r["reward"], r["tanimoto"], r.get("lift", np.nan), 1.0, float(n), float(sum(x["n_tokens"] for x in first[:n]))]
    return [0.0, 0.0, 0.0, 0.0, float(len(first)), float(sum(x["n_tokens"] for x in first))]


def aggregate(rows, pass_ids=None, cluster=None):
    """Scored samples of ONE split -> drug-level arrays. pass_ids restricts which prompt rows may feed the @8
    metrics. cluster maps every group to its cluster_id (the bootstrap units); None makes each drug its own unit.
    Values that do not exist are NaN and drop out of the means: tanimoto_valid of a row without a valid finished
    sample, lift of a scored row from before v3."""
    by_id, group = {}, {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
        if group.setdefault(r["id"], r["group"]) != r["group"]:
            raise InputError("a prompt id appears under two different groups")
    per_drug = {}
    for i, rs in by_id.items():
        rs.sort(key=lambda r: r["sample_idx"])
        X = np.array([[bool(r["conn_match"]), bool(r["conn_match"]) or bool(r["scaffold_match"]),
                       bool(r["stereo_match"]), r.get("lift", np.nan), r["tanimoto"], r["tanimoto"] if _ok(r) else np.nan,
                       r["reward"], _ok(r), r["finish_reason"] == "length", r["n_tokens"]] for r in rs], float)
        per_drug.setdefault(group[i], []).append((i, X, rs))
    drugs = sorted(per_drug)
    V = np.full((len(drugs), len(METRICS)), np.nan)
    k_ids, tok, gi, w = set(), [], [], []
    for d, g in enumerate(drugs):
        R = per_drug[g]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)         # an all-NaN column has no value, not an error
            V[d, :P90] = np.nanmean([np.nanmean(X, axis=0) for _, X, _ in R], axis=0)
        P = []
        for i, X, rs in R:
            if len(X) >= K and (pass_ids is None or i in pass_ids):
                k_ids.add(i)
                P.append([passk(len(X), int(X[:, c].sum()), K) for c in (0, 1)] + [_maj(rs[:K])] + _first_valid(rs[:K]))
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
    """Point values, 95% percentile interval (2, M), the number of drugs behind each metric and the bootstrap standard
    error (M, the standard deviation of the resampled values). The bootstrap draws clusters (all drugs of a drawn
    cluster together); one-drug clusters make that the drug bootstrap."""
    V = a["V"] if b is None else a["V"] - b["V"]
    D, clu = len(a["drugs"]), a["clu"]
    C = int(clu.max()) + 1 if D else 0
    point = _reps(a, np.ones((1, D)), b)[0]
    nd = (~np.isnan(V)).sum(0)
    nd[P90] = D
    ci, se = np.full((2, len(METRICS)), np.nan), np.full(len(METRICS), np.nan)
    if C >= 2 and boot >= 1:
        idx = rng.integers(0, C, size=(boot, C))
        per_cluster = np.bincount((idx + C * np.arange(boot)[:, None]).ravel(), minlength=boot * C).reshape(boot, C)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # metric undefined for every drug: all-NaN column
            reps = _reps(a, per_cluster[:, clu].astype(float), b)
            ci, se = np.nanpercentile(reps, CI, axis=0), np.nanstd(reps, axis=0, ddof=1)
        ncl = np.array([len(np.unique(clu[~np.isnan(V[:, j])])) for j in range(len(METRICS))])
        ncl[P90] = C
        ci[:, ncl < 2], se[ncl < 2] = np.nan, np.nan            # no interval from fewer than two units
    return point, ci, nd, se


def sign_flip_p(d, rng, flips=FLIPS):
    """Two-sided sign-flip permutation p-value for H0: each unit's total difference is as likely + as -. d holds one
    total per unit (a cluster, or a drug when there are no clusters). The statistic is T = sum(d); tied units
    (|d| < TIE) never change it and are dropped. Up to EXACT_MAX non-tied units: every one of the 2^n sign patterns,
    p = share with |T*| >= |T|. More: `flips` seeded random patterns with p = (1 + #{|T*| >= |T|}) / (flips + 1),
    which is never 0. Returns (p, method, n non-tied units). A NaN or inf difference raises InputError: it would be dropped
    as a tie and change the p-value."""
    d = np.asarray(d, float)
    if not np.isfinite(d).all():
        raise InputError("sign_flip_p needs finite differences: a NaN or inf unit would be dropped as a tie")
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
    """Metric table for a, or for the paired difference a - b (which also carries boot_se, the bootstrap standard error
    of the difference)."""
    point, ci, nd, se = _estimate(a, boot, rng, b)
    ones = np.ones((1, len(a["drugs"])))
    pa, pb = _reps(a, ones)[0], (_reps(b, ones)[0] if b is not None else None)
    out = {}
    for j, m in enumerate(METRICS):
        n = len(a["k_ids"]) if j > P90 else len(a["ids"])
        lo, hi = _cell(ci[0, j]), _cell(ci[1, j])
        e = {"value": _cell(point[j]), "lo": lo, "hi": hi, "n_drugs": int(nd[j]), "n_rows": n,
             "skipped_rows": len(a["ids"]) - n, "degenerate": lo is not None and lo == hi}
        if m in ORACLE or m in FV:
            e["label"] = ORACLE_LABEL if m in ORACLE else FV_LABEL
        if b is not None:
            e.update(a=_cell(pa[j]), b=_cell(pb[j]), diff=e.pop("value"), boot_se=_cell(se[j]),
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
        if len(d):
            p, method, n = sign_flip_p(np.bincount(clu, weights=d, minlength=int(a["clu"].max()) + 1), flips)
        else:
            p, method, n = None, None, 0            # no drug has a value in both runs (an fv metric of runs with k < 8): no test
        e = out[m]
        e.update(wins=int((d >= TIE).sum()), losses=int((d <= -TIE).sum()), ties=int((np.abs(d) < TIE).sum()),
                 n_clusters=len(np.unique(clu)), n_nontied=n, p_unit="cluster" if cluster else "drug",
                 p_value=p, p_method=method,
                 # the paired mean difference over its bootstrap standard error; undefined when the SE is 0 up to float noise
                 studentized=None if e["diff"] is None or e["boot_se"] is None or not e["boot_se"] > TIE
                 else e["diff"] / e["boot_se"])
    return {"n_drugs": len(a["drugs"]), "n_rows": len(ids), "metrics": out}


def _by_split(rows):
    out = {}
    for r in rows:
        out.setdefault(r["split"], []).append(r)
    return out


def _fill(rows, index, what):
    """Scored rows with cluster_id and ref_alt_smiles taken from the data row where the scored row lacks them. A cluster_id
    that disagrees with the data row's is an InputError (the scored file belongs to another version of the data)."""
    if index is None:
        return rows
    out, clash = [], 0
    for r in rows:
        d = index[r["id"]]
        clash += r.get("cluster_id") is not None and "cluster_id" in d and r["cluster_id"] != d["cluster_id"]
        add = {k: d[k] for k in ROW_KEYS if k in d and (r.get(k) is None if k == "cluster_id" else k not in r)}
        out.append({**r, **add} if add else r)
    if clash:
        raise InputError(f"{what}: {clash} row(s) whose cluster_id disagrees with the data; score the run again with this data")
    return out


def _cluster_map(rows, what):
    """group -> cluster_id of one run, or None when no row carries one. Component metadata is all or nothing: rows that
    carry cluster_id only in part, or a drug in two clusters, raise InputError (a half-labelled run would silently change
    the unit of the bootstrap and of the sign-flip test)."""
    have = sum(r.get("cluster_id") is not None for r in rows)
    if not have:
        return None
    if have < len(rows):
        raise InputError(f"{what}: {len(rows) - have} of {len(rows)} rows lack cluster_id while the others carry it; give every "
                         "row its component (--data fills them from the data files) or none")
    cl = {}
    for r in rows:
        if cl.setdefault(r["group"], r["cluster_id"]) != r["cluster_id"]:
            raise InputError(f"{what}: a drug has rows in two different clusters")
    return cl


def _same_clusters(ca, cb):
    """A paired report needs the components of both runs: both carry cluster_id and agree on every common drug, or neither does."""
    if (ca is None) != (cb is None):
        raise InputError("run A and run B differ in carrying cluster_id: a paired test needs the same components for both "
                         "(--data fills them from the data files)")
    bad = sorted(g for g in (ca.keys() & cb.keys() if ca else ()) if ca[g] != cb[g])
    if bad:
        raise InputError(f"{len(bad)} drug(s) have different cluster_id in A and B, e.g. {[g[:60] for g in bad[:3]]}")


def _prompt_tokens(index, ids):
    """Drug-balanced mean of n_prompt_tokens over the prompt rows `ids`, from the data rows; None without data or when a row has no
    finite n_prompt_tokens. The count belongs to the row's original prompt, which arm files (retrieval, name, shuffle) keep."""
    if index is None:
        return None
    by_drug = {}
    for i in ids:
        n = index[i].get("n_prompt_tokens")
        if not is_number(n):
            return None
        by_drug.setdefault(index[i]["group"], []).append(n)
    # fsum: the same number whatever order the (set of) ids is visited in, so a report is bit-for-bit reproducible across processes
    return math.fsum(math.fsum(v) / len(v) for v in by_drug.values()) / len(by_drug) if by_drug else None


def report(scored, vs=None, splits=None, boot=10000, seed=0, data=None, scorer=None, names=("A", "B")):
    """Per-split drug-level metrics with bootstrap intervals; JSON-able dict (undefined numbers are None).
    vs: a second run for paired A - B. data: data rows, enabling coverage, cluster ids and alternate references
    for scored rows that lack them, and the constant-answer bar. Stored scores must be finite, and the cluster_id metadata
    all or nothing and the same in both runs (InputError otherwise)."""
    _check_rows(scored, SCORED_KEYS, "scored")
    _typed(scored, _FLAGS, "scored")
    _finite(scored, FINITE_SCORED, "scored")
    if vs is not None:
        _check_rows(vs, SCORED_KEYS, "--vs")
        _typed(vs, _FLAGS, "--vs")
        _finite(vs, FINITE_SCORED, "--vs")
    index = _index(data) if data is not None else None
    if index is not None:
        _known(scored, index, "scored")
        if vs is not None:
            _known(vs, index, "--vs")
    scored, vs = _fill(scored, index, "scored"), (_fill(vs, index, "--vs") if vs is not None else None)
    cl = _cluster_map(scored, "scored")
    if vs is not None:
        _same_clusters(cl, _cluster_map(vs, "--vs"))
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
        a = aggregate(rows, cluster=cl)
        out = res["splits"][s] = {"n_drugs": len(a["drugs"]), "n_rows": len(a["ids"]), "n_samples": a["n_samples"],
                                  "prompt_tokens": _prompt_tokens(index, a["ids"]),
                                  "bootstrap": {"unit": "cluster" if cl else "drug", "n_units": int(a["clu"].max()) + 1},
                                  "metrics": _block(a, boot, _rng(seed, s, 0))}
        if index is not None:
            panel = [r for r in data if r["split"] == s]        # every drug of the split in the data
            out["coverage"] = {"drugs": len({r["group"] for r in panel}), "rows": len(panel)}
            if out["coverage"]["drugs"] == 1:
                res["notes"].append(f"split {s} has a single drug in the data: lift is 0.0 by definition (no other drug "
                                    "to compare with)")
        if vs is not None:
            out["paired"] = _paired(rows, byb.get(s, []), boot, seed, s, cl)
        if bar is not None:
            (smiles, centrality), sc = bar
            mine = [r for r in data if r["id"] in a["ids"]]
            # K copies of the constant per row, so fv_* is defined; lift against the same full panel as the samples' lift
            draws = [dict(x, sample_idx=j) for x in answer_samples(mine, [smiles] * len(mine), "constant") for j in range(K)]
            ab = aggregate(score_samples(draws, panel, sc), pass_ids=a["k_ids"], cluster=cl)     # the run's own fv rows
            out["constant"] = {"smiles": smiles, "mean_bit_tanimoto_to_train": centrality,
                               "metrics": {m: e for m, e in _block(ab, boot, _rng(seed, s, 2)).items() if m in BAR}}
    for name, rs in ((names[0], scored), (names[1], vs or [])):
        if old := sum("lift" not in r for r in rs):
            res["notes"].append(f"{name}: {old} of {len(rs)} scored rows have no lift (scored before v3): lift is n/a for them "
                                "and samples cut by the length cap may still count as valid; re-run `score` for the v3 rules")
    return res


# ----------------------------------------------------------------------------------------------- text
def _v(x, name, signed=False):
    if x is None:
        return "n/a"
    x = 0.0 if abs(x) < 1e-12 else x        # float noise around zero (a constant answer's lift) prints as 0, never -0.000
    return format(x, f"{'+' if signed else ''}.{1 if name in ('n_tokens_mean', 'fv_tokens') else 0 if name == 'n_tokens_p90' else 3}f")


def _ci(e, name, signed=False):
    return "n/a" if e["lo"] is None else f"[{_v(e['lo'], name, signed)}, {_v(e['hi'], name, signed)}]"


def _note(e):
    return " ".join(x for x in ("*" if e.get("excludes_zero") else "", "deg" if e["degenerate"] else "",
                                f"skip {e['skipped_rows']}" if e["skipped_rows"] else "") if x)


def _name(k):
    return f"  {k}" if k in ORACLE or k in FV else k     # the rows of a labelled block sit one level in, under their label


def _table(entries, line):
    """Rows of a metric table; the oracle rows and the first-valid rows come under their labels."""
    out = []
    for k, e in entries.items():
        if k in LABELS:
            out.append(f"  {LABELS[k]}:")
        out.append(line(k, e))
    return out


def _p(p):
    return "n/a" if p is None else f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def format_report(res):
    m = res["meta"]
    L = [f"A: {m['a']}" + (f"\nB: {m['b']}" if m["b"] else ""),
         f"{m['ci']}% percentile bootstrap, descriptive: resamples cluster_id components when the rows carry "
         f"cluster_id, else drugs; {m['boot']} resamples, seed {m['seed']}"]
    small, paired_shown = [], False
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
        pt = sp.get("prompt_tokens")
        L.append("  prompt_tokens " + ("n/a" if pt is None else f"{pt:.1f}") + " (mean n_prompt_tokens of the prompt rows, from the data files: the "
                 "original prompt; fv_tokens + prompt_tokens is the token cost of an answer if the prompt is read once)")
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
                tested = {k: p["metrics"][k] for k in TESTED}        # the confirmatory endpoint first
                unit = next(iter(tested.values()))["p_unit"]
                L += ["", "  sign-flip permutation test, the decision statistic. "
                      + ("The drug differences are summed within each cluster and the cluster totals flipped as units"
                         if unit == "cluster" else "Rows carry no cluster_id, so the drug differences are flipped one by one")
                      + "; ties dropped; wins, losses and ties count drugs.",
                      f"  {'metric':<16}{'wins':>6}{'losses':>8}{'ties':>6}{'clusters' if unit == 'cluster' else 'drugs':>10}"
                      f"{'non-tied':>10}{'p (2-sided)':>13}   method"]
                L += [f"  {k:<16}{e['wins']:>6}{e['losses']:>8}{e['ties']:>6}{e['n_clusters']:>10}{e['n_nontied']:>10}"
                      f"{_p(e['p_value']):>13}   "
                      + ("no drug with a value in both runs" if e["p_method"] is None
                         else f"exact, 2^{e['n_nontied']} sign patterns" if e["p_method"] == "exact"
                         else f"Monte Carlo, {FLIPS} flips, p = (1 + hits) / (flips + 1)")
                      for k, e in tested.items()]
                t = lambda e: "n/a" if e.get("studentized") is None else format(e["studentized"], "+.2f")
                L += ["", "  studentized component sensitivity (descriptive): paired mean difference / its bootstrap SE over "
                          f"{m['boot']} resampled {'clusters' if unit == 'cluster' else 'drugs'}",
                      "    " + "   ".join(f"{k} {t(e)}" for k, e in tested.items())
                      + "   (a size in bootstrap standard errors, not a calibrated test: it can disagree with the p-value)"]
                paired_shown = True
        c = sp.get("constant")
        if c is not None:
            L += ["", f"  constant-answer bar (Train reference with the highest mean bit Tanimoto to the other Train "
                      f"references, {c['mean_bit_tanimoto_to_train']:.3f}), same prompt rows, lift against every drug of the split",
                  f"  {'metric':<16}{'value':>9}  {'95% CI':<24}{'drugs':>5}{'rows':>6}   note"]
            L += [f"  {k:<16}{_v(e['value'], k):>9}  {_ci(e, k):<24}{e['n_drugs']:>5}{e['n_rows']:>6}   {_note(e)}"
                  for k, e in c["metrics"].items()]
    L += [""] + [f"note: {n}" for n in res["notes"]]
    L += ["drugs = drugs with a value; rows = prompt rows used; skip = rows with fewer than "
          f"{K} samples (unusable for the @{K} metrics); deg = every unit has the same value, so the interval has zero",
          "  width and understates the uncertainty (a rate of 0 over n drugs is only bounded near 3/n); "
          "* = bootstrap interval excludes 0 (descriptive).",
          "exact@8 and core@8 are oracle coverage: they need the answer to pick the sample, never a deployable result.",
          f"fv_tanimoto is the confirmatory endpoint: the Tanimoto of the first valid finished answer among the first {K} samples of "
          "a row (0 if none). Its cost is stated, not matched:",
          "  read it with fv_samples (draws per answer) and fv_tokens (generated tokens up to the first valid answer; prompt_tokens adds the "
          "prompt): an arm that invalidates its weak answers can raise it by paying in draws.",
          "  every other metric is a diagnostic. tanimoto_valid conditions on a valid finished answer: read it beside valid_closed.",
          "lift = Tanimoto of the answer to its own drug's reference minus its mean Tanimoto to the other drugs' references of the "
          "split (0 for an invalid, unclosed or length-terminated answer); fv_lift is the lift of the first valid answer;",
          "  they measure the association between the drug and everything the policy does. A constant answer has lift 0 over a "
          "complete panel only if completion does not depend on the drug",
          "  (a drug-dependent finish rate, or a cue such as prompt length, can raise lift without the answer using the profile); "
          "fv_lift of a constant is 0 when every row has a valid answer.",
          "Intervals and p-values are per metric and not adjusted for multiplicity: only fv_tanimoto, the endpoint declared in "
          "docs/contracts.md, is confirmatory,",
          "  and for it the sign-flip p-value, not the bootstrap interval, is the decision statistic."]
    if paired_shown:
        L += ["Sign-flip test assumptions: the components (cluster_id groups, or single drugs when rows carry no cluster_id) are "
              "the exchangeable units and are independent;",
              "  flipping the sign of each component's summed A - B difference is valid under a null in which those sums are "
              "symmetric about zero (a zero mean alone is not enough);",
              "  the p-value holds only under those assumptions: it is not assumption-free."]
    if small and min(small) < 30:
        L.append(f"A split has {min(small)} drugs (< 30): percentile intervals run somewhat narrow at this size.")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ cli
def _splits_of(rows, what):
    """The splits that an input file names. The data files to open depend on it, so a row without one is an InputError."""
    if (miss := sum("split" not in r for r in rows)):
        raise InputError(f"{what}: {miss} row(s) lack split, so the data files to read cannot be chosen")
    return {r["split"] for r in rows}


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
            need = _splits_of(samples, "samples")
            scored = score_samples(samples, load_data(a.data, splits=need) if need else [])
            write_jsonl(a.out, scored)
            rate = lambda k: sum(bool(r[k]) for r in scored) / (len(scored) or 1)
            print(f"scored {len(scored)} samples ({len({r['id'] for r in scored})} prompt rows, "
                  f"{len({r['group'] for r in scored})} drugs) -> {a.out}; closed {rate('closed'):.3f}, "
                  f"valid {rate('valid'):.3f}, conn_match {rate('conn_match'):.3f}")
            return
        wanted = set(a.split.split(",")) if a.split else None
        scored = [r for r in read_jsonl(a.scored, drop=("text",)) if wanted is None or r.get("split") in wanted]
        need = _splits_of(scored, "scored")
        vs = None
        if a.vs:                        # the other run only matters on the splits that are reported
            vs = [r for r in read_jsonl(a.vs, drop=("text",)) if "split" not in r or r["split"] in need]
        try:
            data, note = load_data(a.data, splits=need | {"train"}), None
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
