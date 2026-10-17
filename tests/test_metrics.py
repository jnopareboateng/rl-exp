"""metrics.py on synthetic data with a fake scorer; the lift and finished tests also use the real rlexp.reward.score."""
import functools
import itertools
import json
import math
import random
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import rlexp
from rlexp import chem, metrics

# label -> score() keys. The fake scorer reads the label after </think>.
OUT = {
    "exact": dict(closed=True, valid=True, tanimoto=1.0, scaffold_match=True, conn_match=True, stereo_match=True, reward=1.0),
    "flat": dict(closed=True, valid=True, tanimoto=0.9, scaffold_match=True, conn_match=True, stereo_match=False, reward=1.0),
    "scaf": dict(closed=True, valid=True, tanimoto=0.5, scaffold_match=True, conn_match=False, stereo_match=False, reward=0.65),
    "miss": dict(closed=True, valid=True, tanimoto=0.1, scaffold_match=False, conn_match=False, stereo_match=False, reward=0.11),
    "bad": dict(closed=True, valid=False, tanimoto=0.0, scaffold_match=False, conn_match=False, stereo_match=False, reward=0.0),
    "open": dict(closed=False, valid=False, tanimoto=0.0, scaffold_match=False, conn_match=False, stereo_match=False, reward=0.0),
}
LIFT = {"exact": 0.9, "flat": 0.8, "scaf": 0.4, "miss": 0.05, "bad": 0.0, "open": 0.0}     # lift of a hand-made scored row


def fake_scorer(text, ref):
    """OUT[label]; an answer equal to ref_smiles is exact, so it only scores right if the ref was joined right."""
    if "</think>" not in text:
        return {**OUT["open"], "answer": None}
    tail = text.rsplit("</think>", 1)[1].strip()
    label = "exact" if tail == ref["ref_smiles"] else tail
    return {**OUT.get(label, OUT["miss"]), "answer": None if label == "bad" else tail}


def scored(split, group, row, labels, tok=100, fin="stop", answers=None, cluster=None, lifts=None):
    """Scored samples of one prompt row, one per label. answers: the extracted SMILES per sample (default None).
    lifts: the lift per sample (default LIFT[label]: a v3 scored row has one)."""
    return [{"id": f"{group}/{row}", "group": group, "split": split, "sample_idx": i, "finish_reason": fin,
             "n_tokens": tok, "answer": answers[i] if answers else None,
             **({"cluster_id": cluster} if cluster is not None else {}), **OUT[lab],
             "lift": lifts[i] if lifts else LIFT[lab]} for i, lab in enumerate(labels)]


def valued(split, group, row, tanimoto, lift, n=8, cluster=None):
    """n identical valid samples of one prompt row with a chosen tanimoto and lift (a plain miss otherwise)."""
    return [dict(r, tanimoto=tanimoto, lift=lift) for r in scored(split, group, row, ["miss"] * n, cluster=cluster)]


def data_row(split, group, row, smiles=None, **extra):
    return {"id": f"{group}/{row}", "group": group, "task_type": "A", "split": split,
            "ref_smiles": smiles or f"smi-{group}", "ref_conn": f"conn-{group}", "ref_scaffold": f"scaf-{group}", **extra}


def m(res, split="val", metric="exact@1"):
    return res["splits"][split]["metrics"][metric]


def pm(res, split="val", metric="exact@1"):
    return res["splits"][split]["paired"]["metrics"][metric]


def _eight(split, group, row, label):
    return scored(split, group, row, [label] * 8)


# ------------------------------------------------------------------------------------------ pass@k
@pytest.mark.parametrize("n,c,k,want", [
    (8, 0, 8, 0.0), (8, 1, 8, 1.0), (8, 8, 8, 1.0),      # n = k = 8: any of 8
    (16, 1, 8, 0.5),                                      # 1 - C(15,8)/C(16,8) = 1 - 6435/12870
    (4, 1, 2, 0.5),                                       # 1 - C(3,2)/C(4,2) = 1 - 3/6
    (5, 2, 3, 0.9),                                       # 1 - C(3,3)/C(5,3) = 1 - 1/10
    (6, 1, 3, 0.5),                                       # 1 - C(5,3)/C(6,3) = 1 - 10/20
    (10, 3, 1, 0.3), (3, 1, 1, 1 / 3), (6, 0, 3, 0.0),
])
def test_passk_hand_values(n, c, k, want):
    assert metrics.passk(n, c, k) == pytest.approx(want, abs=1e-12)


def test_passk_short_row_is_nan():
    assert math.isnan(metrics.passk(7, 3, 8))


def test_passk_matches_exact_combinatorics():
    for n in range(1, 25):
        for c in range(n + 1):
            for k in range(1, n + 1):
                exact = 1 - Fraction(math.comb(n - c, k), math.comb(n, k))
                assert metrics.passk(n, c, k) == pytest.approx(float(exact), abs=1e-12)
    # stable at sizes where naive float binomials overflow
    exact = 1 - Fraction(math.comb(4000, 100), math.comb(5000, 100))
    assert metrics.passk(5000, 1000, 100) == pytest.approx(float(exact), abs=1e-12)


# --------------------------------------------------------------------------------------- weighting
def test_metric_set_and_order_follow_the_contract():
    assert metrics.METRICS == ("exact@1", "core@1", "stereo_exact@1", "lift", "mean_tanimoto", "tanimoto_valid",
                               "mean_reward", "valid_closed", "truncation", "n_tokens_mean", "n_tokens_p90",
                               "exact@8", "core@8", "maj@8", "fv_reward", "fv_tanimoto", "fv_lift", "fv_rate", "fv_samples",
                               "fv_tokens")
    assert metrics.METRICS.index("n_tokens_p90") == metrics.P90 and metrics.K == 8
    assert metrics.TESTED == ("fv_tanimoto", "mean_tanimoto", "lift", "fv_lift", "core@1", "exact@1")     # fv_tanimoto: the endpoint
    assert metrics.BAR == ("exact@1", "core@1", "lift", "mean_tanimoto", "mean_reward", "fv_tanimoto")
    res = metrics.report(_eight("val", "a", 0, "exact"), boot=20)
    assert tuple(res["splits"]["val"]["metrics"]) == metrics.METRICS


def test_drugs_count_equally_not_rows():
    rows = [r for i in range(6) for r in scored("val", "big", i, ["exact"] * 2)] + scored("val", "small", 0, ["miss"] * 2)
    res = metrics.report(rows, boot=50)
    sp = res["splits"]["val"]
    assert (sp["n_drugs"], sp["n_rows"], sp["n_samples"]) == (2, 7, 14)
    assert m(res)["value"] == pytest.approx(0.5)                       # not 6/7
    assert m(res, metric="mean_tanimoto")["value"] == pytest.approx((1.0 + 0.1) / 2)
    assert (m(res)["n_drugs"], m(res)["n_rows"]) == (2, 7)


def test_prompt_rows_count_equally_within_a_drug():
    # row X/0 has 1 sample (exact), row X/1 has 3 (all miss): drug value (1 + 0) / 2, not 1/4
    rows = scored("val", "X", 0, ["exact"]) + scored("val", "X", 1, ["miss"] * 3) + scored("val", "Y", 0, ["exact"] * 2)
    assert m(metrics.report(rows, boot=50))["value"] == pytest.approx((0.5 + 1.0) / 2)


def test_stereo_exact_at_1_is_the_stereo_match_mean_over_drugs():
    # "flat": right connectivity, stereo not matched. It counts for exact@1 and core@1 but not for stereo_exact@1
    rows = scored("val", "A", 0, ["exact", "flat", "miss", "miss"]) + scored("val", "B", 0, ["exact"] * 2)
    res = metrics.report(rows, boot=50)
    assert m(res, metric="stereo_exact@1")["value"] == pytest.approx((0.25 + 1.0) / 2)
    assert m(res)["value"] == pytest.approx((0.5 + 1.0) / 2)
    assert m(res, metric="core@1")["value"] == pytest.approx((0.5 + 1.0) / 2)
    e = m(res, metric="stereo_exact@1")
    assert (e["n_drugs"], e["n_rows"], e["skipped_rows"]) == (2, 2, 0)      # a per-sample metric: every row counts


def test_p90_is_drug_balanced_nearest_rank():
    # pooled, 36 of 40 samples are short so the pooled p90 is 100; with one unit of weight per drug it is 1000
    rows = scored("val", "L", 0, ["miss"] * 4, tok=1000)
    rows += [r for i in range(9) for r in scored("val", "S", i, ["miss"] * 4, tok=100)]
    res = metrics.report(rows, boot=50)
    assert m(res, metric="n_tokens_p90")["value"] == 1000
    assert m(res, metric="n_tokens_mean")["value"] == pytest.approx(550)
    # one row of tokens 1..10: nearest rank gives 9
    rows = [dict(r, n_tokens=t) for t, r in zip(range(1, 11), scored("val", "Z", 0, ["miss"] * 10))]
    assert m(metrics.report(rows, boot=50), metric="n_tokens_p90")["value"] == 9


def test_truncation_and_valid_closed_accounting():
    # valid_closed is valid AND closed AND finished: bad is not valid, open is not closed, and the last sample is a
    # parseable answer that the length cap cut, which a deployed system could not use (v3), so it fails too
    rows = (scored("val", "A", 0, ["exact"], fin="stop") + scored("val", "A", 0, ["open"], fin="length")
            + scored("val", "A", 0, ["bad"], fin="stop") + scored("val", "A", 0, ["miss"], fin="length"))
    for i, r in enumerate(rows):
        r["sample_idx"] = i
    rows += scored("val", "B", 0, ["exact"])
    res = metrics.report(rows, boot=50)
    assert m(res, metric="valid_closed")["value"] == pytest.approx((1 / 4 + 1) / 2)
    assert m(res, metric="truncation")["value"] == pytest.approx((2 / 4 + 0) / 2)
    assert m(res, metric="core@1")["value"] == pytest.approx((1 / 4 + 1) / 2)


# ---------------------------------------------------------------------------------- pass@k and n<k
def test_pass_at_8_skips_short_rows_and_counts_them():
    rows = scored("val", "d1", 0, ["exact"] + ["miss"] * 7) + scored("val", "d1", 1, ["miss"] * 8)
    rows += scored("val", "d2", 0, ["exact"] * 8) + scored("val", "d2", 1, ["exact"] * 3)
    rows += scored("val", "d3", 0, ["exact"] * 3)
    res = metrics.report(rows, boot=50)
    e = m(res, metric="exact@8")
    assert e["value"] == pytest.approx((0.5 + 1.0) / 2)       # d3 has no usable row; d2/1 is skipped
    assert (e["n_drugs"], e["n_rows"], e["skipped_rows"]) == (2, 3, 2)
    assert m(res)["skipped_rows"] == 0 and m(res)["n_rows"] == 5     # @1 uses every row


def test_pass_at_8_is_unbiased_with_more_samples_and_undefined_when_all_short():
    rows = scored("val", "d", 0, ["exact"] + ["miss"] * 15)       # n=16, c=1: 1 - C(15,8)/C(16,8) = 0.5
    assert m(metrics.report(rows, boot=50), metric="exact@8")["value"] == pytest.approx(0.5)
    res = metrics.report(scored("val", "d", 0, ["exact"] * 4) + scored("val", "e", 0, ["miss"] * 4), boot=50)
    for k in ("exact@8", "core@8", "maj@8"):
        e = m(res, metric=k)
        assert e["value"] is None and e["lo"] is None and e["n_drugs"] == 0 and e["skipped_rows"] == 2


def test_pass_at_8_columns_are_labelled_oracle_coverage_in_json_and_text():
    label = "oracle coverage (needs the answer to select; not deployable)"
    fv_label = "deployable with resampling: costs fv_samples generations per answer"
    fv = ("fv_reward", "fv_tanimoto", "fv_lift", "fv_rate", "fv_samples", "fv_tokens")
    want = lambda k: label if k in ("exact@8", "core@8") else fv_label if k in fv else None
    res = metrics.report(_eight("val", "a", 0, "exact") + _eight("val", "b", 0, "miss"), vs=_eight("val", "a", 0, "miss")
                         + _eight("val", "b", 0, "miss"), boot=50)
    for k, e in res["splits"]["val"]["metrics"].items():
        assert e.get("label") == want(k), k
    for k, e in res["splits"]["val"]["paired"]["metrics"].items():
        assert e.get("label") == want(k), k
    text = metrics.format_report(res)
    assert text.count(label + ":") == 2                                  # above the pass@8 rows of both tables
    assert text.count(fv_label + ":") == 2                               # and above the first-valid rows of both
    first = text.index(label)
    assert first < text.index("exact@8") < text.index("core@8") < text.index("maj@8", first)   # maj@8 comes after the block
    start = text.index(fv_label)
    assert start > text.index("maj@8", first)                            # the first-valid block follows maj@8
    at = [text.index(k, start + len(fv_label)) for k in fv]              # its six rows come under the label, in order
    assert at == sorted(at) and len(set(at)) == 6
    json.dumps(res, allow_nan=False)                                     # and the labels survive into the JSON


# ----------------------------------------------------------------------------------------- maj@8
SMILES = ["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "CC(C(=O)O)c1cccc(c1)C(=O)c1ccccc1",
          "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1",
          "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", "CN1CCCC1c1cccnc1"]       # ibuprofen, naproxen, ketoprofen, caffeine, ...
IBU, NAP, KET, CAF = SMILES[:4]
IBU2 = "OC(=O)C(C)c1ccc(CC(C)C)cc1"                                    # ibuprofen written another way


def votes(answers, group="d", row=0):
    """One prompt row whose samples answer 'ref' (right connectivity), 'bad' (nothing parseable) or a SMILES."""
    labels = ["exact" if a == "ref" else "bad" if a == "bad" else "miss" for a in answers]
    return scored("val", group, row, labels, answers=[None if a in ("ref", "bad") else a for a in answers])


def maj(answers):
    return m(metrics.report(votes(answers), boot=20), metric="maj@8")["value"]


@pytest.mark.parametrize("answers,want", [
    (["ref"] * 5 + [IBU] * 3, 1.0),                              # the reference is the plurality
    (["ref"] * 3 + [IBU] * 5, 0.0),                              # a wrong molecule is the plurality
    (["ref"] * 4 + [IBU] * 4, 0.0),                              # a tie for first earns nothing
    (["ref"] * 3 + [IBU] * 2 + [NAP, KET, CAF], 1.0),            # 3 beats 2, 1, 1, 1 although 5 answers are wrong
    (["ref"] * 3 + ["bad"] * 5, 1.0),                            # unparseable answers cast no vote
    (["ref"] + ["bad"] * 7, 1.0),                                # a lone vote is a plurality
    (["bad"] * 8, 0.0),                                          # nobody voted: nothing to earn
    ([IBU] * 4 + [NAP] * 4, 0.0),                                # the reference got no vote
    (["ref"] * 2 + [IBU, NAP] + ["bad"] * 4, 1.0),               # 2 beats 1 and 1
    (["ref"] * 2 + [IBU] * 2 + ["bad"] * 4, 0.0),                # tie between the reference and a wrong molecule
    (["ref"] * 3 + [IBU] * 2 + [IBU2] * 2 + [NAP], 0.0),         # two spellings of one wrong molecule add up: 4 > 3
])
def test_maj_at_8_plurality_rules(answers, want):
    assert maj(answers) == want


def test_maj_at_8_votes_by_connectivity_not_by_string():
    from rlexp import chem
    conn = lambda smiles: chem.conn(chem.standardize(smiles))
    assert conn(IBU) == conn(IBU2)                               # premise: two spellings, one connectivity
    r, s = "C[C@H](c1ccc(CC(C)C)cc1)C(=O)O", "C[C@@H](c1ccc(CC(C)C)cc1)C(=O)O"
    assert conn(r) == conn(s)                                    # premise: enantiomers share it
    t1, t2 = "O=c1cccc[nH]1", "Oc1ccccn1"                        # 2-pyridone and 2-hydroxypyridine
    assert conn(t1) == conn(t2)                                  # premise: the identity is tautomer-tolerant (contract)
    for a, b in ((r, s), (t1, t2)):                              # 4 votes for one wrong molecule beat the reference's 3
        assert maj(["ref"] * 3 + [a] * 2 + [b] * 2 + [NAP]) == 0.0
    assert maj(["ref"] * 3 + [t1] * 2 + [KET] * 2 + [NAP]) == 1.0     # control: spellings that differ keep their votes apart


def test_maj_at_8_uses_the_first_eight_answers_by_sample_idx():
    first = ["ref"] * 3 + [IBU] * 2 + [NAP, KET, CAF]            # sample_idx 0..7: the reference wins 3 to 2
    rows = votes(first + [IBU, IBU])                             # samples 8 and 9 would give ibuprofen 4 to 3 if counted
    assert m(metrics.report(rows, boot=20), metric="maj@8")["value"] == 1.0
    assert m(metrics.report(rows[::-1], boot=20), metric="maj@8")["value"] == 1.0    # list order does not matter
    random.Random(1).shuffle(rows)
    assert m(metrics.report(rows, boot=20), metric="maj@8")["value"] == 1.0


def test_maj_at_8_skips_short_rows_and_weights_drugs_equally():
    rows = votes(["ref"] * 7, "short") + votes(["ref"] * 8, "a") + votes([IBU] * 8, "b") + votes(["ref"] * 8, "b", row=1)
    e = m(metrics.report(rows, boot=20), metric="maj@8")
    assert e["value"] == pytest.approx((1.0 + 0.5) / 2)          # drug b: one row wrong, one right; "short" has no usable row
    assert (e["n_drugs"], e["n_rows"], e["skipped_rows"]) == (2, 3, 1)


KETO, ENOL = "CC(=O)c1ccccc1", "C=C(O)c1ccccc1"       # acetophenone and its enol: different conn, same skeleton


def alt_votes(answers, alts, group="d"):
    return [dict(r, ref_alt_smiles=alts) for r in votes(answers, group)]


def maj_alt(answers, alts):
    return m(metrics.report(alt_votes(answers, alts), boot=20), metric="maj@8")["value"]


def test_alternate_references_are_registry_spellings_with_their_own_strict_key():
    from rlexp import chem
    conn = lambda smiles: chem.conn(chem.standardize(smiles))
    assert conn(KETO) != conn(ENOL) and chem.skeleton(chem.standardize(KETO)) == chem.skeleton(chem.standardize(ENOL))


@pytest.mark.parametrize("answers,alts,want", [
    ([ENOL] * 8, [ENOL], 1.0),                                   # every answer is the registered alternate
    ([ENOL] * 8, [], 0.0),                                       # ... and without the registry entry it is another molecule
    (["ref"] * 3 + [ENOL] * 3 + [IBU] * 2, [ENOL], 1.0),         # the two spellings add up: 6 against 2
    (["ref"] * 3 + [ENOL] * 3 + [IBU] * 2, [], 0.0),             # unregistered: 3 against 3 is a tie
    (["ref"] * 2 + [ENOL] + [IBU] * 5, [ENOL], 0.0),             # an alternate does not rescue a wrong plurality
    (["ref"] * 2 + [ENOL] * 2 + [IBU] * 3 + ["bad"], [ENOL], 1.0),    # 4 against 3, one unparseable answer casts nothing
    ([ENOL] * 3 + [IBU] * 3 + [NAP] * 2, [ENOL], 0.0),           # the alternate is 3 and ties with ibuprofen's 3
    (["ref"] * 4 + [ENOL] * 4, ["not a smiles", ENOL], 1.0),     # an alternate that does not parse is skipped, others still work
    ([KETO] * 8, [ENOL], 0.0),                                   # a spelling that is not registered stays another molecule
])
def test_maj_at_8_counts_alternate_spellings_of_the_reference(answers, alts, want):
    assert maj_alt(answers, alts) == want


def test_alternates_missing_from_the_scored_rows_come_from_the_data_rows():
    rows = votes([ENOL] * 8)                                     # scored rows without ref_alt_smiles
    data = [data_row("val", "d", 0, ref_alt_smiles=[ENOL])]
    assert m(metrics.report(rows, boot=20, data=data), metric="maj@8")["value"] == 1.0
    assert m(metrics.report(rows, boot=20), metric="maj@8")["value"] == 0.0           # nothing to learn them from
    own = [dict(r, ref_alt_smiles=[]) for r in rows]                                   # the scored row's own list wins
    assert m(metrics.report(own, boot=20, data=data), metric="maj@8")["value"] == 0.0


# ----------------------------------------------------------------------------------------- intervals
def test_bootstrap_coverage_of_a_known_proportion():
    hits, sims, p, D = 0, 200, 0.3, 60
    for s in range(sims):
        win = np.random.default_rng(1000 + s).random(D) < p
        rows = [r for d in range(D) for r in scored("val", f"d{d}", 0, ["exact" if win[d] else "miss"])]
        e = m(metrics.report(rows, boot=400, seed=s))
        hits += e["lo"] <= p <= e["hi"]
    assert 0.92 <= hits / sims <= 0.985         # nominal 0.95 (a 90% interval would land near 0.89)


def test_interval_width_matches_binomial_theory():
    rows = [r for d in range(100) for r in scored("val", f"d{d}", 0, ["exact" if d < 30 else "miss"])]
    e = m(metrics.report(rows, boot=4000, seed=1))
    half = 1.96 * math.sqrt(0.3 * 0.7 / 100)
    assert e["value"] == pytest.approx(0.3) and e["lo"] < 0.3 < e["hi"]
    assert (e["lo"], e["hi"]) == pytest.approx((0.3 - half, 0.3 + half), abs=0.02)


def test_degenerate_interval_is_flagged():
    rows = [r for d in range(5) for r in scored("val", f"d{d}", 0, ["miss"] * 2)]
    e = m(metrics.report(rows, boot=100))
    assert (e["value"], e["lo"], e["hi"], e["degenerate"]) == (0.0, 0.0, 0.0, True)
    one = m(metrics.report(scored("val", "only", 0, ["exact"]), boot=100))
    assert one["value"] == 1.0 and one["lo"] is None          # one drug: no interval at all


def test_seed_determinism_and_independence_of_other_splits():
    rows = []
    for split in ("val", "test_r", "test_t"):
        for d in range(8):
            rows += scored(split, f"{split}{d}", 0, ["exact" if (d + len(split)) % 3 == 0 else "scaf"] * 8, tok=100 + d)
    a = metrics.report(rows, boot=300, seed=3)
    assert metrics.report(rows, boot=300, seed=3) == a
    b = metrics.report(rows, boot=300, seed=4)
    tok = "n_tokens_mean"                                    # continuous, so other seeds move its percentiles
    assert m(b, metric=tok)["value"] == m(a, metric=tok)["value"]
    assert (m(b, metric=tok)["lo"], m(b, metric=tok)["hi"]) != (m(a, metric=tok)["lo"], m(a, metric=tok)["hi"])
    assert metrics.report(rows, splits=["val"], boot=300, seed=3)["splits"]["val"] == a["splits"]["val"]
    assert list(a["splits"]) == ["test_t", "test_r", "val"]       # fixed order, never pooled


# ------------------------------------------------------------------------------- cluster bootstrap
def _pairs(n_clusters, tag=None, win=lambda k: k % 2 == 0, clusters=True):
    """n_clusters clusters of two drugs. In cluster k drug 'a' is exact and drug 'b' misses when win(k) is true, else the
    other way round, so every cluster holds exactly one exact drug."""
    rows = []
    for k in range(n_clusters):
        for j, g in enumerate("ab"):
            label = "exact" if (j == 0) == win(k) else "miss"
            rows += scored("val", f"c{k}{g}", 0, [label], cluster=k if clusters else None)
    return rows


def test_cluster_bootstrap_draws_all_drugs_of_a_component_together():
    # every cluster has one exact and one missing drug, so any resample of whole clusters gives exactly 0.5
    e = m(metrics.report(_pairs(8), boot=500))
    assert e["value"] == 0.5 and (e["lo"], e["hi"], e["degenerate"]) == (0.5, 0.5, True)
    res = metrics.report(_pairs(8), boot=500)["splits"]["val"]
    assert res["bootstrap"] == {"unit": "cluster", "n_units": 8} and res["n_drugs"] == 16
    # without cluster_id the drugs are resampled one by one and the interval opens up
    free = metrics.report(_pairs(8, clusters=False), boot=500)["splits"]["val"]
    assert free["bootstrap"] == {"unit": "drug", "n_units": 16}
    assert free["metrics"]["exact@1"]["lo"] < 0.5 < free["metrics"]["exact@1"]["hi"]
    assert "bootstrap resamples 8 clusters of 16 drugs" in metrics.format_report(metrics.report(_pairs(8), boot=50))
    assert "no cluster_id" in metrics.format_report(metrics.report(_pairs(8, clusters=False), boot=50))


def test_cluster_bootstrap_covers_correlated_drugs_where_the_drug_bootstrap_does_not():
    p, C, sims = 0.3, 30, 150
    cover = {True: 0, False: 0}
    for s in range(sims):
        win = np.random.default_rng(5000 + s).random(C) < p      # one outcome per cluster, shared by its two drugs
        for clustered in cover:
            rows = [r for k in range(C) for g in "ab"
                    for r in scored("val", f"c{k}{g}", 0, ["exact" if win[k] else "miss"], cluster=k if clustered else None)]
            e = m(metrics.report(rows, boot=300, seed=s))
            cover[clustered] += e["lo"] <= p <= e["hi"]
    assert cover[True] / sims >= 0.92                            # nominal 0.95
    assert cover[False] / sims <= cover[True] / sims - 0.04      # treating the 60 drugs as independent is overconfident


def test_one_drug_clusters_reproduce_the_drug_bootstrap_exactly():
    base = [r for d in range(12) for r in scored("val", f"d{d}", 0, ["exact" if d % 3 else "miss"] * 8, tok=100 + d)]
    tagged = [dict(r, cluster_id=1000 - int(r["group"][1:])) for r in base]       # distinct ids in another order
    assert metrics.report(tagged, boot=300, seed=2)["splits"]["val"]["metrics"] == \
        metrics.report(base, boot=300, seed=2)["splits"]["val"]["metrics"]


def test_cluster_ids_come_from_the_scored_rows_or_else_the_data_rows():
    rows = _pairs(8, clusters=False)
    data = [data_row("val", r["group"], 0, cluster_id=int(r["group"][1:-1])) for r in rows]
    via_data = metrics.report(rows, data=data, boot=300)
    assert via_data["splits"]["val"]["bootstrap"] == {"unit": "cluster", "n_units": 8}
    assert m(via_data)["lo"] == m(via_data)["hi"] == 0.5
    own = metrics.report(_pairs(8), boot=300)["splits"]["val"]["metrics"]
    assert via_data["splits"]["val"]["metrics"] == own
    # one drug without a cluster id: a half-labelled run is an error (it used to fall back to drugs without a word)
    part = [dict(r, cluster_id=int(r["group"][1:-1])) for r in rows[1:]] + rows[:1]
    with pytest.raises(metrics.InputError, match="1 of 16 rows lack cluster_id"):
        metrics.report(part, boot=100)
    with pytest.raises(metrics.InputError, match="two different clusters"):
        metrics.report(_pairs(2) + [dict(r, sample_idx=1, cluster_id=99) for r in _pairs(2)[:1]], boot=20)


def test_paired_bootstrap_resamples_clusters_too():
    A = _pairs(8)                                                # drug a exact in even clusters, b in odd ones
    B = _pairs(8, win=lambda k: k % 2 == 1)                      # the other way round: every drug flips
    e = pm(metrics.report(A, vs=B, boot=500))
    assert e["diff"] == 0.0 and (e["lo"], e["hi"], e["degenerate"]) == (0.0, 0.0, True)    # +1 and -1 inside each cluster
    assert (e["wins"], e["losses"], e["ties"]) == (8, 8, 0) and e["p_value"] == 1.0
    free = pm(metrics.report(_pairs(8, clusters=False), vs=_pairs(8, win=lambda k: k % 2 == 1, clusters=False), boot=500))
    assert free["lo"] < 0.0 < free["hi"]


# ---------------------------------------------------------------------------------------------- paired
def test_paired_difference_of_a_run_with_itself_is_exactly_zero():
    rows = []
    for d in range(10):
        rows += scored("val", f"d{d}", 0, (["exact"] + ["scaf"] * 3 + ["miss"] * 4)[: 8 - d % 2 * 4], tok=500 + 37 * d)
        rows += scored("val", f"d{d}", 1, ["exact" if d % 3 == 0 else "bad"] * 8, tok=900 - 11 * d, fin="length")
    res = metrics.report(rows, vs=rows, boot=500)
    p = res["splits"]["val"]["paired"]
    assert p["n_drugs"] == 10 and p["n_rows"] == 20
    for name, e in p["metrics"].items():
        assert e["diff"] == 0.0 and e["lo"] <= 0.0 <= e["hi"] and not e["excludes_zero"], name
    assert p["metrics"]["exact@1"]["a"] == p["metrics"]["exact@1"]["b"] == m(res)["value"]
    for k in metrics.TESTED:
        assert (pm(res, metric=k)["wins"], pm(res, metric=k)["losses"], pm(res, metric=k)["ties"]) == (0, 0, 10)
        assert pm(res, metric=k)["p_value"] == 1.0


def test_paired_difference_has_the_right_sign_and_flags_real_effects():
    A = [r for d in range(12) for r in _eight("val", f"d{d}", 0, "exact" if d < 10 else "miss")]
    B = [r for d in range(12) for r in _eight("val", f"d{d}", 0, "exact" if d < 4 else "miss")]
    e = metrics.report(A, vs=B, boot=2000)["splits"]["val"]["paired"]["metrics"]["exact@1"]
    assert e["diff"] == pytest.approx(6 / 12) and (e["a"], e["b"]) == pytest.approx((10 / 12, 4 / 12))
    assert 0 < e["lo"] < e["diff"] < e["hi"] and e["excludes_zero"]
    assert (e["wins"], e["losses"], e["ties"]) == (6, 0, 6) and e["p_value"] == 2 / 2 ** 6     # 6 wins: only all-plus, all-minus
    r = metrics.report(B, vs=A, boot=2000)["splits"]["val"]["paired"]["metrics"]["exact@1"]
    assert r["diff"] == pytest.approx(-0.5) and r["hi"] < 0 and r["excludes_zero"]
    assert (r["wins"], r["losses"], r["ties"]) == (0, 6, 6) and r["p_value"] == e["p_value"]  # two-sided: same p reversed
    # same outcomes, 50 more tokens per sample: exact@1 unchanged, token mean down by exactly 50
    C = [dict(r, n_tokens=r["n_tokens"] + 50) for r in A]
    pc = metrics.report(A, vs=C, boot=500)["splits"]["val"]["paired"]["metrics"]
    assert pc["exact@1"]["diff"] == 0 and pc["n_tokens_mean"]["diff"] == pytest.approx(-50)


def test_paired_uses_only_prompt_rows_both_runs_scored():
    A = scored("val", "d3", 0, ["miss"] * 8) + scored("val", "d3", 1, ["exact"] * 8) + _eight("val", "d4", 0, "exact")
    A += _eight("val", "d5", 0, "exact")                      # d5 only in A
    B = scored("val", "d3", 0, ["miss"] * 8) + _eight("val", "d4", 0, "miss") + _eight("val", "d9", 0, "exact")
    p = metrics.report(A, vs=B, boot=100)["splits"]["val"]["paired"]
    assert (p["n_drugs"], p["n_rows"]) == (2, 2)              # d3/0 and d4/0
    e = p["metrics"]["exact@1"]
    assert (e["a"], e["b"]) == (0.5, 0.0)                     # d3/1 (exact in A, absent in B) is left out of A
    none = metrics.report(A, vs=_eight("val", "zz", 0, "exact"), boot=100)["splits"]["val"]["paired"]
    assert none["metrics"] is None


def test_paired_pass_at_8_only_on_rows_where_both_runs_have_8():
    A = _eight("val", "d", 0, "exact") + _eight("val", "d", 1, "exact")
    B = _eight("val", "d", 0, "miss") + scored("val", "d", 1, ["miss"] * 4)         # B/d/1 has 4 samples
    for k in ("exact@8", "core@8", "maj@8"):
        e = metrics.report(A, vs=B, boot=100)["splits"]["val"]["paired"]["metrics"][k]
        assert (e["a"], e["b"], e["n_rows"], e["skipped_rows"]) == (1.0, 0.0, 1, 1), k
    assert "p_value" not in pm(metrics.report(A, vs=B, boot=100), metric="exact@8")      # only the tested metrics have one


# -------------------------------------------------------------------------------- sign-flip p-value
def brute_p(d):
    """Independent oracle: enumerate every sign pattern with itertools."""
    d = [x for x in d if abs(x) > 1e-9]
    t = abs(sum(d))
    hits = sum(abs(sum(s * abs(x) for s, x in zip(signs, d))) >= t - 1e-12 for signs in itertools.product((1, -1), repeat=len(d)))
    return hits / 2 ** len(d)


@pytest.mark.parametrize("d,want", [
    ([], 1.0), ([0.0, 0.0], 1.0), ([0.5], 1.0),                  # nothing to flip, or one drug: p = 2/2
    ([1, 1, 1, 1], 2 / 16), ([1, 1, 1, -1], 10 / 16),            # |sum| 4 -> 2 of 16 patterns; |sum| 2 -> 10 of 16
    ([3, -1, -1, -1], 1.0),                                      # sum 0: every pattern is at least as extreme
    ([3, 1, 1, 1], 2 / 16),                                      # T = 6 is reached only by all-plus and all-minus
    ([3, 1, 1, -1], 8 / 16),                                     # T = 4: 3 + (3 or 1 net of the small signs), both ways
])
def test_sign_flip_p_hand_values(d, want):
    p, method, n = metrics.sign_flip_p(np.array(d, float), np.random.default_rng(0))
    assert p == pytest.approx(want, abs=1e-12) and method == "exact"
    assert p == pytest.approx(brute_p(d), abs=1e-12) and n == sum(abs(x) > 1e-9 for x in d)


def test_sign_flip_p_matches_brute_force_on_random_differences():
    rng = np.random.default_rng(7)
    for n in range(1, 11):
        for _ in range(4):
            d = rng.normal(size=n) * (rng.random(n) > 0.25)      # a quarter of the drugs tied at exactly 0
            assert metrics.sign_flip_p(d, rng)[0] == pytest.approx(brute_p(d), abs=1e-12)


def test_reviewers_counterexample_five_wins_no_losses_thirty_eight_ties():
    # ties carry no information: p = 2 / 2^5 = 0.0625, not the tiny value a test that counts 43 drugs would give
    A = [r for d in range(43) for r in _eight("val", f"d{d}", 0, "exact" if d < 5 or d % 2 else "miss")]
    B = [r for d in range(43) for r in _eight("val", f"d{d}", 0, "miss" if d < 5 else "exact" if d % 2 else "miss")]
    res = metrics.report(A, vs=B, boot=200)
    for k in ("exact@1", "core@1"):
        e = pm(res, metric=k)
        assert (e["wins"], e["losses"], e["ties"], e["n_nontied"]) == (5, 0, 38, 5), k
        assert e["p_value"] == 0.0625 and e["p_method"] == "exact", k
    text = metrics.format_report(res)
    assert "sign-flip permutation test" in text and "0.0625" in text and "exact, 2^5 sign patterns" in text
    lines = text.splitlines()                      # the lines of the two metrics that were tested before v3 are unchanged
    assert "  metric            wins  losses  ties     drugs  non-tied  p (2-sided)   method" in lines
    assert "  exact@1              5       0    38        43         5       0.0625   exact, 2^5 sign patterns" in lines
    assert "  core@1               5       0    38        43         5       0.0625   exact, 2^5 sign patterns" in lines
    assert lines.index("  exact@1              5       0    38        43         5       0.0625   exact, 2^5 sign patterns") \
        == lines.index("  core@1               5       0    38        43         5       0.0625   exact, 2^5 sign patterns") + 1    # TESTED order


def test_float_noise_in_drug_means_is_a_tie_not_a_win():
    # drug 'noise': rows of 10 samples with 1, 2, 3 hits in A and 3, 2, 1 in B. The drug means are equal in theory but
    # differ by 5.6e-17 in floating point (0.20000000000000004 against 0.19999999999999998)
    def run(counts):
        return [r for i, c in enumerate(counts) for r in scored("val", "noise", i, ["exact"] * c + ["miss"] * (10 - c))]
    A = run((1, 2, 3)) + [r for d in range(3) for r in scored("val", f"w{d}", 0, ["exact"] * 10)]
    B = run((3, 2, 1)) + [r for d in range(3) for r in scored("val", f"w{d}", 0, ["miss"] * 10)]
    e = pm(metrics.report(A, vs=B, boot=100))
    assert (e["wins"], e["losses"], e["ties"], e["n_nontied"]) == (3, 0, 1, 3) and e["p_value"] == 0.25


def test_sign_flip_switches_from_exact_to_monte_carlo_after_twenty_non_tied_drugs():
    assert (metrics.EXACT_MAX, metrics.FLIPS) == (20, 100_000)            # the contract's numbers
    rng = np.random.default_rng(0)
    d = np.array([0.1] * 17 + [-0.1] * 3)                                  # 17 wins, 3 losses: 20 non-tied drugs
    p, method, n = metrics.sign_flip_p(d, rng)
    assert (method, n) == ("exact", 20)
    assert p == pytest.approx(2 * sum(math.comb(20, k) for k in range(17, 21)) / 2 ** 20, abs=1e-12)    # 0.002577
    assert metrics.sign_flip_p(np.append(d, 0.1), rng)[1:] == ("monte_carlo", 21)                      # one more drug
    assert metrics.sign_flip_p(np.append(d, [0.0] * 30), rng)[1:] == ("exact", 20)                     # ties never count


def test_sign_flip_monte_carlo_beyond_twenty_non_tied_drugs():
    rng = np.random.default_rng(3)
    p, method, n = metrics.sign_flip_p(np.full(25, 0.1), rng)             # every drug a win
    assert (method, n) == ("monte_carlo", 25) and p == pytest.approx(1 / (metrics.FLIPS + 1))   # never exactly 0
    d = np.array([0.1] * 17 + [-0.1] * 7)                                  # 17 wins, 7 losses of equal size
    exact = 2 * sum(math.comb(24, k) for k in range(17, 25)) / 2 ** 24     # two-sided binomial tail, 0.0639
    p, method, n = metrics.sign_flip_p(d, np.random.default_rng(3))
    assert method == "monte_carlo" and p == pytest.approx(exact, abs=0.004)
    assert metrics.sign_flip_p(d, np.random.default_rng(3))[0] == p        # seeded: repeatable
    assert metrics.sign_flip_p(d, np.random.default_rng(4))[0] != p
    assert metrics.sign_flip_p(np.full(25, 0.1) * np.where(np.arange(25) % 2, 1, -1), rng)[0] == 1.0    # 13 vs 12: |T| small


def test_paired_sign_flip_is_seeded_through_report_and_uses_monte_carlo_for_many_drugs():
    A = [r for d in range(30) for r in _eight("val", f"d{d}", 0, "exact" if d < 22 else "miss")]
    B = [r for d in range(30) for r in _eight("val", f"d{d}", 0, "miss" if d < 22 else "exact" if d < 24 else "miss")]
    one = metrics.report(A, vs=B, boot=100, seed=5)
    assert one == metrics.report(A, vs=B, boot=100, seed=5)
    e = pm(one)
    assert (e["wins"], e["losses"], e["ties"], e["p_method"]) == (22, 2, 6, "monte_carlo")
    exact = 2 * sum(math.comb(24, k) for k in (22, 23, 24)) / 2 ** 24      # 3.6e-5; 100,000 flips resolve it only coarsely
    assert 1 / (metrics.FLIPS + 1) <= e["p_value"] < 3e-4 and abs(e["p_value"] - exact) < 1.5e-4


# ------------------------------------------------------------------------- sign-flip over clusters
def _runs(spec):
    """spec: (group, cluster_id or None, label in A, label in B) per drug -> (A rows, B rows), one 8-sample row each."""
    A, B = [], []
    for g, c, la, lb in spec:
        extra = {"cluster_id": c} if c is not None else {}
        A += [dict(r, **extra) for r in _eight("val", g, 0, la)]
        B += [dict(r, **extra) for r in _eight("val", g, 0, lb)]
    return A, B


def test_six_concordant_wins_inside_one_cluster_are_one_unit_not_six():
    spec = [(f"w{i}", 7, "exact", "miss") for i in range(6)]
    A, B = _runs(spec)
    e = pm(metrics.report(A, vs=B, boot=100))
    assert (e["wins"], e["losses"], e["ties"]) == (6, 0, 0)                    # wins, losses, ties stay per drug
    assert (e["n_clusters"], e["n_nontied"], e["p_unit"], e["p_value"]) == (1, 1, "cluster", 1.0)     # one unit: no rejection
    # the same six wins as six unrelated drugs (no cluster_id) do reject: the overstatement the cluster unit prevents
    A0, B0 = _runs([(g, None, la, lb) for g, _, la, lb in spec])
    f = pm(metrics.report(A0, vs=B0, boot=100))
    assert (f["p_unit"], f["n_clusters"], f["n_nontied"], f["p_value"]) == ("drug", 6, 6, 2 / 2 ** 6)


def test_five_wins_over_forty_three_singleton_clusters_still_gives_0_0625():
    spec = [(f"d{d}", 1000 + d, "exact" if d < 5 or d % 2 else "miss", "miss" if d < 5 else "exact" if d % 2 else "miss")
            for d in range(43)]
    A, B = _runs(spec)
    res = metrics.report(A, vs=B, boot=100)
    for k in ("exact@1", "core@1"):
        e = pm(res, metric=k)
        assert (e["wins"], e["losses"], e["ties"], e["n_clusters"], e["n_nontied"]) == (5, 0, 38, 43, 5), k
        assert (e["p_unit"], e["p_value"], e["p_method"]) == ("cluster", 0.0625, "exact"), k


def test_cluster_totals_are_sums_of_the_drug_differences():
    # totals (2, 1, -1): T = 2 and 6 of the 8 sign patterns reach |T*| >= 2, so p = 0.75.
    # Cluster means (1, 1, -1) would give T = 1, every pattern reaches it, p = 1.0
    A, B = _runs([("a1", 1, "exact", "miss"), ("a2", 1, "exact", "miss"), ("b", 2, "exact", "miss"), ("c", 3, "miss", "exact")])
    e = pm(metrics.report(A, vs=B, boot=100))
    assert (e["wins"], e["losses"], e["ties"], e["n_clusters"], e["n_nontied"]) == (3, 1, 0, 3, 3)
    assert e["p_value"] == 0.75 == brute_p([2.0, 1.0, -1.0])


def test_a_cluster_whose_drug_differences_cancel_is_a_tied_unit():
    A, B = _runs([("a", 1, "exact", "miss"), ("b", 1, "miss", "exact"), ("c", 2, "exact", "miss")])
    e = pm(metrics.report(A, vs=B, boot=100))
    assert (e["wins"], e["losses"], e["ties"]) == (2, 1, 0)                    # drug level: nothing tied
    assert (e["n_clusters"], e["n_nontied"], e["p_value"]) == (2, 1, 1.0)      # cluster 1 sums to 0 and carries no information


def test_exact_enumeration_is_decided_by_the_number_of_clusters_not_drugs():
    A, B = _runs([(f"d{i}", i // 3, "exact", "miss") for i in range(30)])      # 30 winning drugs in 10 clusters of 3
    e = pm(metrics.report(A, vs=B, boot=100))
    assert (e["wins"], e["n_clusters"], e["n_nontied"], e["p_method"], e["p_value"]) == (30, 10, 10, "exact", 2 / 2 ** 10)
    A, B = _runs([(f"d{i}", None, "exact", "miss") for i in range(30)])        # without cluster_id: 30 units, Monte Carlo
    f = pm(metrics.report(A, vs=B, boot=100))
    assert (f["n_nontied"], f["p_method"], f["p_unit"]) == (30, "monte_carlo", "drug")


def test_cluster_flipping_matches_brute_force_on_random_clusters_and_is_seeded():
    rng = random.Random(4)
    spec, totals = [], {}
    for i in range(14):
        c = rng.randrange(6)
        la, lb = rng.choice([("exact", "miss"), ("miss", "exact"), ("exact", "exact")])
        spec.append((f"d{i}", c, la, lb))
        totals[c] = totals.get(c, 0.0) + (la == "exact") - (lb == "exact")
    A, B = _runs(spec)
    e = pm(metrics.report(A, vs=B, boot=50, seed=1))
    assert e["p_value"] == pytest.approx(brute_p(list(totals.values())), abs=1e-12)
    assert e == pm(metrics.report(A, vs=B, boot=50, seed=1))


def test_report_text_names_the_unit_of_the_sign_flip_test():
    A, B = _runs([(f"w{i}", 7, "exact", "miss") for i in range(6)])
    text = metrics.format_report(metrics.report(A, vs=B, boot=50))
    assert "summed within each cluster and the cluster totals flipped as units" in text and "clusters" in text
    A, B = _runs([(f"w{i}", None, "exact", "miss") for i in range(6)])
    text = metrics.format_report(metrics.report(A, vs=B, boot=50))
    assert "flipped one by one" in text and "no cluster_id" in text


# ------------------------------------------------------------------------------- joining and errors
def test_score_samples_joins_refs_by_id_and_keeps_sample_fields():
    data = [data_row("val", "a", 0), data_row("val", "b", 0), data_row("test_t", "c", 0)]
    smp = [{"id": r["id"], "group": r["group"], "split": r["split"], "sample_idx": 0, "model": "m", "mode": "think",
            "text": "think</think>\n" + r["ref_smiles"], "finish_reason": "stop", "n_tokens": 5} for r in data]
    out = metrics.score_samples(smp, data, fake_scorer)
    assert [r["conn_match"] for r in out] == [True, True, True]
    assert all(set(metrics.SCORE_KEYS) <= set(r) and r["model"] == "m" for r in out)
    crossed = [dict(s, id=data[(i + 1) % 3]["id"], group=data[(i + 1) % 3]["group"], split=data[(i + 1) % 3]["split"])
               for i, s in enumerate(smp)]
    assert [r["conn_match"] for r in metrics.score_samples(crossed, data, fake_scorer)] == [False] * 3


def test_score_samples_passes_optional_refs_to_the_scorer_and_copies_cluster_and_alternates():
    seen = []

    def spy(text, ref):
        seen.append(dict(ref))
        return fake_scorer(text, ref)
    data = [data_row("val", "a", 0, ref_inchikey="KEY-A", cluster_id=7, ref_alt_smiles=[ENOL]), data_row("val", "b", 0)]
    smp = [{"id": r["id"], "sample_idx": 0, "text": "</think>\nx"} for r in data]
    out = metrics.score_samples(smp, data, spy)
    assert set(seen[0]) == {"ref_smiles", "ref_conn", "ref_scaffold", "ref_inchikey", "ref_alt_smiles"}
    assert seen[0]["ref_inchikey"] == "KEY-A" and seen[0]["ref_alt_smiles"] == [ENOL]    # the scorer gets the alternates
    assert set(seen[1]) == {"ref_smiles", "ref_conn", "ref_scaffold"}               # optional keys left out, not None
    assert out[0]["cluster_id"] == 7 and out[0]["ref_alt_smiles"] == [ENOL]          # and so do the scored rows
    assert "cluster_id" not in out[1] and "ref_alt_smiles" not in out[1]


def test_unknown_id_is_a_clear_error():
    data = [data_row("val", "a", 0)]
    smp = [{"id": "ghost/0", "sample_idx": 0, "text": "</think>\nx"}, {"id": "a/0", "sample_idx": 0, "text": "</think>\nx"}]
    with pytest.raises(metrics.InputError, match=r"1 unknown id.*ghost"):
        metrics.score_samples(smp, data, fake_scorer)
    with pytest.raises(metrics.InputError, match="unknown id"):
        metrics.report(scored("val", "ghost", 0, ["miss"]), data=data)
    with pytest.raises(ValueError):                           # InputError is a ValueError
        metrics.score_samples(smp, data, fake_scorer)


def test_inconsistent_inputs_are_rejected():
    data = [data_row("val", "a", 0)]
    wrong = [{"id": "a/0", "group": "a", "split": "test_t", "sample_idx": 0, "text": "</think>\nx"}]
    with pytest.raises(metrics.InputError, match="disagree"):
        metrics.score_samples(wrong, data, fake_scorer)
    twice = scored("val", "a", 0, ["miss"]) * 2
    with pytest.raises(metrics.InputError, match="duplicate"):
        metrics.report(twice)
    with pytest.raises(metrics.InputError, match="lacks"):
        metrics.report([{"id": "x"}])
    with pytest.raises(metrics.InputError, match="lacks"):
        metrics.score_samples([{"id": "a/0", "sample_idx": 0, "text": "t"}], [{"id": "a/0", "group": "a"}], fake_scorer)
    bad = scored("val", "a", 0, ["miss"])
    with pytest.raises(metrics.InputError, match="two different groups"):
        metrics.report(bad + [dict(bad[0], group="other", sample_idx=1)])
    old = [{k: v for k, v in r.items() if k != "stereo_match"} for r in scored("val", "a", 0, ["miss"])]
    with pytest.raises(metrics.InputError, match="stereo_match"):                  # scored files from before the key existed
        metrics.report(old)


def test_scorer_result_must_have_the_contract_keys():
    data = [data_row("val", "a", 0)]
    with pytest.raises(metrics.InputError, match="lacks"):
        metrics.score_samples([{"id": "a/0", "sample_idx": 0, "text": "t"}], data, lambda t, r: {"reward": 0.0})
    no_stereo = lambda t, r: {k: v for k, v in fake_scorer(t, r).items() if k != "stereo_match"}
    with pytest.raises(metrics.InputError, match="stereo_match"):
        metrics.score_samples([{"id": "a/0", "sample_idx": 0, "text": "t"}], data, no_stereo)


def test_default_scorer_is_the_lazily_imported_reward_score(monkeypatch):
    data = [data_row("val", "a", 0)]
    smp = [{"id": "a/0", "sample_idx": 0, "text": "</think>\nsmi-a"}]
    monkeypatch.setattr(rlexp, "reward", types.SimpleNamespace(score=fake_scorer), raising=False)
    assert metrics.score_samples(smp, data)[0]["conn_match"] is True
    monkeypatch.setattr(rlexp, "reward", types.SimpleNamespace(compute_score=None), raising=False)
    with pytest.raises(RuntimeError, match="reward.score"):
        metrics.score_samples(smp, data)


# ------------------------------------------------------------------------------------ data and bar
def test_coverage_and_constant_bar_use_train_refs_and_the_scored_rows_only():
    train = [data_row("train", f"t{i}", 0, s) for i, s in enumerate(SMILES[:6])]
    centre, _ = metrics.best_constant({r["group"]: r["ref_smiles"] for r in train})
    # val v0 has the central answer as its reference, v1 and v2 do not; v2 is not in the scored run
    val = [data_row("val", "v0", 0, centre), data_row("val", "v0", 1, centre), data_row("val", "v1", 0, "CCCCCCCCO"),
           data_row("val", "v2", 0, "CCCCCCCCCO")]
    run = [r for d in ("v0/0", "v0/1", "v1/0") for r in scored("val", d.split("/")[0], int(d[-1]), ["miss"] * 8)]
    res = metrics.report(run, data=train + val, scorer=fake_scorer, boot=100)
    sp = res["splits"]["val"]
    assert sp["coverage"] == {"drugs": 3, "rows": 4} and (sp["n_drugs"], sp["n_rows"]) == (2, 3)
    bar = sp["constant"]
    assert bar["smiles"] == centre and set(bar["metrics"]) == set(metrics.BAR)
    assert bar["metrics"]["exact@1"]["value"] == pytest.approx(0.5)       # v0 yes, v1 no; v2 not scored
    assert (bar["metrics"]["exact@1"]["n_drugs"], bar["metrics"]["exact@1"]["n_rows"]) == (2, 3)
    assert "PARTIAL" in metrics.format_report(res) and "constant-answer bar" in metrics.format_report(res)
    assert "highest mean bit Tanimoto" in metrics.format_report(res) and 0 < bar["mean_bit_tanimoto_to_train"] < 1
    # the bar's lift is against every drug of the split in the data (v0, v1 and v2), so on this partial run it is not 0
    refs = {"v0": centre, "v1": "CCCCCCCCO", "v2": "CCCCCCCCCO"}
    want = float(np.mean([oracle_lift(centre, g, refs) for g in ("v0", "v1")]))
    assert abs(want) > 1e-3 and bar["metrics"]["lift"]["value"] == pytest.approx(want, abs=1e-12)
    assert bar["metrics"]["lift"]["n_drugs"] == 2
    assert (bar["metrics"]["fv_tanimoto"]["n_drugs"], bar["metrics"]["fv_tanimoto"]["n_rows"]) == (2, 3)       # eight copies per row


def test_the_constant_bar_reports_lift_zero_through_the_real_scorer_and_prints_it_as_0_000():
    train = real_data("train", {f"t{i}": (s, 1) for i, s in enumerate(SMILES[:6])})
    val = real_data("val", VAL)
    res = metrics.report(run_real(val, lambda r, j: r["ref_smiles"]), data=train + val, boot=20)      # the real reward.score
    bar = res["splits"]["val"]["constant"]
    assert abs(bar["metrics"]["lift"]["value"]) < 1e-12 and bar["metrics"]["lift"]["n_drugs"] == 4
    assert m(res, "val", "lift")["value"] > 0.2                                    # answers that depend on the profile beat it
    block = metrics.format_report(res).split("constant-answer bar")[1]
    assert [ln.split()[1] for ln in block.splitlines() if ln.startswith("  lift ")] == ["0.000"]


def test_bar_is_skipped_with_a_note_when_it_cannot_be_computed(monkeypatch):
    run = scored("val", "v0", 0, ["miss"])
    monkeypatch.setattr(rlexp, "reward", types.SimpleNamespace(), raising=False)         # no reward.score at all
    res = metrics.report(run, data=[data_row("train", "t0", 0, SMILES[0]), data_row("train", "t1", 0, SMILES[1]),
                                    data_row("val", "v0", 0, SMILES[2])], boot=50)
    assert "constant" not in res["splits"]["val"] and "skipped" in res["notes"][0] and "reward.score" in res["notes"][0]
    res = metrics.report(run, data=[data_row("val", "v0", 0, SMILES[2])], scorer=fake_scorer, boot=50)
    assert "constant" not in res["splits"]["val"] and "train" in res["notes"][0]


# ----------------------------------------------------------------------- v3: finished, with the real reward.score
VAL = {"a": (SMILES[0], 3), "b": (SMILES[1], 1), "c": (SMILES[2], 2), "d": (SMILES[3], 1)}      # drug: (reference, prompt rows)
TEST_T = {"x": (SMILES[4], 2), "y": (SMILES[5], 1), "z": (SMILES[7], 2)}
CONSTANTS = [SMILES[1], SMILES[3], SMILES[6], "CCCCCCCC(=O)N", "c1ccccc1CCCCCCC(=O)O", "CC(C)(C)c1ccc(cc1)C(=O)O"]


def real_data(split, drugs):
    """Data rows with real references: drugs = {group: (SMILES, number of prompt rows)}."""
    return [data_row(split, g, i, smi) for g, (smi, n) in drugs.items() for i in range(n)]


def sample(row, j, text, fin="stop"):
    return {"id": row["id"], "sample_idx": j, "text": text, "finish_reason": fin, "n_tokens": 5}


def run_real(data, answer_of, n=1, fin="stop"):
    """score_samples through the real reward.score: n samples of every data row, each answering answer_of(row, j)."""
    return metrics.score_samples([sample(r, j, f"t</think>\n{answer_of(r, j)}", fin) for r in data for j in range(n)], data)


def drug_means(rows, key="lift"):
    """drug -> mean over its prompt rows of the mean over each row's samples: how the metrics aggregate."""
    by = {}
    for r in rows:
        by.setdefault(r["group"], {}).setdefault(r["id"], []).append(r[key])
    return {g: float(np.mean([np.mean(v) for v in ids.values()])) for g, ids in by.items()}


def oracle_lift(answer, own, refs, similarity=chem.tanimoto):
    """The definition of lift through the public chem.tanimoto, one pair at a time. refs: {drug: SMILES} of the whole split."""
    a = chem.standardize(answer)
    t = {g: similarity(a, chem.standardize(s)) for g, s in refs.items()}
    return t[own] - float(np.mean([v for g, v in t.items() if g != own]))


def test_score_samples_hands_finished_to_a_scorer_that_accepts_it():
    named, keyword_only, anything = [], [], []

    def by_name(text, ref, finished=True):
        named.append(finished)
        return fake_scorer(text, ref)

    def by_keyword(text, ref, *, finished):
        keyword_only.append(finished)
        return fake_scorer(text, ref)

    def by_kwargs(text, ref, **kw):
        anything.append(kw)
        return fake_scorer(text, ref)

    data = [data_row("val", "a", 0)]
    smp = [{"id": "a/0", "sample_idx": i, "text": "</think>\nx", **({"finish_reason": f} if f else {})}
           for i, f in enumerate(["stop", "length", None, "abort", "stop"])]
    expect = [True, False, True, False, True]                 # "stop" and no finish_reason finished; "length" and "abort" are failures
    for scorer in (by_name, by_keyword, by_kwargs):
        metrics.score_samples(smp, data, scorer)
    assert named == keyword_only == expect and anything == [{"finished": f} for f in expect]


def test_a_scorer_without_finished_is_called_as_before_and_the_scored_row_keeps_finish_reason():
    calls = []

    class Plain:
        def __call__(self, text, ref):
            calls.append(text)
            return fake_scorer(text, ref)

    class Opaque(Plain):
        __signature__ = "not a signature"               # inspect.signature raises TypeError: treated as not accepting it
    data = [data_row("val", "a", 0)]
    smp = [{"id": "a/0", "sample_idx": i, "text": "</think>\nx", "finish_reason": f} for i, f in enumerate(["stop", "length"])]
    for scorer in (lambda t, r: fake_scorer(t, r), fake_scorer, Plain(), functools.partial(fake_scorer), Opaque()):
        out = metrics.score_samples(smp, data, scorer)
        assert [r["finish_reason"] for r in out] == ["stop", "length"]
    assert len(calls) == 4                                  # Plain() and Opaque(), two samples each


def test_a_type_error_inside_a_scorer_is_not_swallowed_and_nothing_is_scored_twice():
    calls = []

    def broken(text, ref, finished=True):
        calls.append(finished)
        raise TypeError("inside the scorer")
    with pytest.raises(TypeError, match="inside the scorer"):
        metrics.score_samples([{"id": "a/0", "sample_idx": 0, "text": "t"}], [data_row("val", "a", 0)], broken)
    assert calls == [True]


def test_a_length_terminated_sample_is_a_failure_through_the_real_scorer():
    data = real_data("val", {"a": (SMILES[0], 1), "b": (SMILES[1], 1)})
    plain = sample(data[0], 2, f"t</think>\n{SMILES[0]}")
    del plain["finish_reason"]
    ok, cut, nofr = metrics.score_samples([sample(data[0], 0, f"t</think>\n{SMILES[0]}"),
                                           sample(data[0], 1, f"t</think>\n{SMILES[0]}", "length"), plain], data)
    assert ok["valid"] and ok["conn_match"] and ok["reward"] == 1.0 and ok["lift"] > 0 and ok["finish_reason"] == "stop"
    assert cut["finish_reason"] == "length" and cut["closed"] is True and cut["answer"] == SMILES[0]    # diagnostics survive
    assert (cut["valid"], cut["conn_match"], cut["scaffold_match"], cut["stereo_match"]) == (False,) * 4
    assert (cut["tanimoto"], cut["reward"], cut["lift"]) == (0.0, 0.0, 0.0)
    assert nofr["valid"] and nofr["reward"] == 1.0 and nofr["lift"] == ok["lift"] and "finish_reason" not in nofr
    res = metrics.report([ok, cut], boot=20)
    for k, want in (("valid_closed", 0.5), ("exact@1", 0.5), ("mean_reward", 0.5), ("truncation", 0.5),
                    ("tanimoto_valid", 1.0)):
        assert m(res, metric=k)["value"] == want, k
    assert m(res, metric="lift")["value"] == pytest.approx(ok["lift"] / 2)


def test_the_scorers_verdict_gates_lift_even_when_the_answer_is_a_real_molecule():
    data = real_data("val", {"a": (SMILES[0], 1), "b": (SMILES[1], 1)})
    verdicts = iter([OUT["bad"], OUT["open"], OUT["miss"], {**OUT["miss"], "closed": False}])
    rows = metrics.score_samples([sample(data[0], i, f"t</think>\n{SMILES[0]}") for i in range(4)], data,
                                 lambda text, ref: {**next(verdicts), "answer": SMILES[0]})
    assert [r["lift"] > 0 for r in rows] == [False, False, True, False]         # invalid, unclosed, ok, valid but unclosed
    # and a cut sample has lift 0 even when a scorer that does not know about finished called it valid
    cut = metrics.score_samples([sample(data[0], 0, "t", "length")], data, lambda text, ref: {**OUT["miss"], "answer": SMILES[0]})
    assert cut[0]["valid"] and cut[0]["lift"] == 0.0


# ----------------------------------------------------------------------------------------- v3: lift
def test_lift_of_any_constant_answer_is_zero_on_every_split():
    data = real_data("val", VAL) + real_data("test_t", TEST_T)
    for c in CONSTANTS:
        rows = run_real(data, lambda r, j: c, n=2)
        assert all(r["valid"] and r["answer"] == c for r in rows)                 # premise: a real, scored answer
        means, res = drug_means(rows), metrics.report(rows, boot=20)
        for split, drugs in (("val", VAL), ("test_t", TEST_T)):
            assert sum(means[g] for g in drugs) == pytest.approx(0.0, abs=1e-12), (c, split)
            assert max(abs(means[g]) for g in drugs) > 1e-3, (c, split)           # not vacuous: the drugs differ, the mean does not
            assert m(res, split, "lift")["value"] == pytest.approx(0.0, abs=1e-12), (c, split)


def test_lift_of_the_drugs_own_reference_is_positive_and_follows_the_definition():
    data = real_data("val", VAL) + real_data("test_t", TEST_T)
    rows = run_real(data, lambda r, j: r["ref_smiles"], n=2)
    res = metrics.report(rows, boot=20)
    for split, drugs in (("val", VAL), ("test_t", TEST_T)):
        refs = {g: s for g, (s, _) in drugs.items()}
        want = {g: oracle_lift(s, g, refs) for g, s in refs.items()}
        for r in (r for r in rows if r["split"] == split):
            assert r["conn_match"] and r["lift"] > 0.2
            assert r["lift"] == pytest.approx(want[r["group"]], abs=1e-12)
        assert m(res, split, "lift")["value"] == pytest.approx(np.mean(list(want.values())), abs=1e-12)
        assert m(res, split, "lift")["value"] > 0.2


def test_the_others_mean_counts_each_drug_once_and_uses_every_drug_of_the_split_in_the_data():
    refs = {g: s for g, (s, _) in VAL.items()}
    data = real_data("val", VAL)                                  # a has three rows, c two, b and d one
    wrong = SMILES[6]                                             # an answer that is nobody's reference
    rows = metrics.score_samples([sample(r, 0, f"t</think>\n{wrong}") for r in data if r["group"] in ("b", "d")], data)
    for r in rows:
        assert r["lift"] == pytest.approx(oracle_lift(wrong, r["group"], refs), abs=1e-12)    # against all four drugs
    # the same answer to every row of the split: the drugs with more rows must not weigh more in the others' mean
    allrows = run_real(data, lambda r, j: wrong)
    lifts = {r["group"]: r["lift"] for r in allrows}
    assert lifts == pytest.approx({g: oracle_lift(wrong, g, refs) for g in refs}, abs=1e-12)
    assert sum(lifts.values()) == pytest.approx(0.0, abs=1e-12)


def test_lift_is_the_count_vector_tanimoto_of_the_reward_not_a_lookalike():
    # a 16- and a 108-heavy-atom phenyl fatty acid: the bit Tanimoto calls them identical, the count vector does not
    short, long_ = "c1ccccc1" + "C" * 7 + "C(=O)O", "c1ccccc1" + "C" * 99 + "C(=O)O"
    refs = {"p": short, "q": long_, "r": SMILES[3], "s": SMILES[0]}
    data = real_data("val", {g: (s, 1) for g, s in refs.items()})
    for answer in (short, SMILES[2]):
        rows = run_real(data, lambda r, j: answer)
        for r in rows:
            assert r["lift"] == pytest.approx(oracle_lift(answer, r["group"], refs), abs=1e-12)
    lifts = {r["group"]: r["lift"] for r in run_real(data, lambda r, j: short)}
    assert abs(lifts["p"] - oracle_lift(short, "p", refs, chem.tanimoto_bits)) > 0.05      # premise: the lookalike would show


def test_invalid_unfinished_and_unclosed_samples_have_lift_zero_and_keep_their_drug():
    data = real_data("val", {"a": (SMILES[0], 1), "b": (SMILES[1], 1), "c": (SMILES[2], 1)})
    ra, rb, rc = data
    refs = {"a": SMILES[0], "b": SMILES[1], "c": SMILES[2]}
    smp = [sample(ra, 0, f"t</think>\n{SMILES[0]}"),                   # its own reference, valid and finished
           sample(ra, 1, "t</think>\nno idea at all"),                  # no molecule
           sample(ra, 2, f"t</think>\n{SMILES[0]}", "length"),          # the right molecule, cut by the length cap
           sample(ra, 3, f"t {SMILES[0]}"),                             # never closed the think block
           sample(rb, 0, "t</think>\nnope"), sample(rc, 0, "t</think>\nnope")]
    rows = metrics.score_samples(smp, data)
    a = oracle_lift(SMILES[0], "a", refs)
    assert a > 0 and rows[0]["lift"] == pytest.approx(a, abs=1e-12) and [r["lift"] for r in rows[1:]] == [0.0] * 5
    assert [r["valid"] for r in rows] == [True, False, False, False, False, False]
    res = metrics.report(rows, boot=20)
    e = m(res, metric="lift")
    assert e["n_drugs"] == 3 and e["value"] == pytest.approx(a / 4 / 3, abs=1e-12)     # b and c stay in, at 0
    t = m(res, metric="tanimoto_valid")
    assert (t["n_drugs"], t["value"]) == (1, 1.0)                                       # the valid-only metric drops them


def test_a_split_with_a_single_drug_has_lift_zero_and_a_note():
    data = real_data("val", {"a": (SMILES[0], 2)}) + real_data("test_t", {"x": (SMILES[4], 1), "y": (SMILES[5], 1)})
    rows = run_real(data, lambda r, j: r["ref_smiles"])
    assert all(r["valid"] for r in rows)
    assert {r["lift"] for r in rows if r["split"] == "val"} == {0.0}              # valid answers, but nobody to compare with
    assert all(r["lift"] > 0 for r in rows if r["split"] == "test_t")
    res = metrics.report(rows, data=data, boot=20)
    assert m(res, "val", "lift")["value"] == 0.0 and m(res, "test_t", "lift")["value"] > 0
    assert [n for n in res["notes"] if "single drug" in n] == \
        ["split val has a single drug in the data: lift is 0.0 by definition (no other drug to compare with)"]


def test_rows_of_one_drug_must_share_the_reference_and_every_reference_must_standardize():
    smp = [sample(data_row("val", "a", 0, SMILES[0]), 0, f"t</think>\n{SMILES[0]}")]
    views = [data_row("val", "a", 0, SMILES[0]), data_row("val", "a", 1, SMILES[1]), data_row("val", "b", 0, SMILES[2])]
    with pytest.raises(metrics.InputError, match=r"1 drug\(s\) have rows with different ref_smiles.*val.*\['a'\]"):
        metrics.score_samples(smp, views)
    agree = [data_row("val", "a", 0, SMILES[0]), data_row("val", "a", 1, SMILES[0]), data_row("val", "b", 0, SMILES[2])]
    assert metrics.score_samples(smp, agree)[0]["lift"] > 0                      # the same reference twice is one reference
    # a split that holds no sample is not inspected
    other = agree + [data_row("test_t", "z", 0, SMILES[2]), data_row("test_t", "z", 1, SMILES[3])]
    assert metrics.score_samples(smp, other)[0]["lift"] > 0
    broken = [data_row("val", "a", 0, SMILES[0]), data_row("val", "b", 0, "not a smiles")]
    with pytest.raises(metrics.InputError, match="1 reference.*split val do not standardize"):
        metrics.score_samples(smp, broken)
    quiet = [sample(broken[0], 0, "t</think>\nnope")]                            # nothing to compare: nothing needed
    assert metrics.score_samples(quiet, broken)[0]["lift"] == 0.0


def test_each_molecule_is_fingerprinted_once_not_once_per_pair(monkeypatch):
    data = real_data("val", VAL) + real_data("test_t", TEST_T)                  # 7 + 5 prompt rows, 4 + 3 drugs
    generator, built = metrics._generator(), []

    class Counting:
        def GetCountFingerprint(self, mol):
            built.append(1)
            return generator.GetCountFingerprint(mol)
    metrics._fp_of.cache_clear()
    monkeypatch.setattr(metrics, "_generator", lambda: Counting())
    try:
        rows = run_real(data, lambda r, j: CONSTANTS[3 + j % 3], n=4)           # 48 samples, three distinct answers
    finally:
        metrics._fp_of.cache_clear()
    assert len(rows) == 48 and all(r["valid"] for r in rows)
    assert len(built) == 3 + 7                                                  # the answers once, the references once


# ------------------------------------------------------------------ v3: old scored files, first-valid, tanimoto_valid
def test_scored_rows_from_before_v3_stay_readable_and_are_flagged():
    new = _eight("val", "a", 0, "exact") + _eight("val", "b", 0, "miss")
    old = [{k: v for k, v in r.items() if k != "lift"} for r in new]
    res = metrics.report(old, boot=50)
    for k in ("lift", "fv_lift"):
        e = m(res, metric=k)
        assert e["value"] is None and e["n_drugs"] == 0 and e["lo"] is None, k
    for k in metrics.METRICS:
        if k not in ("lift", "fv_lift"):
            assert m(res, metric=k) == m(metrics.report(new, boot=50), metric=k), k      # everything else is computed as before
    assert [n for n in res["notes"] if "scored before v3" in n] == \
        ["A: 16 of 16 scored rows have no lift (scored before v3): lift is n/a for them and samples cut by the length cap "
         "may still count as valid; re-run `score` for the v3 rules"]
    assert not metrics.report(new, boot=50)["notes"]
    text = metrics.format_report(res)
    assert [ln for ln in text.splitlines() if ln.startswith("  lift ")][0].split()[1] == "n/a"
    paired = metrics.report(new, vs=old, boot=50)["splits"]["val"]["paired"]["metrics"]       # one side old: no lift difference
    assert paired["lift"]["diff"] is None and paired["lift"]["p_value"] is None and paired["exact@1"]["diff"] == 0.0     # no drug to test
    assert "no drug with a value in both runs" in metrics.format_report(metrics.report(new, vs=old, boot=50))
    json.dumps(metrics.report(new, vs=old, boot=50), allow_nan=False)


def test_first_valid_policy_picks_the_first_valid_finished_sample_of_the_first_eight():
    z = scored("val", "z", 0, ["flat"] + ["bad"] * 7)                                      # first valid at index 0
    p = scored("val", "p", 0, ["miss"] + ["exact"] * 7)
    p[0]["finish_reason"] = "length"                    # a valid label that the cap cut: the first usable sample is index 1
    q = scored("val", "q", 0, ["bad", "open", "bad", "scaf", "flat", "exact", "bad", "bad"])     # first valid at 3, last at 5
    r = scored("val", "r", 0, ["bad", "open", "bad", "open", "bad", "bad", "open", "bad", "exact", "exact"])   # valid only after 8
    s = scored("val", "s", 0, ["exact"] * 7)            # seven samples: skipped
    t0, t1 = scored("val", "t", 0, ["bad"] * 7 + ["scaf"]), scored("val", "t", 1, ["bad", "flat"] + ["bad"] * 6)
    rows = z + p + q + r + s + t0 + t1
    # per prompt row: reward, tanimoto, lift, found, samples drawn up to the first usable one (8 when none), from OUT and LIFT
    per_drug = {"z": [(1.0, 0.9, 0.8, 1, 1)], "p": [(1.0, 1.0, 0.9, 1, 2)], "q": [(0.65, 0.5, 0.4, 1, 4)],
                "r": [(0.0, 0.0, 0.0, 0, 8)], "t": [(0.65, 0.5, 0.4, 1, 8), (1.0, 0.9, 0.8, 1, 2)]}
    want = {k: np.mean([np.mean([row[i] for row in drug]) for drug in per_drug.values()])
            for i, k in enumerate(("fv_reward", "fv_tanimoto", "fv_lift", "fv_rate", "fv_samples"))}
    assert want == pytest.approx({"fv_reward": 0.695, "fv_tanimoto": 0.62, "fv_lift": 0.54, "fv_rate": 0.8, "fv_samples": 4.0})
    for order in (rows, rows[::-1], random.Random(2).sample(rows, len(rows))):
        res = metrics.report(order, boot=50)
        for k, v in want.items():
            e = m(res, metric=k)
            assert e["value"] == pytest.approx(v), k
            assert (e["n_drugs"], e["n_rows"], e["skipped_rows"]) == (5, 6, 1), k       # s has no usable row: counted like @8
        assert (m(res, metric="exact@8")["n_drugs"], m(res, metric="exact@8")["skipped_rows"]) == (5, 1)


def test_first_valid_counts_the_draws_and_charges_all_eight_for_giving_up():
    rows = scored("val", "a", 0, ["exact"] + ["bad"] * 7) + scored("val", "b", 0, ["bad"] * 7 + ["miss"])
    res = metrics.report(rows, boot=20)
    assert m(res, metric="fv_samples")["value"] == (1 + 8) / 2 and m(res, metric="fv_rate")["value"] == 1.0
    none = metrics.report(_eight("val", "a", 0, "bad") + _eight("val", "b", 0, "open"), boot=20)
    assert (m(none, metric="fv_rate")["value"], m(none, metric="fv_samples")["value"], m(none, metric="fv_reward")["value"]) \
        == (0.0, 8.0, 0.0)                                                   # giving up costs all eight draws


def test_tanimoto_valid_is_the_mean_over_valid_finished_samples_and_drops_drugs_without_one():
    A0 = scored("val", "A", 0, ["exact", "bad", "miss", "bad"])             # valid mean (1.0 + 0.1) / 2; all-attempts mean 0.275
    A1 = scored("val", "A", 1, ["bad", "open"])                            # no valid sample: no value, not a zero
    B = scored("val", "B", 0, ["scaf", "miss", "bad", "bad"])               # (0.5 + 0.1) / 2
    C = scored("val", "C", 0, ["bad", "open"])                             # no valid sample in any row: the drug drops out
    res = metrics.report(A0 + A1 + B + C, boot=50)
    t = m(res, metric="tanimoto_valid")
    assert t["value"] == pytest.approx((0.55 + 0.3) / 2) and (t["n_drugs"], t["n_rows"]) == (2, 4)
    assert m(res, metric="mean_tanimoto")["value"] == pytest.approx(((1.1 / 4 + 0) / 2 + 0.6 / 4 + 0) / 3)   # invalid = 0, all drugs
    assert m(res, metric="mean_tanimoto")["n_drugs"] == 3 and m(res, metric="valid_closed")["n_drugs"] == 3
    # a valid-looking sample that the cap cut does not count either
    B[3].update(OUT["exact"], finish_reason="length")
    assert m(metrics.report(A0 + A1 + B + C, boot=50), metric="tanimoto_valid")["value"] == pytest.approx((0.55 + 0.3) / 2)


# ------------------------------------------------------------------------------------ v3: paired lift and the note
def test_paired_lift_and_mean_tanimoto_give_the_hand_computed_wins_losses_ties_and_p():
    lift_a, tan_a, tan_b = [0.10, 0.08, 0.05, 0.02, 0.00, -0.03], [0.30, 0.30, 0.30, 0.10, 0.20, 0.20], \
        [0.25, 0.25, 0.25, 0.15, 0.20, 0.20]
    A = [r for d in range(6) for r in valued("val", f"d{d}", 0, tan_a[d], lift_a[d])]
    B = [r for d in range(6) for r in valued("val", f"d{d}", 0, tan_b[d], 0.0)]
    res = metrics.report(A, vs=B, boot=300)
    lift, tan = pm(res, metric="lift"), pm(res, metric="mean_tanimoto")
    # lift A-B = +.10 +.08 +.05 +.02 0 -.03: 4 wins, 1 loss, 1 tie; T = .22, and 6 of the 32 sign patterns reach it
    assert (lift["wins"], lift["losses"], lift["ties"], lift["n_nontied"], lift["p_unit"], lift["p_method"]) == \
        (4, 1, 1, 5, "drug", "exact")
    assert lift["p_value"] == 6 / 32 == pytest.approx(brute_p([0.10, 0.08, 0.05, 0.02, 0.0, -0.03]), abs=1e-12)
    assert lift["diff"] == pytest.approx(0.22 / 6) and lift["a"] == pytest.approx(0.22 / 6) and lift["b"] == 0.0
    # mean Tanimoto A-B = +.05 +.05 +.05 -.05 0 0: 3 wins, 1 loss, 2 ties; T = .10, and 10 of the 16 patterns reach it
    assert (tan["wins"], tan["losses"], tan["ties"], tan["n_nontied"]) == (3, 1, 2, 4)
    assert tan["p_value"] == 10 / 16 and tan["diff"] == pytest.approx(0.10 / 6)
    for k in ("exact@1", "core@1"):                                              # unchanged metrics: nothing differs
        e = pm(res, metric=k)
        assert (e["wins"], e["losses"], e["ties"], e["p_value"]) == (0, 0, 6, 1.0), k
    assert {k for k, e in res["splits"]["val"]["paired"]["metrics"].items() if "p_value" in e} == set(metrics.TESTED)
    assert all("wins" not in e for k, e in res["splits"]["val"]["paired"]["metrics"].items() if k not in metrics.TESTED)
    text = metrics.format_report(res)
    assert "  lift                 4       1     1         6         5       0.1875   exact, 2^5 sign patterns" in text
    assert "  mean_tanimoto        3       1     2         6         4       0.6250   exact, 2^4 sign patterns" in text


def test_paired_lift_flips_the_cluster_totals():
    # clusters {a1: +.10, a2: +.08}, {b: +.05}, {c: -.03} have totals .18, .05, -.03: T = .20, and 4 of the 8 patterns reach it
    spec = [("a1", 1, 0.10), ("a2", 1, 0.08), ("b", 2, 0.05), ("c", 3, -0.03)]
    A = [r for g, c, v in spec for r in valued("val", g, 0, 0.2, v, cluster=c)]
    B = [r for g, c, v in spec for r in valued("val", g, 0, 0.2, 0.0, cluster=c)]
    res = metrics.report(A, vs=B, boot=50)
    e = pm(res, metric="lift")
    assert (e["wins"], e["losses"], e["ties"], e["n_clusters"], e["n_nontied"], e["p_unit"]) == (3, 1, 0, 3, 3, "cluster")
    assert e["p_value"] == 0.5 == pytest.approx(brute_p([0.18, 0.05, -0.03]), abs=1e-12)
    t = pm(res, metric="mean_tanimoto")
    assert (t["wins"], t["losses"], t["ties"], t["p_value"]) == (0, 0, 4, 1.0)          # equal Tanimoto: all ties, p = 1


def test_paired_lift_uses_monte_carlo_flips_above_twenty_non_tied_units():
    A = [r for d in range(25) for r in valued("val", f"d{d}", 0, 0.2, 0.01)]
    B = [r for d in range(25) for r in valued("val", f"d{d}", 0, 0.2, 0.0)]
    e = pm(metrics.report(A, vs=B, boot=50), metric="lift")
    assert (e["wins"], e["n_nontied"], e["p_method"]) == (25, 25, "monte_carlo")
    assert e["p_value"] == pytest.approx(1 / (metrics.FLIPS + 1))


def test_the_sign_flip_assumptions_are_stated_once_per_report_and_only_with_a_test():
    def runs(split):
        return ([r for d in range(4) for r in valued(split, f"{split}{d}", 0, 0.2 + 0.01 * d, 0.01 * d)],
                [r for d in range(4) for r in valued(split, f"{split}{d}", 0, 0.2, 0.0)])
    (a1, b1), (a2, b2) = runs("val"), runs("test_t")
    text = metrics.format_report(metrics.report(a1 + a2, vs=b1 + b2, boot=50))
    assert text.count("test_t   drugs") == 1 and text.count("sign-flip permutation test, the decision statistic") == 2
    assert text.count("Sign-flip test assumptions") == 1 and text.count("exchangeable") == 1      # two splits, one note
    note = text[text.index("Sign-flip test assumptions"):]
    for phrase in ("components", "exchangeable units", "symmetric about zero", "not assumption-free"):
        assert phrase in note, phrase
    assert "assumption" not in metrics.format_report(metrics.report(a1 + a2, boot=50))     # no pairing, no test, no note


def test_the_studentized_line_is_the_paired_difference_over_its_component_bootstrap_se():
    d = np.random.default_rng(3).normal(0.02, 0.05, 30)
    A = [r for k in range(30) for r in valued("val", f"d{k}", 0, 0.2, float(d[k]), cluster=k)]
    B = [r for k in range(30) for r in valued("val", f"d{k}", 0, 0.2, 0.0, cluster=k)]
    res = metrics.report(A, vs=B, boot=4000, seed=1)
    e = pm(res, metric="lift")
    se = d.std() / math.sqrt(30)                       # the plug-in bootstrap standard error of a mean of 30 components
    assert e["diff"] == pytest.approx(d.mean()) and e["boot_se"] == pytest.approx(se, rel=0.05)
    assert e["studentized"] == pytest.approx(e["diff"] / e["boot_se"]) and e["studentized"] == pytest.approx(d.mean() / se, rel=0.06)
    assert pm(res, metric="mean_tanimoto")["studentized"] is None           # no difference, no spread: undefined, not 0 / 0
    assert "boot_se" not in m(res, metric="lift") and "studentized" not in pm(res, metric="exact@8")
    text = metrics.format_report(res)
    line = [ln for ln in text.splitlines() if "fv_tanimoto n/a" in ln]            # the endpoint first: no difference, no spread
    assert len(line) == 1 and f"lift {e['studentized']:+.2f}   fv_lift {e['studentized']:+.2f}" in line[0]
    assert "mean_tanimoto n/a   lift" in line[0] and "core@1 n/a   exact@1 n/a" in line[0]
    assert "paired mean difference / its bootstrap SE over 4000 resampled clusters" in text
    # no resamples, or a single component: no standard error
    for res in (metrics.report(A, vs=B, boot=0), metrics.report(valued("val", "x", 0, 0.2, 0.5, cluster=1),
                                                                    vs=valued("val", "x", 0, 0.2, 0.0, cluster=1), boot=50)):
        assert pm(res, metric="lift")["studentized"] is None and pm(res, metric="lift")["boot_se"] is None
        assert "lift n/a" in metrics.format_report(res)


def test_float_noise_around_zero_prints_as_zero_and_a_report_json_from_before_v3_still_formats():
    assert metrics._v(-3e-17, "lift") == "0.000" and metrics._v(3e-17, "lift", True) == "+0.000"      # never -0.000
    assert metrics._v(-0.0004, "lift") == "-0.000" and metrics._v(-0.0006, "lift") == "-0.001"        # real values keep their sign
    res = metrics.report(_eight("val", "a", 0, "exact") + _eight("val", "b", 0, "miss"),
                         vs=_eight("val", "a", 0, "miss") + _eight("val", "b", 0, "miss"), boot=50)
    for e in res["splits"]["val"]["paired"]["metrics"].values():
        e.pop("studentized", None)                                          # the paired entries of a v2 report.json
        e.pop("boot_se", None)
    assert "studentized component sensitivity" in metrics.format_report(res)


# ------------------------------------------------------------ round 2: the endpoint and the conditions of the lift zero
def test_a_fixed_molecule_has_lift_above_zero_when_only_one_drug_finishes():
    # the zero of a constant answer needs completion that does not depend on the drug (contracts.md): here only drug a finishes,
    # so the mean is its lift over four, not the cancelling sum. The limitation is pinned, not hidden.
    data = real_data("val", VAL)
    refs = {g: s for g, (s, _) in VAL.items()}
    fixed = SMILES[0]                                                    # a's reference
    rows = metrics.score_samples([sample(r, 0, f"t</think>\n{fixed}", "stop" if r["group"] == "a" else "length") for r in data], data)
    assert {r["group"] for r in rows if r["valid"]} == {"a"}
    want = oracle_lift(fixed, "a", refs) / len(refs)                    # the other three contribute 0, not their negative lifts
    assert want > 0.1
    assert m(metrics.report(rows, boot=20), "val", "lift")["value"] == pytest.approx(want, abs=1e-12)
    everyone = run_real(data, lambda r, j: fixed)                       # the same molecule finished by every drug cancels again
    assert m(metrics.report(everyone, boot=20), "val", "lift")["value"] == pytest.approx(0.0, abs=1e-12)


def test_fv_lift_of_a_constant_is_zero_when_every_row_has_a_valid_answer_and_not_otherwise():
    data = real_data("val", VAL) + real_data("test_t", TEST_T)
    for c in CONSTANTS:
        res = metrics.report(run_real(data, lambda r, j: c, n=8), boot=20)
        for split in ("val", "test_t"):
            assert m(res, split, "fv_lift")["value"] == pytest.approx(0.0, abs=1e-12), (c, split)
            assert m(res, split, "fv_rate")["value"] == 1.0 and m(res, split, "fv_samples")["value"] == 1.0
            # the endpoint of a constant that always parses is its own mean Tanimoto to the references
            assert m(res, split, "fv_tanimoto")["value"] == pytest.approx(m(res, split, "mean_tanimoto")["value"], abs=1e-12)
    refs = {g: s for g, (s, _) in VAL.items()}
    val = real_data("val", VAL)
    cut = metrics.score_samples([sample(r, j, f"t</think>\n{SMILES[0]}", "length" if r["group"] == "a" else "stop")
                                 for r in val for j in range(8)], val)       # drug a never finishes: its rows give up
    got = m(metrics.report(cut, boot=20), "val", "fv_lift")
    assert got["value"] == pytest.approx(-oracle_lift(SMILES[0], "a", refs) / 4, abs=1e-12)      # no longer 0
    assert m(metrics.report(cut, boot=20), "val", "fv_samples")["value"] == pytest.approx((8 + 1 + 1 + 1) / 4)


def test_fv_tanimoto_is_a_first_class_endpoint_in_the_table_the_json_the_paired_block_and_the_bar():
    train = real_data("train", {f"t{i}": (s, 1) for i, s in enumerate(SMILES[:6])})
    val = real_data("val", VAL)
    A = run_real(val, lambda r, j: r["ref_smiles"], n=8)                # its own reference at the first draw
    B = run_real(val, lambda r, j: CONSTANTS[3], n=8)
    res = metrics.report(A, vs=B, data=train + val, boot=100)
    sp = res["splits"]["val"]
    assert sp["metrics"]["fv_tanimoto"]["value"] == 1.0 and sp["metrics"]["fv_samples"]["value"] == 1.0
    bar = sp["constant"]["metrics"]
    assert set(bar) == set(metrics.BAR) and "fv_tanimoto" in bar
    assert bar["fv_tanimoto"]["value"] == pytest.approx(bar["mean_tanimoto"]["value"], abs=1e-12)    # always valid: its Tanimoto
    assert (bar["fv_tanimoto"]["n_drugs"], bar["fv_tanimoto"]["n_rows"]) == (4, 7)
    e = sp["paired"]["metrics"]["fv_tanimoto"]
    assert {"a", "b", "diff", "lo", "hi", "wins", "losses", "ties", "p_value", "boot_se", "studentized"} <= set(e)
    assert (e["wins"], e["losses"], e["ties"]) == (4, 0, 0) and e["a"] == 1.0
    text = metrics.format_report(res)
    lines = text.splitlines()
    table = lines[lines.index("  metric            wins  losses  ties     drugs  non-tied  p (2-sided)   method") + 1:]
    assert table[0].split()[0] == "fv_tanimoto"                                       # the confirmatory endpoint heads the test table
    assert [ln.split()[0] for ln in table[:6]] == list(metrics.TESTED)
    block = text.split("constant-answer bar")[1]
    assert [ln.split()[0] for ln in block.splitlines() if ln.startswith("  fv_tanimoto")] == ["fv_tanimoto"]


def test_paired_fv_tanimoto_gives_the_hand_computed_wins_losses_ties_and_p_where_mean_tanimoto_disagrees():
    t, u = [0.30, 0.30, 0.30, 0.10, 0.20, 0.20], [0.25, 0.25, 0.25, 0.15, 0.20, 0.20]

    def run(g, tan, bad_first):
        rows = scored("val", g, 0, (["bad"] if bad_first else ["miss"]) + ["miss"] * 7)
        return [dict(r, tanimoto=tan) if r["valid"] else r for r in rows]
    A = [r for d in range(6) for r in run(f"d{d}", t[d], True)]          # the first draw fails, the next seven give t
    B = [r for d in range(6) for r in run(f"d{d}", u[d], False)]         # eight valid draws of u
    res = metrics.report(A, vs=B, boot=200)
    fv, mean = pm(res, metric="fv_tanimoto"), pm(res, metric="mean_tanimoto")
    # the first valid answer scores t in A: t - u = +.05 +.05 +.05 -.05 0 0 -> 3 wins, 1 loss, 2 ties, T = .10, 10 of 16 patterns
    assert (fv["wins"], fv["losses"], fv["ties"], fv["n_nontied"]) == (3, 1, 2, 4) and fv["p_value"] == 10 / 16
    assert fv["diff"] == pytest.approx(0.10 / 6) and fv["a"] == pytest.approx(sum(t) / 6) and fv["b"] == pytest.approx(sum(u) / 6)
    # over all attempts A scores 7t/8 (invalid = 0): +.0125 x3, -.0625, -.025 x2 -> 3 wins, 3 losses, no ties
    assert (mean["wins"], mean["losses"], mean["ties"], mean["n_nontied"]) == (3, 3, 0, 6)
    assert mean["p_value"] == pytest.approx(brute_p([0.0125] * 3 + [-0.0625, -0.025, -0.025]), abs=1e-12)
    cost = pm(res, metric="fv_samples")                                  # the cost reported beside it: 2 draws against 1
    assert (cost["a"], cost["b"], cost["diff"]) == (2.0, 1.0, 1.0)
    assert pm(res, metric="fv_rate")["a"] == pm(res, metric="fv_rate")["b"] == 1.0


# ------------------------------------------------------------------------------ round 2: finite scores, components, panel
@pytest.mark.parametrize("field,bad", [("reward", float("nan")), ("tanimoto", float("inf")), ("lift", float("nan")),
                                       ("lift", None), ("n_tokens", float("nan")), ("sample_idx", float("-inf")),
                                       ("sample_idx", float("nan")), ("reward", "0.5"), ("tanimoto", True), ("reward", False),
                                       ("lift", True), ("n_tokens", True), ("sample_idx", True)])      # a bool is not a number here
def test_non_finite_stored_scores_are_rejected_naming_the_field_and_the_count(field, bad):
    def run(corrupt):
        rows = [r for g in "abcd" for r in _eight("val", g, 0, "exact" if g in "ab" else "miss")]
        if corrupt:
            for g in "abc":                                              # one sample of three different prompt rows
                next(r for r in rows if r["group"] == g)[field] = bad
        return rows
    with pytest.raises(metrics.InputError, match=rf"^scored: 3 row\(s\) with a non-finite or non-numeric {field};"):
        metrics.report(run(True), boot=20)
    with pytest.raises(metrics.InputError, match=rf"^--vs: 3 row\(s\) with a non-finite or non-numeric {field};"):
        metrics.report(run(False), vs=run(True), boot=20)
    assert metrics.report(run(False), vs=run(False), boot=20)["splits"]["val"]["paired"]["metrics"]       # the clean pair reports


def test_nan_lift_cannot_silently_remove_drugs_from_a_constant_run():
    rows = run_real(real_data("val", VAL), lambda r, j: SMILES[0], n=2)
    assert abs(m(metrics.report(rows, boot=20), metric="lift")["value"]) < 1e-12
    for r in rows:
        if r["group"] != "a":
            r["lift"] = float("nan")                  # nanmean would have kept one drug and reported its lift as the mean
    with pytest.raises(metrics.InputError, match=r"8 row\(s\) with a non-finite or non-numeric lift"):
        metrics.report(rows, boot=20)


def test_score_samples_rejects_non_finite_sample_fields_and_scorer_results():
    data = [data_row("val", "a", 0)]
    nan = float("nan")
    with pytest.raises(metrics.InputError, match=r"^samples: 1 row\(s\) with a non-finite or non-numeric n_tokens;"):
        metrics.score_samples([{"id": "a/0", "sample_idx": 0, "text": "t", "n_tokens": nan}], data, fake_scorer)
    smp = [{"id": "a/0", "sample_idx": i, "text": "t"} for i in range(2)]
    for key in ("tanimoto", "reward"):
        with pytest.raises(metrics.InputError, match=rf"^scorer result: 2 row\(s\) with a non-finite or non-numeric {key};"):
            metrics.score_samples(smp, data, lambda text, ref: {**OUT["miss"], "answer": None, key: nan})


def test_an_absent_lift_is_not_a_non_finite_lift():
    old = [{k: v for k, v in r.items() if k != "lift"} for r in _eight("val", "a", 0, "exact")]
    assert m(metrics.report(old, boot=20), metric="lift")["value"] is None                  # absent is not non-finite


def _component(n=6, cluster=7):
    """n drugs of one component with the same win for A: the cluster is one unit."""
    A = [r for d in range(n) for r in valued("val", f"d{d}", 0, 0.2, 0.05, cluster=cluster)]
    B = [r for d in range(n) for r in valued("val", f"d{d}", 0, 0.2, 0.0, cluster=cluster)]
    return A, B


def test_partial_or_conflicting_component_metadata_raises_instead_of_changing_the_unit():
    A, B = _component()
    assert pm(metrics.report(A, vs=B, boot=50), metric="lift")["p_value"] == 1.0           # six wins in one component: one unit
    free = pm(metrics.report([{k: v for k, v in r.items() if k != "cluster_id"} for r in A],
                             vs=[{k: v for k, v in r.items() if k != "cluster_id"} for r in B], boot=50), metric="lift")
    assert free["p_value"] == 2 / 2 ** 6 and free["p_unit"] == "drug"                      # no ids at all: six units, 0.03125
    # the review's reproduction: delete ONE id from A and the p-value used to fall from 1.0 to 0.03125 without a word
    A[0].pop("cluster_id")
    with pytest.raises(metrics.InputError, match=r"^scored: 1 of 48 rows lack cluster_id while the others carry it"):
        metrics.report(A, vs=B, boot=50)
    with pytest.raises(metrics.InputError, match=r"^scored: 1 of 48 rows lack cluster_id"):
        metrics.report(A, boot=50)                                                         # alone, without --vs, too
    A, B = _component()
    B[5].pop("cluster_id")
    with pytest.raises(metrics.InputError, match=r"^--vs: 1 of 48 rows lack cluster_id"):
        metrics.report(A, vs=B, boot=50)
    A, B = _component()
    stripped = [{k: v for k, v in r.items() if k != "cluster_id"} for r in A]
    with pytest.raises(metrics.InputError, match="differ in carrying cluster_id"):          # A has none, B has all (and the reverse)
        metrics.report(stripped, vs=B, boot=50)
    with pytest.raises(metrics.InputError, match="differ in carrying cluster_id"):
        metrics.report(A, vs=stripped, boot=50)
    A, B = _component()
    for r in B:
        if r["group"] == "d0":
            r["cluster_id"] = 9
    with pytest.raises(metrics.InputError, match=r"^1 drug\(s\) have different cluster_id in A and B, e.g. \['d0'\]"):
        metrics.report(A, vs=B, boot=50)
    A, B = _component()
    B[0]["cluster_id"] = 9                                                                  # one drug in two clusters inside B
    with pytest.raises(metrics.InputError, match=r"^--vs: a drug has rows in two different clusters"):
        metrics.report(A, vs=B, boot=50)


def test_cluster_ids_that_disagree_with_the_data_are_rejected_and_data_fills_the_rest():
    A, B = _component()
    data = [data_row("val", f"d{d}", 0, SMILES[d], cluster_id=7) for d in range(6)]
    assert pm(metrics.report(A, vs=B, data=data, boot=50), metric="lift")["p_value"] == 1.0
    bare = [{k: v for k, v in r.items() if k != "cluster_id"} for r in A]                    # no ids in A: the data fills them
    assert pm(metrics.report(bare, vs=B, data=data, boot=50), metric="lift")["p_unit"] == "cluster"
    wrong = [data_row("val", f"d{d}", 0, SMILES[d], cluster_id=8) for d in range(6)]
    with pytest.raises(metrics.InputError, match=r"^scored: 48 row\(s\) whose cluster_id disagrees with the data"):
        metrics.report(A, data=wrong, boot=50)
    half = [data_row("val", f"d{d}", 0, SMILES[d], **({"cluster_id": 7} if d else {})) for d in range(6)]     # the data lacks one id
    with pytest.raises(metrics.InputError, match="lack cluster_id"):
        metrics.report(bare, data=half, boot=50)


def test_the_constant_bar_uses_the_full_reference_panel_so_a_partial_run_is_not_forced_to_zero():
    train = real_data("train", {f"t{i}": (s, 1) for i, s in enumerate(SMILES[:6])})
    val = real_data("val", VAL)
    refs = {g: s for g, (s, _) in VAL.items()}
    partial = [r for r in val if r["group"] in ("a", "d")]               # a run that sampled two of the four drugs
    own = metrics.score_samples([sample(r, j, f"t</think>\n{r['ref_smiles']}") for r in partial for j in range(8)], val)
    res = metrics.report(own, data=train + val, boot=20)
    bar = res["splits"]["val"]["constant"]
    centre = bar["smiles"]
    want = float(np.mean([oracle_lift(centre, g, refs) for g in ("a", "d")]))          # mean over the sampled drugs, against all four
    assert abs(want) > 0.01 and bar["metrics"]["lift"]["value"] == pytest.approx(want, abs=1e-12)
    assert bar["metrics"]["lift"]["n_drugs"] == 2
    # a run that answers the bar's constant on the same rows has exactly the bar's lift: one panel for both
    same = metrics.score_samples([sample(r, j, f"t</think>\n{centre}") for r in partial for j in range(8)], val)
    assert m(metrics.report(same, boot=20), "val", "lift")["value"] == pytest.approx(bar["metrics"]["lift"]["value"], abs=1e-12)
    # only a complete panel forces the constant to 0
    full = metrics.report(run_real(val, lambda r, j: r["ref_smiles"], n=8), data=train + val, boot=20)
    assert abs(full["splits"]["val"]["constant"]["metrics"]["lift"]["value"]) < 1e-12


# ------------------------------------------------------------------------------ round 2: which data files the CLI opens
def test_the_cli_opens_only_the_data_files_of_the_splits_it_needs(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    data.mkdir()
    drugs = {"train": ["t0", "t1"], "val": ["v0", "v1", "v2"], "test_r": ["r0"], "test_t": ["x0", "x1"]}
    ref = {g: SMILES[k] for k, g in enumerate(g for gs in drugs.values() for g in gs)}
    for s, gs in drugs.items():
        _write(data / f"{s}.jsonl", [data_row(s, g, 0, ref[g]) | {"cluster_id": k} for k, g in enumerate(gs)])
    opened = []
    real = metrics.read_jsonl
    monkeypatch.setattr(metrics, "read_jsonl", lambda path, drop=(): opened.append(Path(path).name) or real(path, drop))

    def samples(splits):
        return [{"id": f"{g}/0", "group": g, "split": s, "sample_idx": j, "finish_reason": "stop", "n_tokens": 5,
                 "text": f"t</think>\n{ref[g]}"} for s in splits for g in drugs[s] for j in range(8)]
    val_only = tmp_path / "val.samples.jsonl"
    _write(val_only, samples(["val"]))
    scored = tmp_path / "val.scored.jsonl"
    opened.clear()
    metrics.main(["score", "--samples", str(val_only), "--data", str(data), "--out", str(scored)])
    assert sorted(opened) == ["val.jsonl", "val.samples.jsonl"]                           # not train, not test_r, not test_t
    opened.clear()
    metrics.main(["report", "--scored", str(scored), "--data", str(data), "--boot", "20"])
    assert sorted(opened) == ["train.jsonl", "val.jsonl", "val.scored.jsonl"]             # val and the train references of the bar
    assert "constant-answer bar" in capsys.readouterr().out
    # a run that names test_t is the only way to open it; --split can keep it closed again
    both = tmp_path / "both.samples.jsonl"
    _write(both, samples(["val", "test_t"]))
    both_scored = tmp_path / "both.scored.jsonl"
    opened.clear()
    metrics.main(["score", "--samples", str(both), "--data", str(data), "--out", str(both_scored)])
    assert sorted(opened) == ["both.samples.jsonl", "test_t.jsonl", "val.jsonl"]
    opened.clear()
    metrics.main(["report", "--scored", str(both_scored), "--split", "val", "--data", str(data), "--boot", "20"])
    assert sorted(opened) == ["both.scored.jsonl", "train.jsonl", "val.jsonl"]
    opened.clear()
    metrics.main(["report", "--scored", str(both_scored), "--vs", str(both_scored), "--split", "val", "--data", str(data), "--boot", "20"])
    assert "test_t.jsonl" not in opened and "test_r.jsonl" not in opened
    # the other run's rows of splits that are not reported are left out, not flagged as unknown
    opened.clear()
    metrics.main(["report", "--scored", str(scored), "--vs", str(both_scored), "--data", str(data), "--boot", "20"])
    assert sorted(set(opened)) == ["both.scored.jsonl", "train.jsonl", "val.jsonl", "val.scored.jsonl"]
    capsys.readouterr()
    no_split = tmp_path / "no_split.jsonl"
    _write(no_split, [{k: v for k, v in samples(["val"])[0].items() if k != "split"}])
    with pytest.raises(SystemExit, match="lack split"):
        metrics.main(["score", "--samples", str(no_split), "--data", str(data), "--out", str(tmp_path / "never.jsonl")])
    other = tmp_path / "other"
    other.mkdir()
    _write(other / "train.jsonl", [data_row("train", "t0", 0, SMILES[0])])
    with pytest.raises(SystemExit, match="no val .jsonl files"):                          # the val file the samples need is absent
        metrics.main(["score", "--samples", str(val_only), "--data", str(other), "--out", str(tmp_path / "never.jsonl")])


# ------------------------------------------------------------------------------ round 2: wording and tolerances
def test_the_studentized_line_says_what_it_is_and_disagrees_with_the_p_value_where_it_can():
    # five wins of equal size among 41 drugs, 36 ties: exact p = 2 / 2^5 = 0.0625, yet the difference is above 2 bootstrap SEs
    A = [r for d in range(41) for r in valued("val", f"d{d}", 0, 0.2, 0.1 if d < 5 else 0.0)]
    B = [r for d in range(41) for r in valued("val", f"d{d}", 0, 0.2, 0.0)]
    res = metrics.report(A, vs=B, boot=4000, seed=1)
    e = pm(res, metric="lift")
    assert (e["wins"], e["losses"], e["ties"], e["p_value"]) == (5, 0, 36, 0.0625)
    assert e["studentized"] > 2.0 and e["p_value"] > 0.05                                   # the two do not agree: not a test
    text = metrics.format_report(res)
    assert "above about 2" not in text and "agrees with p" not in text
    assert sum("a size in bootstrap standard errors, not a calibrated test: it can disagree with the p-value" in ln
               for ln in text.splitlines()) == 1                                           # said once, on the studentized line


def test_the_sign_flip_tie_tolerance_is_a_nanometre_of_float_noise_not_a_threshold_on_the_effect():
    assert metrics.TIE == 1e-9
    rng = np.random.default_rng(0)
    assert metrics.sign_flip_p(np.array([1e-7] * 3), rng) == (0.25, "exact", 3)            # 1e-7 is a win: all-plus or all-minus
    assert metrics.sign_flip_p(np.array([5e-10] * 3), rng) == (1.0, "exact", 0)            # 5e-10 is float noise: a tie
    big, small = 0.2 + 1e-7, 0.2 + 5e-10
    A = [r for d in range(6) for r in valued("val", f"d{d}", 0, big if d < 3 else small, 0.0)]
    B = [r for d in range(6) for r in valued("val", f"d{d}", 0, 0.2, 0.0)]
    e = pm(metrics.report(A, vs=B, boot=50), metric="mean_tanimoto")
    assert (e["wins"], e["losses"], e["ties"], e["n_nontied"], e["p_value"]) == (3, 0, 3, 3, 0.25)


# ------------------------------------------------------------------------------- round 3: the cost of an answer
def _tokens(rows, tokens):
    for r, t in zip(rows, tokens):
        r["n_tokens"] = t
    return rows


def test_fv_tokens_counts_the_generated_tokens_up_to_the_first_valid_answer():
    t = [10, 20, 30, 40, 50, 60, 70, 90]
    a = _tokens(scored("val", "a", 0, ["bad", "bad", "miss", "exact", "bad", "bad", "bad", "bad"]), t)       # first valid at 2: 10 + 20 + 30
    b = _tokens(scored("val", "b", 0, ["bad"] * 8), t)                                                       # none: all eight samples
    c = scored("val", "c", 0, ["miss", "exact"] + ["bad"] * 6)
    c[0]["finish_reason"] = "length"                      # a valid label that the cap cut: its 10 tokens were spent, the first usable is 1
    c = _tokens(c, t)
    d0 = _tokens(scored("val", "d", 0, ["bad"] * 7 + ["miss"]), t)                                           # index 7: 370
    d1 = _tokens(scored("val", "d", 1, ["bad", "exact"] + ["bad"] * 6), t)                                   # index 1: 30
    e = _tokens(scored("val", "e", 0, ["exact"] * 5), t[:5])                                                 # five samples: skipped
    A = a + b + c + d0 + d1 + e
    res = metrics.report(A, boot=50)
    f = m(res, metric="fv_tokens")
    assert f["value"] == (60 + 370 + 30 + (370 + 30) / 2) / 4 == 165.0
    assert (f["n_drugs"], f["n_rows"], f["skipped_rows"], f["label"]) == (4, 5, 1, "deployable with resampling: costs fv_samples generations per answer")
    assert m(res, metric="fv_samples")["value"] == (3 + 8 + 2 + (8 + 2) / 2) / 4 == 4.5            # the draws it came with
    B = [dict(r, n_tokens=2 * r["n_tokens"]) for r in A]                                              # the same answers at twice the tokens
    paired = pm(metrics.report(A, vs=B, boot=50), metric="fv_tokens")
    assert (paired["a"], paired["b"], paired["diff"]) == (165.0, 330.0, -165.0) and "p_value" not in paired      # a cost, not a tested metric
    text = metrics.format_report(res)
    assert [ln.split()[1] for ln in text.splitlines() if ln.startswith("    fv_tokens")] == ["165.0"]


def test_prompt_tokens_is_the_drug_balanced_mean_n_prompt_tokens_of_the_prompt_rows_in_the_data_files():
    data = [data_row("val", "a", 0, n_prompt_tokens=100), data_row("val", "a", 1, n_prompt_tokens=300),
            data_row("val", "b", 0, n_prompt_tokens=600), data_row("val", "c", 0, n_prompt_tokens=9000)]        # c is not in the run
    run = _eight("val", "a", 0, "miss") + _eight("val", "a", 1, "miss") + _eight("val", "b", 0, "miss")
    res = metrics.report(run, data=data, boot=20)
    assert res["splits"]["val"]["prompt_tokens"] == 400.0                       # drugs count equally: (200 + 600) / 2, not 1000 / 3
    assert any(ln.startswith("  prompt_tokens 400.0 ") for ln in metrics.format_report(res).splitlines())
    json.dumps(res, allow_nan=False)
    assert metrics.report(run, boot=20)["splits"]["val"]["prompt_tokens"] is None                                # no data files
    for bad in (None, float("nan"), "100", True):                                # a row without a usable count: no value, not a guess
        broken = [dict(r, n_prompt_tokens=bad) if r["id"] == "b/0" else r for r in data]
        assert metrics.report(run, data=broken, boot=20)["splits"]["val"]["prompt_tokens"] is None
    missing = [{k: v for k, v in r.items() if k != "n_prompt_tokens"} if r["id"] == "a/1" else r for r in data]
    res = metrics.report(run, data=missing, boot=20)
    assert res["splits"]["val"]["prompt_tokens"] is None and "  prompt_tokens n/a " in metrics.format_report(res)


def test_prompt_tokens_is_the_same_number_whatever_order_the_rows_are_visited_in():
    rows = {"a": [296, 834], "b": [653, 662, 587], "c": [754, 254, 337], "d": [635, 499]}      # np.mean of their drug means depends on the order
    index = {r["id"]: r for g, v in rows.items() for j, n in enumerate(v) for r in [data_row("val", g, j, n_prompt_tokens=n)]}
    want = math.fsum(math.fsum(v) / len(v) for v in rows.values()) / len(rows)
    for order in itertools.permutations(rows):                              # the report visits a set of ids: any drug order may come
        ids = [i for g in order for i in index if index[i]["group"] == g]
        assert metrics._prompt_tokens(index, ids) == metrics._prompt_tokens(index, ids[::-1]) == metrics._prompt_tokens(index, set(ids)) == want
    assert metrics._prompt_tokens(index, []) is None and metrics._prompt_tokens(None, ["a/0"]) is None


def test_the_cost_is_stated_not_matched_and_fv_tanimoto_is_read_with_its_draws_and_tokens():
    text = metrics.format_report(metrics.report(_eight("val", "a", 0, "exact") + _eight("val", "b", 0, "miss"), boot=20))
    for where in (metrics.__doc__, text):
        assert "matched cost" not in where and "stated, not matched" in where
    assert "fv_samples (draws per answer) and fv_tokens (generated tokens up to the first valid answer" in text
    assert "an arm that invalidates its weak answers can raise it by paying in draws" in text
    assert "fv_tokens" in metrics.__doc__ and "prompt_tokens" in metrics.__doc__


# ---------------------------------------------------------------------------------- round 3: types, abort, the bar's rows
@pytest.mark.parametrize("field,bad", [("sample_idx", [1]), ("sample_idx", {"a": 1}), ("id", ["a/0"]), ("id", 7), ("group", {"g": 1}),
                                       ("group", 3), ("split", ["val"]), ("cluster_id", [1]), ("finish_reason", ["stop"])])
def test_non_scalar_identifiers_raise_input_error_not_a_traceback(field, bad):
    rows = _eight("val", "a", 0, "exact") + _eight("val", "b", 0, "miss")
    rows[0][field] = bad
    pattern = rf"^scored: 1 row\(s\) with a (non-string|non-finite or non-numeric|non-scalar) {field}"
    with pytest.raises(metrics.InputError, match=pattern):
        metrics.report(rows, boot=20)
    with pytest.raises(metrics.InputError, match=pattern.replace("scored", "--vs")):
        metrics.report(_eight("val", "a", 0, "exact"), vs=rows, boot=20)
    if field in ("sample_idx", "id"):
        smp = [{"id": "a/0", "sample_idx": 0, "text": "t"}, {"id": "a/0", "sample_idx": 1, "text": "t"}]
        smp[0][field] = bad
        with pytest.raises(metrics.InputError, match=rf"^samples: 1 row\(s\) with a (non-string|non-finite or non-numeric) {field}"):
            metrics.score_samples(smp, [data_row("val", "a", 0)], fake_scorer)


def test_two_nan_sample_indices_in_one_row_are_rejected_because_nan_is_never_equal_to_itself(tmp_path):
    rows = _eight("val", "a", 0, "exact")
    rows[0]["sample_idx"] = rows[1]["sample_idx"] = float("nan")            # NaN != NaN: a duplicate check alone would let both through
    with pytest.raises(metrics.InputError, match=r"^scored: 2 row\(s\) with a non-finite or non-numeric sample_idx"):
        metrics.report(rows, boot=20)
    listy = _eight("val", "a", 0, "exact")
    listy[3]["sample_idx"] = [3]
    path = tmp_path / "listy.jsonl"
    _write(path, listy)
    with pytest.raises(SystemExit, match=r"^error: scored: 1 row\(s\) with a non-finite or non-numeric sample_idx"):      # not a traceback
        metrics.main(["report", "--scored", str(path), "--data", str(tmp_path / "no-data"), "--boot", "20"])


@pytest.mark.parametrize("field,bad", [("valid", "false"), ("closed", 1), ("conn_match", None), ("scaffold_match", "True"),
                                       ("stereo_match", 0), ("answer", ["x"])])
def test_flags_must_be_real_booleans_so_the_text_false_never_counts_as_true(field, bad):
    rows = _eight("val", "a", 0, "exact") + _eight("val", "b", 0, "miss")
    rows[0][field] = bad
    pattern = rf"^scored: 1 row\(s\) with a (non-boolean|non-string) {field}$"
    with pytest.raises(metrics.InputError, match=pattern):
        metrics.report(rows, boot=20)
    with pytest.raises(metrics.InputError, match=pattern.replace("scored", "--vs")):
        metrics.report(rows[8:], vs=rows, boot=20)
    data = [data_row("val", "a", 0)]
    with pytest.raises(metrics.InputError, match=rf"^scorer result: 1 row\(s\) with a (non-boolean|non-string) {field}$"):
        metrics.score_samples([{"id": "a/0", "sample_idx": 0, "text": "t"}], data, lambda text, ref: {**OUT["miss"], "answer": None, field: bad})


def test_sign_flip_p_refuses_a_nan_or_inf_difference_instead_of_dropping_it_as_a_tie():
    rng = np.random.default_rng(0)
    assert metrics.sign_flip_p(np.array([0.2, 0.1, 0.3]), rng) == metrics.sign_flip_p([0.2, 0.1, 0.3], rng) == (0.25, "exact", 3)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(metrics.InputError, match="finite differences"):
            metrics.sign_flip_p(np.array([0.2, 0.1, bad, 0.3]), rng)           # the review's example: it gave the p of the three others
    with pytest.raises(metrics.InputError, match="finite differences"):
        metrics.sign_flip_p([0.2, None, 0.3], rng)


def test_abort_and_every_finish_reason_but_stop_is_a_failure_and_a_missing_one_is_finished():
    assert metrics.finished({}) and metrics.finished({"finish_reason": "stop"})
    assert not any(metrics.finished({"finish_reason": f}) for f in ("length", "abort", "error", "", None))
    data = real_data("val", {"a": (SMILES[0], 1), "b": (SMILES[1], 1)})
    ok, aborted = metrics.score_samples([sample(data[0], 0, f"t</think>\n{SMILES[0]}"),
                                         sample(data[0], 1, f"t</think>\n{SMILES[0]}", "abort")], data)
    assert ok["valid"] and ok["reward"] == 1.0 and ok["lift"] > 0
    assert (aborted["valid"], aborted["reward"], aborted["lift"], aborted["answer"]) == (False, 0.0, 0.0, SMILES[0])   # answer kept to inspect
    res = metrics.report([ok, aborted], boot=20)
    assert m(res, metric="valid_closed")["value"] == 0.5 and m(res, metric="truncation")["value"] == 0.0      # an abort is no length cut
    handmade = scored("val", "a", 0, ["exact"] * 8) + scored("val", "b", 0, ["miss"] * 8)
    for r in handmade[:8]:
        r["finish_reason"] = "abort"                                       # valid labels, but none of drug a's samples ended by itself
    res = metrics.report(handmade, boot=20)
    assert m(res, metric="valid_closed")["value"] == 0.5 and m(res, metric="fv_rate")["value"] == 0.5
    assert m(res, metric="tanimoto_valid")["n_drugs"] == 1 and m(res, metric="fv_samples")["value"] == (8 + 1) / 2


def test_the_bars_fv_rows_are_the_runs_fv_rows_when_some_rows_are_short():
    train = real_data("train", {f"t{i}": (s, 1) for i, s in enumerate(SMILES[:6])})
    val = real_data("val", VAL)
    rows = [r for r in run_real(val, lambda r, j: r["ref_smiles"], n=8) if not (r["id"] == "b/0" and r["sample_idx"] >= 5)]    # b/0: 5 samples
    res = metrics.report(rows, data=train + val, boot=20)
    sp = res["splits"]["val"]
    own, bar = sp["metrics"]["fv_tanimoto"], sp["constant"]["metrics"]["fv_tanimoto"]
    assert (own["n_drugs"], own["n_rows"], own["skipped_rows"]) == (3, 6, 1)           # drug b has only the short row: no value
    assert (bar["n_drugs"], bar["n_rows"], bar["skipped_rows"]) == (3, 6, 1)           # the bar uses the same rows
    centre = chem.standardize(sp["constant"]["smiles"])
    refs = {g: s for g, (s, _) in VAL.items()}
    want = float(np.mean([chem.tanimoto(centre, chem.standardize(refs[g])) for g in "acd"]))
    assert bar["value"] == pytest.approx(want, abs=1e-12) and sp["constant"]["metrics"]["mean_tanimoto"]["n_drugs"] == 4
    short = run_real(val, lambda r, j: r["ref_smiles"], n=2)                           # no row has eight samples: no run fv, no bar fv
    bare = metrics.report(short, data=train + val, boot=20)["splits"]["val"]
    assert bare["metrics"]["fv_tanimoto"]["value"] is None and bare["constant"]["metrics"]["fv_tanimoto"]["value"] is None


# ----------------------------------------------------------------------------------------------- cli
def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_cli_score_then_report_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(metrics, "_default_scorer", lambda: fake_scorer)
    data = tmp_path / "data"
    data.mkdir()
    drugs = {"train": ["t0", "t1", "t2"], "val": ["v0", "v1", "v2"], "test_t": ["x0", "x1"]}
    ref = {g: SMILES[k] for k, g in enumerate(g for gs in drugs.values() for g in gs)}      # one reference per drug
    cluster = {"v0": 10, "v1": 10, "v2": 11, "x0": 12, "x1": 13}                           # v0 and v1 are one component
    rows = {s: [data_row(s, g, i, ref[g]) | {"prompt": "P" * 50, "completion": "C" * 50, "cluster_id": cluster.get(g, 1),
                                             "ref_alt_smiles": []}
                for g in gs for i in range(2)] for s, gs in drugs.items()}
    for s, rs in rows.items():
        _write(data / f"{s}.jsonl", rs)
    eval_rows = rows["val"] + rows["test_t"]

    def samples(better):
        return [{"id": r["id"], "group": r["group"], "split": r["split"], "sample_idx": j, "finish_reason": "stop",
                 "n_tokens": 100, "model": "m", "mode": "think",
                 "text": "t</think>\n" + (r["ref_smiles"] if better and j < 5 else "miss")}
                for r in eval_rows for j in range(8)]

    for name, better in (("a", True), ("b", False)):
        _write(tmp_path / f"{name}.samples.jsonl", samples(better))
        metrics.main(["score", "--samples", str(tmp_path / f"{name}.samples.jsonl"), "--data", str(data),
                      "--out", str(tmp_path / f"{name}.scored.jsonl")])
    assert "scored 80 samples (10 prompt rows, 5 drugs)" in capsys.readouterr().out
    got = [json.loads(line) for line in (tmp_path / "a.scored.jsonl").read_text().splitlines()]
    assert len(got) == 80 and set(metrics.SCORE_KEYS) <= set(got[0]) and got[0]["text"].startswith("t</think>")
    assert {r["group"]: r["cluster_id"] for r in got} == cluster                           # scored rows carry the cluster
    assert all(r["ref_alt_smiles"] == [] for r in got)
    assert all(isinstance(r["lift"], float) for r in got) and any(r["lift"] > 0 for r in got)       # v3: lift is in the row
    assert not list(tmp_path.glob("*.tmp"))

    out_json = tmp_path / "rep.json"
    metrics.main(["report", "--scored", str(tmp_path / "a.scored.jsonl"), "--vs", str(tmp_path / "b.scored.jsonl"),
                  "--split", "test_t,val", "--boot", "300", "--data", str(data), "--json", str(out_json)])
    text = capsys.readouterr().out
    res = json.loads(out_json.read_text())
    assert list(res["splits"]) == ["test_t", "val"] and "split test_t   drugs 2 of 2   rows 4 of 4" in text
    e = res["splits"]["val"]["paired"]["metrics"]["exact@1"]
    assert e["diff"] == pytest.approx(5 / 8) and e["excludes_zero"] and res["splits"]["val"]["constant"]
    assert (e["wins"], e["losses"], e["ties"]) == (3, 0, 0)                                  # drugs
    assert (e["n_clusters"], e["n_nontied"], e["p_unit"], e["p_value"]) == (2, 2, "cluster", 0.5)   # v0, v1 flip together
    assert res["splits"]["val"]["bootstrap"] == {"unit": "cluster", "n_units": 2}
    assert res["splits"]["val"]["metrics"]["exact@8"]["label"].startswith("oracle coverage")
    assert "paired A - B on 3 common drugs" in text and "*" in text and "n_tokens_p90" in text
    assert "bootstrap resamples 2 clusters of 3 drugs" in text and "oracle coverage (needs the answer" in text
    assert "sign-flip permutation test" in text and "0.5000" in text and "clusters" in text
    lift = res["splits"]["val"]["paired"]["metrics"]["lift"]                                    # v3: lift is tested too
    assert lift["diff"] > 0 and (lift["wins"], lift["losses"], lift["ties"], lift["p_value"]) == (3, 0, 0, 0.5)
    assert set(lift) >= {"boot_se", "studentized"} and "studentized component sensitivity" in text
    assert text.count("Sign-flip test assumptions") == 1 and abs(res["splits"]["val"]["constant"]["metrics"]["lift"]["value"]) < 1e-12
    # a data directory that does not match the run is an error, not a silent partial report
    other = tmp_path / "other"
    other.mkdir()
    _write(other / "val.jsonl", [data_row("val", "z", 0, SMILES[0])])
    with pytest.raises(SystemExit, match="unknown id"):
        metrics.main(["report", "--scored", str(tmp_path / "a.scored.jsonl"), "--data", str(other)])
    with pytest.raises(SystemExit, match="unknown id"):
        metrics.main(["score", "--samples", str(tmp_path / "a.samples.jsonl"), "--data", str(other),
                      "--out", str(tmp_path / "never.jsonl")])
    assert not (tmp_path / "never.jsonl").exists()
    metrics.main(["report", "--scored", str(tmp_path / "a.scored.jsonl"), "--data", str(tmp_path / "missing"), "--boot", "50"])
    assert "coverage and constant-answer bar skipped" in capsys.readouterr().out


# ------------------------------------------------------------------------- real reward.py, if present
def test_real_reward_score_contract_smoke():
    reward = pytest.importorskip("rlexp.reward")
    from rlexp import chem
    if not hasattr(reward, "score"):
        pytest.skip("reward.py v2 (rlexp.reward.score) not written yet")
    aspirin, ibuprofen = SMILES[4], SMILES[0]
    s_ibu, r_ibu = "C[C@H](c1ccc(CC(C)C)cc1)C(=O)O", "C[C@@H](c1ccc(CC(C)C)cc1)C(=O)O"
    mol = chem.standardize(aspirin)
    ref = {"ref_smiles": aspirin, "ref_conn": chem.conn(mol), "ref_scaffold": chem.scaffold(mol)}
    data = [{"id": "a/0", "group": "a", "split": "val", **ref}]
    texts = [f"thinking</think>\n{aspirin}", f"thinking</think>\n**{ibuprofen}**.", f"thinking {aspirin}", "</think>\nno idea"]
    smp = [{"id": "a/0", "sample_idx": i, "text": t} for i, t in enumerate(texts)]
    exact, other, unclosed, none = metrics.score_samples(smp, data)
    assert set(metrics.SCORE_KEYS) <= set(exact)
    assert exact["conn_match"] and exact["reward"] == 1.0 and exact["closed"] and exact["valid"]
    assert not other["conn_match"] and other["valid"] and 0 < other["reward"] < 1 and other["answer"] == ibuprofen
    assert not unclosed["closed"] and unclosed["reward"] == 0
    assert none["answer"] is None and not none["valid"] and none["reward"] == 0
    # stereo_match: same connectivity, different full InChIKey; reported, never rewarded; ref_inchikey optional
    mol = chem.standardize(s_ibu)
    for extra in ({}, {"ref_inchikey": chem.inchikey(mol)}):
        row = [{"id": "s/0", "group": "s", "split": "val", "ref_smiles": s_ibu, "ref_conn": chem.conn(mol),
                "ref_scaffold": chem.scaffold(mol), **extra}]
        same, flipped = metrics.score_samples([{"id": "s/0", "sample_idx": i, "text": f"t</think>\n{a}"}
                                               for i, a in enumerate((s_ibu, r_ibu))], row)
        assert same["stereo_match"] and same["conn_match"]
        assert flipped["conn_match"] and not flipped["stereo_match"] and flipped["reward"] == same["reward"] == 1.0
