"""metrics.py on synthetic data with a fake scorer. Only the last test touches the real rlexp.reward.score (skipped if absent)."""
import itertools
import json
import math
import random
import types
from fractions import Fraction

import numpy as np
import pytest

import rlexp
from rlexp import metrics

# label -> score() keys. The fake scorer reads the label after </think>.
OUT = {
    "exact": dict(closed=True, valid=True, tanimoto=1.0, scaffold_match=True, conn_match=True, stereo_match=True, reward=1.0),
    "flat": dict(closed=True, valid=True, tanimoto=0.9, scaffold_match=True, conn_match=True, stereo_match=False, reward=1.0),
    "scaf": dict(closed=True, valid=True, tanimoto=0.5, scaffold_match=True, conn_match=False, stereo_match=False, reward=0.65),
    "miss": dict(closed=True, valid=True, tanimoto=0.1, scaffold_match=False, conn_match=False, stereo_match=False, reward=0.11),
    "bad": dict(closed=True, valid=False, tanimoto=0.0, scaffold_match=False, conn_match=False, stereo_match=False, reward=0.0),
    "open": dict(closed=False, valid=False, tanimoto=0.0, scaffold_match=False, conn_match=False, stereo_match=False, reward=0.0),
}


def fake_scorer(text, ref):
    """OUT[label]; an answer equal to ref_smiles is exact, so it only scores right if the ref was joined right."""
    if "</think>" not in text:
        return {**OUT["open"], "answer": None}
    tail = text.rsplit("</think>", 1)[1].strip()
    label = "exact" if tail == ref["ref_smiles"] else tail
    return {**OUT.get(label, OUT["miss"]), "answer": None if label == "bad" else tail}


def scored(split, group, row, labels, tok=100, fin="stop", answers=None, cluster=None):
    """Scored samples of one prompt row, one per label. answers: the extracted SMILES per sample (default None)."""
    return [{"id": f"{group}/{row}", "group": group, "split": split, "sample_idx": i, "finish_reason": fin,
             "n_tokens": tok, "answer": answers[i] if answers else None,
             **({"cluster_id": cluster} if cluster is not None else {}), **OUT[lab]} for i, lab in enumerate(labels)]


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
    assert metrics.METRICS == ("exact@1", "core@1", "stereo_exact@1", "mean_tanimoto", "mean_reward", "valid_closed",
                               "truncation", "n_tokens_mean", "n_tokens_p90", "exact@8", "core@8", "maj@8")
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
    rows = (scored("val", "A", 0, ["exact"], fin="stop") + scored("val", "A", 0, ["open"], fin="length")
            + scored("val", "A", 0, ["bad"], fin="stop") + scored("val", "A", 0, ["miss"], fin="length"))
    for i, r in enumerate(rows):
        r["sample_idx"] = i
    rows += scored("val", "B", 0, ["exact"])
    res = metrics.report(rows, boot=50)
    assert m(res, metric="valid_closed")["value"] == pytest.approx((2 / 4 + 1) / 2)   # bad and open do not count
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
    res = metrics.report(_eight("val", "a", 0, "exact") + _eight("val", "b", 0, "miss"), vs=_eight("val", "a", 0, "miss")
                         + _eight("val", "b", 0, "miss"), boot=50)
    for k, e in res["splits"]["val"]["metrics"].items():
        assert e.get("label") == (label if k in ("exact@8", "core@8") else None), k
    for k, e in res["splits"]["val"]["paired"]["metrics"].items():
        assert e.get("label") == (label if k in ("exact@8", "core@8") else None), k
    text = metrics.format_report(res)
    assert text.count(label + ":") == 2                                  # above the pass@8 rows of both tables
    first = text.index(label)
    assert first < text.index("exact@8") < text.index("core@8") < text.index("maj@8", first)   # maj@8 comes after the block
    json.dumps(res, allow_nan=False)                                     # and the label survives into the JSON


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
    # one drug without a cluster id: the split falls back to drugs rather than guessing
    part = [dict(r, cluster_id=int(r["group"][1:-1])) for r in rows[1:]] + rows[:1]
    assert metrics.report(part, boot=100)["splits"]["val"]["bootstrap"]["unit"] == "drug"
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
    run = [r for d in ("v0/0", "v0/1", "v1/0") for r in scored("val", d.split("/")[0], int(d[-1]), ["miss"] * 2)]
    res = metrics.report(run, data=train + val, scorer=fake_scorer, boot=100)
    sp = res["splits"]["val"]
    assert sp["coverage"] == {"drugs": 3, "rows": 4} and (sp["n_drugs"], sp["n_rows"]) == (2, 3)
    bar = sp["constant"]
    assert bar["smiles"] == centre and set(bar["metrics"]) == set(metrics.BAR)
    assert bar["metrics"]["exact@1"]["value"] == pytest.approx(0.5)       # v0 yes, v1 no; v2 not scored
    assert (bar["metrics"]["exact@1"]["n_drugs"], bar["metrics"]["exact@1"]["n_rows"]) == (2, 3)
    assert "PARTIAL" in metrics.format_report(res) and "constant-answer bar" in metrics.format_report(res)
    assert "highest mean bit Tanimoto" in metrics.format_report(res) and 0 < bar["mean_bit_tanimoto_to_train"] < 1


def test_bar_is_skipped_with_a_note_when_it_cannot_be_computed(monkeypatch):
    run = scored("val", "v0", 0, ["miss"])
    monkeypatch.setattr(rlexp, "reward", types.SimpleNamespace(), raising=False)         # no reward.score at all
    res = metrics.report(run, data=[data_row("train", "t0", 0, SMILES[0]), data_row("train", "t1", 0, SMILES[1]),
                                    data_row("val", "v0", 0, SMILES[2])], boot=50)
    assert "constant" not in res["splits"]["val"] and "skipped" in res["notes"][0] and "reward.score" in res["notes"][0]
    res = metrics.report(run, data=[data_row("val", "v0", 0, SMILES[2])], scorer=fake_scorer, boot=50)
    assert "constant" not in res["splits"]["val"] and "train" in res["notes"][0]


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
