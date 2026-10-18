"""Tests for scripts/cohort_diagnostics.py on synthetic fixtures (public structures used only as fixtures)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cohort_diagnostics as cd      # noqa: E402

from rlexp import chem      # noqa: E402

ASPIRIN, CAFFEINE, IBUPROFEN = "CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"


def sample(group, rid, idx, tan, valid=True, finish="stop", split="test_t"):
    return {"group": group, "id": rid, "sample_idx": idx, "tanimoto": tan, "valid": valid, "closed": True, "finish_reason": finish, "split": split}


def test_quartiles_ignore_missing_values():
    q = cd.quartiles([1.0, 2.0, None, 3.0, 4.0])
    assert q["n"] == 4 and q["median"] == 2.5 and cd.quartiles([None])["n"] == 0


def test_the_first_usable_sample_of_a_row_counts_and_a_row_without_one_is_zero():
    rows = [sample("a", "a-1", 0, 0.9, valid=False), sample("a", "a-1", 1, 0.2), sample("a", "a-1", 2, 0.9),     # first usable is 0.2
            sample("a", "a-2", 0, 0.7, finish="length"),                                                      # cut by the cap: no usable answer
            sample("b", "b-1", 0, 0.5, split="val")]                                                          # another split is ignored
    assert cd.fv_per_drug(rows, "test_t") == {"a": pytest.approx(0.1)}


def test_equal_effects_give_a_zero_difference_and_the_mdd_formula():
    diffs = {f"g{i}": 0.01 * (i % 5 - 2) + 0.02 for i in range(40)}
    clusters = {g: g for g in diffs}
    h = cd.heterogeneity(diffs, dict(diffs), clusters, dict(clusters), boot=2000)
    assert h["difference"] == pytest.approx(0.0) and h["ci"][0] < 0 < h["ci"][1]
    assert h["mdd_t"] == pytest.approx(2.8 * np.std(list(diffs.values()), ddof=1) / np.sqrt(40))


def test_a_shift_in_one_split_is_a_shift_in_the_difference():
    rng = np.random.default_rng(0)
    t = {f"t{i}": float(x) for i, x in enumerate(rng.normal(0.0, 0.01, 40))}
    r = {f"r{i}": float(x) + 0.05 for i, x in enumerate(rng.normal(0.0, 0.01, 30))}
    h = cd.heterogeneity(t, r, {g: g for g in t}, {g: g for g in r}, boot=2000)
    assert h["difference"] < -0.04 and h["ci"][1] < 0 and h["share_of_draws_at_or_below_zero"] == 1.0


def test_related_drugs_are_one_unit():
    diffs = {"a": 0.1, "b": 0.1, "c": -0.1}
    units = cd.effect_by_cluster(diffs, {"a": "x", "b": "x", "c": "y"})
    assert sorted(map(tuple, units.tolist())) == [(-0.1, 1), (0.2, 2)]


def test_headroom_is_the_best_train_similarity_minus_the_constants():
    mols = [chem.standardize(s) for s in (ASPIRIN, CAFFEINE)]
    train = [chem.standardize(s) for s in (ASPIRIN, IBUPROFEN)]
    h = cd.headroom(mols, train, chem.standardize(ASPIRIN))
    # aspirin: best train is itself (1.0) and the constant is itself (1.0): gap 0; caffeine: best train minus the constant's similarity to it
    expected = [0.0, max(chem.tanimoto(mols[1], t) for t in train) - chem.tanimoto(chem.standardize(ASPIRIN), mols[1])]
    assert h["n"] == 2 and h["mean"] == pytest.approx(np.mean(expected)) and h["share_at_or_below_zero"] >= 0.5


def test_crowding_counts_the_nearest_other_reference():
    mols = [chem.standardize(s) for s in (ASPIRIN, ASPIRIN, CAFFEINE)]
    c = cd.crowding(mols)
    assert c["n"] == 3 and c["share_nearest_above_0.25"] == pytest.approx(2 / 3) and 0 < c["mean_pairwise"] < 1


def test_the_permutation_test_separates_samples_and_not_equal_ones():
    assert cd.perm_p([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]) == pytest.approx(1.0)
    assert cd.perm_p(list(range(30)), list(range(100, 130)), draws=2000) < 0.01
