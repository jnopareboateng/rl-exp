"""Pure tests for rlexp.grpo: advantages, token weights, the drug schedule, collapse score, batch diagnostics (answer shares, reward
components, profile-specific lift), aggregates and stop rules. No torch, no chemistry. The loop, loss and model-side tests are in
tests/test_rl_pack.py."""
import random
import warnings

import numpy as np
import pytest

from rlexp import grpo


# ---------------------------------------------------------------- advantages
def test_advantages_are_centred_per_group_and_scaled_by_one_batch_std():
    rewards = [[0.0, 1.0, 0.5, 0.5], [0.2, 0.4, 0.6, 0.8]]
    adv, std = grpo.group_advantages(rewards)
    pooled = np.sqrt((0.25 + 0.25 + 0.09 + 0.01 + 0.01 + 0.09) / 8)
    assert std == pytest.approx(pooled)
    centred = [np.array([-0.5, 0.5, 0.0, 0.0]), np.array([-0.3, -0.1, 0.1, 0.3])]
    for got, want in zip(adv, centred):
        np.testing.assert_allclose(got, want / (pooled + 1e-6), rtol=1e-12)
        assert got.sum() == pytest.approx(0.0, abs=1e-12)            # centred on the group
    assert adv[0][1] > 0 > adv[0][0]                                   # above the group mean is a positive advantage
    assert np.concatenate(adv).std() == pytest.approx(1.0, abs=1e-5)  # the batch std is the divisor
    # never the per-group std: the group whose rewards spread less keeps proportionally smaller advantages
    assert adv[1][3] / adv[0][1] == pytest.approx(0.3 / 0.5, rel=1e-5)


def test_advantages_use_every_sample_of_the_group_including_the_truncated_one():
    # sample 3 stands for a truncated completion: scored like any other, so it moves the group mean
    adv, _ = grpo.group_advantages([[1.0, 0.0, 0.0, 0.0]])
    assert adv[0][0] > 0 and adv[0][3] < 0 and adv[0][1] == adv[0][2] == adv[0][3]
    assert adv[0].sum() == pytest.approx(0.0, abs=1e-12)


def test_zero_variance_groups_are_dropped_and_not_counted():
    adv, std = grpo.group_advantages([[0.3, 0.3, 0.3, 0.3], [0.0, 1.0, 0.0, 1.0]])
    assert adv[0] is None and adv[1] is not None
    assert std == pytest.approx(0.5)                                   # only the retained group sets the batch std
    plan = grpo.plan_update([[0.3] * 4, [0.0, 1.0, 0.0, 1.0]], [[5] * 4, [5] * 4])
    assert plan["groups"] == 1 and plan["zero_variance"] == 1 and plan["weights"][0] is None
    # the tolerance is on the spread: inside 1e-9 is dropped, above it is kept
    assert grpo.group_advantages([[0.1, 0.1 + 5e-10]])[0] == [None]
    assert grpo.group_advantages([[0.1, 0.1 + 5e-8]])[0][0] is not None


def test_no_retained_group_gives_no_update_and_no_nan():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adv, std = grpo.group_advantages([[0.2, 0.2], [0.0, 0.0, 0.0]])
        plan = grpo.plan_update([[0.2, 0.2], [0.0, 0.0, 0.0]], [[3, 4], [1, 1, 1]])
    assert adv == [None, None] and std is None
    assert plan["groups"] == 0 and plan["weights"] == [None, None] and plan["zero_variance"] == 2
    assert grpo.group_advantages([]) == ([], None)


@pytest.mark.parametrize("bad", [[], [0.0, float("nan")], [float("inf"), 1.0]])
def test_a_group_with_no_or_non_finite_rewards_is_an_error(bad):
    with pytest.raises(ValueError):
        grpo.group_advantages([bad])


# ---------------------------------------------------------------- per-token coefficients
def test_token_coefficient_is_advantage_over_group_tokens_times_groups():
    rewards = [[0.0, 1.0, 0.5, 0.2], [0.1, 0.9, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5]]
    lengths = [[2, 2, 40, 6], [10, 10, 10, 10], [3, 3, 3, 3]]
    plan = grpo.plan_update(rewards, lengths)
    adv, _ = grpo.group_advantages(rewards)
    assert plan["groups"] == 2 and plan["weights"][2] is None
    np.testing.assert_allclose(plan["weights"][0], adv[0] / (50 * 2), rtol=1e-12)   # T_0 = 50, G = 2
    np.testing.assert_allclose(plan["weights"][1], adv[1] / (40 * 2), rtol=1e-12)   # T_1 = 40
    # every token of one completion carries the same coefficient, whatever its length
    assert plan["weights"][0][2] == pytest.approx(adv[0][2] / 100)


def test_a_long_completion_or_a_long_group_does_not_dominate():
    rewards = [[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
    short, long_ = [2, 4, 2, 4], [20, 40, 20, 40]                      # the second group is ten times longer
    plan = grpo.plan_update(rewards, [short, long_])
    mass = [float(np.abs(w * np.array(n)).sum()) for w, n in zip(plan["weights"], (short, long_))]
    assert mass[0] == pytest.approx(mass[1])                           # each prompt weighs the same
    # scaling a group's lengths uniformly leaves its total weight on tokens unchanged
    scaled = grpo.plan_update(rewards, [[10 * x for x in short], long_])
    before = float((plan["weights"][0] * np.array(short)).sum())
    after = float((scaled["weights"][0] * 10 * np.array(short)).sum())
    assert before == pytest.approx(after)
    # inside a group the longer completion does not get a bigger coefficient, only more tokens at the same coefficient
    mixed = grpo.plan_update([[1.0, 0.0]], [[1, 100]])
    assert mixed["weights"][0][0] == pytest.approx(-mixed["weights"][0][1])


def test_token_weights_reject_empty_completions():
    adv, _ = grpo.group_advantages([[0.0, 1.0]])
    with pytest.raises(ValueError):
        grpo.token_weights(adv, [[0, 3]])
    with pytest.raises(ValueError):
        grpo.token_weights(adv, [[3]])


def test_refill_skip_and_warmup_arithmetic():
    assert grpo.refill_count(16, 32, 16) == 0 and grpo.refill_count(20, 32, 16) == 0
    assert grpo.refill_count(10, 32, 16) == 20                          # 6 missing at a survival rate of 10/32
    assert grpo.refill_count(15, 32, 16) == 3
    assert grpo.refill_count(0, 32, 16) == 32 and grpo.refill_count(1, 32, 16) == 32   # never more than doubling
    assert [grpo.skip_below(m) for m in (16, 4, 1)] == [8, 2, 1]
    assert [grpo.lr_at(s, 1e-5) for s in (1, 2, 5, 6, 100)] == pytest.approx([2e-6, 4e-6, 1e-5, 1e-5, 1e-5])


def test_skip_below_is_half_of_min_groups_rounded_up_never_below_half():
    assert [grpo.skip_below(m) for m in (1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17)] == [1, 1, 2, 2, 3, 3, 4, 4, 8, 8, 9]
    for m in range(1, 60):                                                  # the smallest integer that is at least half of m
        assert 2 * grpo.skip_below(m) >= m and 2 * (grpo.skip_below(m) - 1) < m


# ---------------------------------------------------------------- schedule
SIZES = [1, 2, 3, 1, 2, 1, 3, 2, 1, 4]                                   # views per drug; 10 drugs


def test_the_schedule_is_deterministic_and_changes_with_the_seed():
    first, cursor = grpo.draw(SIZES, 7, 0, 4)
    assert (first, cursor) == grpo.draw(SIZES, 7, 0, 4)
    assert grpo.draw(SIZES, 8, 0, 4)[0] != first
    assert grpo.pass_order(10, 7, 0) == grpo.pass_order(10, 7, 0) != grpo.pass_order(10, 7, 1)


def test_every_drug_is_visited_once_per_pass_and_a_step_never_repeats_a_drug():
    picks, cursor = grpo.draw(SIZES, 3, 0, 5)
    more, cursor = grpo.draw(SIZES, 3, cursor, 5)
    assert sorted(d for d, _ in picks + more) == list(range(10)) and cursor == 10
    assert sorted(d for d, _ in grpo.draw(SIZES, 3, 0, 10)[0]) == list(range(10))
    # a step that spans two passes still holds distinct drugs
    cursor = 0
    for _ in range(25):
        step, cursor = grpo.draw(SIZES, 3, cursor, 4)
        assert len({d for d, _ in step}) == 4 and cursor >= 4


def test_a_resumed_schedule_equals_the_uninterrupted_one():
    whole, cursor = [], 0
    for _ in range(9):
        step, cursor = grpo.draw(SIZES, 11, cursor, 4)
        whole.append(step)
    end = cursor
    head, cursor = [], 0
    for _ in range(4):
        step, cursor = grpo.draw(SIZES, 11, cursor, 4)
        head.append(step)
    saved = int(cursor)                                                  # all a checkpoint keeps of the schedule
    tail = []
    for _ in range(5):
        step, saved = grpo.draw(SIZES, 11, saved, 4)
        tail.append(step)
    assert head + tail == whole and saved == end
    # a resume that lost the cursor would repeat the first step
    assert grpo.draw(SIZES, 11, 0, 4)[0] == whole[0] != whole[4]


def test_exclude_cap_and_views():
    picks, _ = grpo.draw(SIZES, 0, 0, 5, exclude={0, 1, 2})
    assert not {0, 1, 2} & {d for d, _ in picks} and len(picks) == 5
    picks, cursor = grpo.draw([1, 1, 1], 0, 0, 5)
    assert sorted(d for d, _ in picks) == [0, 1, 2] and cursor == 3      # capped at the drugs there are
    assert grpo.draw([1, 1, 1], 0, 0, 2, exclude={0, 1, 2}) == ([], 0)
    seen, cursor = set(), 0
    for _ in range(60):
        step, cursor = grpo.draw([3, 3, 3], 1, cursor, 3)
        assert all(0 <= v < 3 for _, v in step)
        seen |= {v for _, v in step}
    assert seen == {0, 1, 2}                                             # a drug shows other views in later passes
    assert all(v == 0 for _, v in grpo.draw([1, 1, 1], 0, 0, 3)[0])      # a one-view drug always shows it


def test_rollout_seed_follows_the_documented_formula():
    assert grpo.rollout_seed(3, 5) == 3 * 1_000_003 + 5
    assert grpo.rollout_seed(3, 5, 1) == grpo.rollout_seed(3, 5) + 2 ** 40
    assert len({grpo.rollout_seed(s, t, r) for s in (0, 1) for t in range(1, 50) for r in (0, 1)}) == 196


# ---------------------------------------------------------------- collapse score and aggregates
def test_collapse_score_counts_distinct_modal_answers_over_groups_with_a_valid_answer():
    keys = [["A", "A", "B", None], ["A", "C", "C", "C"], [None, None], ["B"]]
    assert grpo.collapse_score(keys) == pytest.approx(1.0)               # modal A, C, B: three distinct over three groups
    assert grpo.collapse_score([["A", "A"], ["A", None], ["A", "B", "A"]]) == pytest.approx(1 / 3)
    assert grpo.collapse_score([["B", "A"], ["A"]]) == pytest.approx(1 / 2)   # a tie takes the smallest key: A and A
    assert grpo.collapse_score([["B", "A"], ["B"]]) == pytest.approx(1.0)     # A and B
    assert grpo.collapse_score([[None, None], []]) is None
    assert grpo.collapse_score([]) is None


def sample(reward, drug="d", valid=True, closed=True, conn=False, scaffold=False, tani=0.1, n=50, truncated=False):
    return dict(reward=reward, drug=drug, valid=valid, closed=closed, conn_match=conn, scaffold_match=scaffold,
                tanimoto=tani, n_tokens=n, truncated=truncated)


def test_summarize_reports_the_documented_rates():
    samples = [sample(1.0, conn=True, tani=1.0, n=10), sample(0.4, scaffold=True, tani=0.5, n=20),
               sample(0.0, valid=False, closed=False, tani=0.0, n=256, truncated=True),
               sample(0.1, closed=True, tani=0.1, n=30), sample(0.0, valid=False, closed=True, tani=0.0, n=40)]
    out = grpo.summarize(samples)
    assert out["samples"] == 5 and out["reward_mean"] == pytest.approx(0.30)
    assert out["reward_std"] == pytest.approx(float(np.std([1.0, 0.4, 0.0, 0.1, 0.0])))
    assert out["valid"] == pytest.approx(0.6) and out["closed"] == pytest.approx(0.8)
    assert out["valid_closed"] == pytest.approx(0.6) and out["truncation"] == pytest.approx(0.2)
    assert out["core"] == pytest.approx(0.4) and out["exact"] == pytest.approx(0.2)
    assert out["tanimoto"] == pytest.approx(0.32) and out["length_mean"] == pytest.approx(71.2)
    assert out["length_p90"] == 256                                       # nearest rank: ceil(0.9 * 5) = 5th of 5
    ten = grpo.summarize([sample(0.0, n=i) for i in range(1, 11)])
    assert ten["length_p90"] == 9                                         # ceil(0.9 * 10) = 9th value
    with pytest.raises(ValueError):
        grpo.summarize([])


def test_drug_balanced_means_weigh_each_drug_once():
    samples = [sample(1.0, "x", conn=True), sample(1.0, "x", conn=True), sample(1.0, "x", conn=True), sample(0.0, "y", valid=False)]
    out = grpo.drug_balanced(samples)
    assert out["reward"] == pytest.approx(0.5) and out["exact"] == pytest.approx(0.5)   # 0.75 by sample, 0.5 by drug
    assert out["valid_closed"] == pytest.approx(0.5) and out["drugs"] == 2 and out["samples"] == 4
    assert set(out) == {"reward", "reward_full", "validity", "valid_closed", "core", "exact", "tanimoto", "tanimoto_valid", "truncation",
                        "length", "drugs", "samples"}


def test_train_window_averages_the_steps_after_a_step():
    metrics = [dict(step=s, reward_mean=r) for s, r in zip(range(1, 7), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))]
    assert grpo.train_window(metrics, 0) == pytest.approx(0.35) and grpo.train_window(metrics, 3) == pytest.approx(0.5)
    assert grpo.train_window(metrics, 6) is None


# ---------------------------------------------------------------- stop rules
def test_collapse_stops_after_three_consecutive_low_steps_and_none_resets():
    streak, seen = 0, []
    for score in (0.5, 0.6, 0.9, 0.5, 0.5, None, 0.69, 0.7, 0.1, 0.1, 0.1):
        streak = grpo.collapse_streak(streak, score)
        seen.append(streak)
    assert seen == [1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 3]
    assert grpo.stop_status(2, 0.0, [], 3) is None and grpo.stop_status(3, 0.0, [], 3) == "stopped_collapse"


def test_truncation_stops_above_thirty_percent_only():
    assert grpo.stop_status(0, 0.30, [], 3) is None
    assert grpo.stop_status(0, 0.31, [], 3) == "stopped_truncation"
    assert grpo.stop_status(3, 0.9, [], 3) == "stopped_collapse"          # collapse is reported first


def ev(val, train):
    return dict(val=val, train=train)


def test_patience_needs_val_flat_for_the_window_while_train_rises():
    base = [ev(0.10, None), ev(0.20, 0.10), ev(0.19, 0.12), ev(0.18, 0.14)]
    assert not grpo.patience_stop(base, 3)                                # only two evals since the best
    assert grpo.patience_stop(base + [ev(0.19, 0.16)], 3)                 # three evals without a new best, train 0.10 -> 0.16
    assert not grpo.patience_stop(base + [ev(0.19, 0.09)], 3)             # train did not rise: no memorisation signal
    assert not grpo.patience_stop(base + [ev(0.19, 0.10)], 3)             # train back at the value of the best eval: a tie is no rise
    assert not grpo.patience_stop(base + [ev(0.21, 0.16)], 3)             # a new best resets the count
    assert grpo.patience_stop(base + [ev(0.20, 0.16)], 3)                 # equal to the best is not an improvement
    assert grpo.patience_stop([ev(0.2, None), ev(0.2, 0.1), ev(0.2, 0.2), ev(0.2, 0.3)], 3)    # ties never count as improvement
    assert grpo.patience_stop([ev(0.30, None), ev(0.20, 0.10), ev(0.25, 0.12), ev(0.28, 0.15)], 3)   # the baseline is the best
    assert not grpo.patience_stop([ev(0.30, None), ev(0.20, 0.10), ev(0.25, 0.12), ev(0.28, 0.05)], 3)
    assert not grpo.patience_stop([ev(0.3, None)], 3) and not grpo.patience_stop([], 3)
    assert grpo.stop_status(0, 0.0, base + [ev(0.19, 0.16)], 3) == "stopped_patience"
    assert grpo.RULE_STOPS == ("stopped_collapse", "stopped_truncation", "stopped_patience")


# ---------------------------------------------------------------- batch diagnostics
KEYS = [["A", "A", "B"], ["A", None, "C"], [None, None, None]]       # three groups; the last has no valid answer


def test_collapse_score_and_collapse_all_differ_only_by_their_denominator():
    assert grpo.modal_keys(KEYS) == ["A", "A"]                         # group 2 ties A against C: the smallest key
    assert grpo.collapse_score(KEYS) == pytest.approx(1 / 2)           # 1 distinct modal answer over the 2 groups with a valid answer
    assert grpo.collapse_all(KEYS) == pytest.approx(1 / 3)             # over all 3 groups
    keys = [["A"], ["B"], [None]]
    assert grpo.collapse_score(keys) == pytest.approx(1.0) and grpo.collapse_all(keys) == pytest.approx(2 / 3)
    assert grpo.collapse_all([[None], [None]]) == 0.0 and grpo.collapse_score([[None], [None]]) is None
    assert grpo.collapse_all([]) is None


def test_answer_shares_over_the_whole_batch():
    assert grpo.top_answer_share(KEYS) == pytest.approx(3 / 5) and grpo.distinct_answer_share(KEYS) == pytest.approx(3 / 5)
    spread = [["A", "A", "B", "B"], ["A", "C", "D", "E"]]              # 8 valid answers, A three times, five distinct keys
    assert grpo.top_answer_share(spread) == pytest.approx(3 / 8) and grpo.distinct_answer_share(spread) == pytest.approx(5 / 8)
    assert grpo.top_answer_share([["Z"] * 4]) == 1.0 and grpo.distinct_answer_share([["Z"] * 4]) == pytest.approx(1 / 4)
    assert grpo.top_answer_share([[None], []]) is None and grpo.distinct_answer_share([]) is None


def test_copy_share_counts_valid_answers_that_are_train_references():
    assert grpo.copy_share(KEYS, {"A", "C"}) == pytest.approx(4 / 5) and grpo.copy_share(KEYS, frozenset({"B"})) == pytest.approx(1 / 5)
    assert grpo.copy_share(KEYS, set()) == 0.0 and grpo.copy_share([[None]], {"A"}) is None


def test_copy_share_leaves_out_an_answer_equal_to_the_drugs_own_reference():
    keys = [["A", "A", "B"], ["A", None, "C"]]                           # 5 valid answers in two groups
    refs = {"A", "B", "C"}                                               # the Train references
    own = [{"A"}, {"C"}]                                                 # group 0's reference is A, group 1's is C
    assert grpo.copy_share(keys, refs, own) == pytest.approx(2 / 5)      # B in group 0 and A in group 1 copy another drug; A, A and C are recall
    assert grpo.copy_share(keys, refs) == pytest.approx(5 / 5) == grpo.copy_share(keys, refs, [set(), set()])     # without `own`: all five count
    assert grpo.copy_share([["A"]], {"A"}, [{"A"}]) == 0.0               # the own hit stays in the denominator
    assert grpo.copy_share([["A", "B"]], {"A", "B"}, [{"A", "B"}]) == 0.0           # a reference with alternates: each of them is the drug's own
    assert grpo.copy_share([[None]], {"A"}, [{"A"}]) is None


def test_similarity_streak_counts_trailing_steps_whose_modal_answers_are_near_duplicates():
    step = lambda value: dict(modal_pairwise_tanimoto=value)
    assert (grpo.SIMILAR_ABOVE, grpo.SIMILAR_STEPS) == (0.5, 3)
    assert grpo.similarity_streak([]) == 0
    assert grpo.similarity_streak([step(0.6), step(0.7), step(0.9)]) == 3
    assert grpo.similarity_streak([step(0.9), step(0.4), step(0.6), step(0.7)]) == 2          # a low step ends the run of high ones
    assert grpo.similarity_streak([step(0.6), step(0.7), step(0.5)]) == 0                      # exactly 0.5 is not above it
    assert grpo.similarity_streak([step(0.6), step(None), step(0.7)]) == 1                     # a step without two modal answers ends it too
    assert grpo.similarity_streak([step(0.51)] * 7) == 7 and grpo.similarity_streak([dict(step=1)]) == 0       # an older log row has no such key


def test_modal_answers_and_mean_pairwise_similarity():
    keys = [["k1", "k1", "k2"], ["k3", None], [None, None], ["k2", "k2"]]
    answers = [["a1", "a1x", "a2"], ["a3", None], [None, None], ["a2", "a2y"]]
    assert grpo.modal_answers(keys, answers) == ["a1", "a3", "a2"]     # the first sample with the modal key; the empty group is skipped
    table = {("a1", "a3"): 0.2, ("a1", "a2"): 0.4, ("a3", "a2"): 0.9}
    similarity = lambda x, y: table[(x, y)] if (x, y) in table else table[(y, x)]
    assert grpo.mean_pairwise(["a1", "a3", "a2"], similarity) == pytest.approx((0.2 + 0.4 + 0.9) / 3)
    assert grpo.mean_pairwise(["a1"], similarity) is None and grpo.mean_pairwise([], similarity) is None
    assert grpo.mean_pairwise(["a", "a"], lambda x, y: 1.0 if x == y else 0.0) == 1.0     # one shared modal answer: fully similar


def test_reward_components_are_batch_means_of_the_rewards_terms():
    samples = [sample(1.0, conn=True, tani=1.0, scaffold=True), sample(0.45, tani=0.5, scaffold=True),
               sample(0.0, valid=False, tani=0.0), sample(0.1, tani=0.1)]
    got = grpo.reward_components(samples, 0.05, 0.6, 0.3)
    assert got == pytest.approx(dict(reward_validity_term=0.05 * 3 / 4, reward_tanimoto_term=0.6 * 1.6 / 4,
                                     reward_scaffold_term=0.3 * 2 / 4, reward_identity=1 / 4))
    with pytest.raises(ValueError):
        grpo.reward_components([], 0.05, 0.6, 0.3)


LIFT_TABLE = {("a0", "r0"): 0.8, ("a0", "r1"): 0.2, ("a0", "r2"): 0.1, ("a1", "r0"): 0.3, ("a1", "r1"): 0.5, ("a1", "r2"): 0.1,
              ("a2", "r0"): 0.2, ("a2", "r1"): 0.2, ("a2", "r2"): 0.2}


def test_profile_lift_by_hand_subtracts_the_mean_over_the_other_drugs_only():
    similarity = lambda answer, ref: LIFT_TABLE[(answer, ref)]
    # drug 0: 0.8 - mean(0.2, 0.1) = 0.65. drug 1: 0.5 - mean(0.3, 0.1) = 0.30. drug 2: a2 is 0.2 - mean(0.2, 0.2) = 0, the None counts 0
    got = grpo.profile_lift(["r0", "r1", "r2"], [["a0"], ["a1"], ["a2", None]], similarity)
    assert got == pytest.approx((0.65 + 0.30 + 0.0) / 3)
    # an invalid answer counts 0 and stays in its drug's mean: drug 0 with one good and one invalid sample halves to 0.325
    assert grpo.profile_lift(["r0", "r1", "r2"], [["a0", None], ["a1"], ["a2"]], similarity) == pytest.approx((0.325 + 0.30 + 0.0) / 3)
    assert grpo.profile_lift(["r0"], [["a0"]], similarity) is None and grpo.profile_lift([], [], similarity) is None


def test_a_constant_answer_has_zero_lift_whatever_the_similarities():
    rng = random.Random(5)
    for drugs in (2, 3, 7, 32):
        refs = [f"r{d}" for d in range(drugs)]
        table = {ref: rng.random() for ref in refs}                    # similarity of the one constant answer to each reference
        similarity = lambda answer, ref: table[ref]
        for valid, total in ((4, 4), (3, 4), (1, 8), (0, 4)):         # the same share of valid samples for every drug
            answers = [["A"] * valid + [None] * (total - valid) for _ in refs]
            assert abs(grpo.profile_lift(refs, answers, similarity)) < 1e-12, (drugs, valid, total)
    # a policy that tells the drugs apart does score above 0: each drug gets the answer closest to its own reference
    refs = ["r0", "r1", "r2"]
    tailored = {("a0", "r0"): 0.9, ("a1", "r1"): 0.9, ("a2", "r2"): 0.9}
    assert grpo.profile_lift(refs, [["a0"], ["a1"], ["a2"]], lambda a, r: tailored.get((a, r), 0.1)) == pytest.approx(0.8)


def test_drug_lift_pools_the_rows_of_a_drug():
    similarity = lambda answer, ref: LIFT_TABLE[(answer, ref)]
    samples = [dict(drug="x", ref="r0", answer="a0"), dict(drug="y", ref="r1", answer="a1"), dict(drug="x", ref="r0", answer=None),
               dict(drug="y", ref="r1", answer="a1")]
    # x: (0.8 - 0.2, 0) -> 0.30. y: 0.5 - 0.3 = 0.20 twice -> 0.20. the mean over the two drugs
    assert grpo.drug_lift(samples, similarity) == pytest.approx(0.25)
    assert grpo.drug_lift(samples[:1], similarity) is None


def test_drug_balanced_adds_validity_valid_only_similarity_and_the_full_reward():
    samples = [sample(1.0, "x", conn=True, tani=1.0), sample(0.3, "x", tani=0.3), sample(0.0, "x", valid=False, tani=0.0),
               sample(0.0, "y", valid=False, tani=0.0), sample(0.0, "y", valid=False, tani=0.0), sample(0.2, "z", tani=0.2)]
    out = grpo.drug_balanced(samples)
    assert out["validity"] == pytest.approx((2 / 3 + 0 + 1) / 3)
    assert out["tanimoto"] == pytest.approx((1.3 / 3 + 0 + 0.2) / 3)                  # every attempt, invalid = 0
    assert out["tanimoto_valid"] == pytest.approx((0.65 + 0.2) / 2)                   # y has no valid answer and is left out
    assert out["reward"] == pytest.approx(out["reward_full"])                         # no reward_full: the same
    assert grpo.drug_balanced([sample(0.0, "y", valid=False, tani=0.0)])["tanimoto_valid"] is None


def test_summarize_and_drug_balanced_carry_the_full_reward_beside_the_training_reward():
    samples = [dict(sample(1.0, "x"), reward_full=0.4), dict(sample(0.0, "x"), reward_full=0.2)]
    got = grpo.summarize(samples)
    assert got["reward_mean"] == 0.5 and got["reward_full_mean"] == pytest.approx(0.3) and got["reward_full_std"] == pytest.approx(0.1)
    assert got["validity"] == 1.0
    assert grpo.drug_balanced(samples)["reward"] == 0.5 and grpo.drug_balanced(samples)["reward_full"] == pytest.approx(0.3)
    assert grpo.summarize([sample(0.2)])["reward_full_mean"] == pytest.approx(0.2)


def test_the_collapse_rule_can_be_switched_off_but_the_other_rules_stay():
    assert grpo.stop_status(3, 0.0, [], 3) == "stopped_collapse"
    assert grpo.stop_status(3, 0.0, [], 3, collapse_stop=False) is None
    assert grpo.stop_status(3, 0.31, [], 3, collapse_stop=False) == "stopped_truncation"
    patience = [ev(0.10, None), ev(0.20, 0.10), ev(0.19, 0.12), ev(0.18, 0.14), ev(0.19, 0.16)]
    assert grpo.stop_status(3, 0.0, patience, 3, collapse_stop=False) == "stopped_patience"


# ---------------------------------------------------------------- the contrastive reward
def test_mean_others_leaves_the_own_drug_out():
    assert grpo.mean_others([0.2, 0.4, 0.9], 1) == pytest.approx(0.55) and grpo.mean_others([0.2, 0.4, 0.9], 0) == pytest.approx(0.65)
    assert grpo.mean_others([0.3, 0.7], 0) == pytest.approx(0.7) and grpo.mean_others([0.3, 0.7], 1) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="at least two"):
        grpo.mean_others([0.5], 0)


def test_the_contrastive_reward_subtracts_the_others_and_keeps_unusable_samples_at_zero():
    assert grpo.contrastive_reward(0.7, 0.25, True, 0.6) == pytest.approx(0.55)               # 0.7 - 0.6 * 0.25: the sign is a subtraction
    assert grpo.contrastive_reward(1.0, 0.0, True, 0.6) == 1.0
    assert grpo.contrastive_reward(0.31, 0.5, True, 0.6) == pytest.approx(0.01)               # just above the floor: untouched
    # an invalid, unclosed or length-terminated sample is 0 whatever the full reward and the others say
    assert grpo.contrastive_reward(0.7, 0.25, False, 0.6) == 0.0 and grpo.contrastive_reward(0.7, 0.0, False, 0.6) == 0.0


def test_the_contrastive_reward_is_floored_at_zero_so_abstention_does_not_pay():
    # closer to the other drugs than to its own would be 0.1 - 0.6 * 0.5 = -0.2 and below a failure (0): the floor makes it a tie with failure
    assert grpo.contrastive_reward(0.1, 0.5, True, 0.6) == 0.0
    assert grpo.contrastive_reward(0.05, 0.9, True, 0.6) == 0.0 and grpo.contrastive_reward(0.0, 0.0, True, 0.6) == 0.0
    for full in (0.05, 0.2, 0.55, 1.0):                                                        # no usable answer ever scores below a failure
        for others in (0.0, 0.3, 0.8, 1.0):
            assert grpo.contrastive_reward(full, others, True, 0.6) >= grpo.contrastive_reward(full, others, False, 0.6) == 0.0
    # in a group the usable-but-bad answer and the failure get the same reward, so abstaining gains nothing over it
    rewards = grpo.contrastive_rewards([0.1, 0.0, 0.8], [True, False, True], [0, 0, 0], [[0.1, 0.6], None, [0.9, 0.2]], 0.6)
    assert rewards[0] == rewards[1] == 0.0 and rewards[2] == pytest.approx(0.8 - 0.6 * 0.2)


def test_contrastive_rewards_of_a_batch_use_each_samples_own_drug_and_force_unusable_ones_to_zero():
    sims = [[0.5, 0.1, 0.3], None, [0.2, 0.4, 0.0], [0.1, 0.2, 0.6]]
    got = grpo.contrastive_rewards([0.8, 0.7, 0.3, 1.0], [True, False, True, True], [0, 1, 1, 2], sims, 0.6)
    assert got == pytest.approx([0.8 - 0.6 * (0.1 + 0.3) / 2, 0.0, 0.3 - 0.6 * (0.2 + 0.0) / 2, 1.0 - 0.6 * (0.1 + 0.2) / 2])
    assert grpo.contrastive_rewards([], [], [], [], 0.6) == []
    with pytest.raises(ValueError):
        grpo.contrastive_rewards([0.8], [True], [0], [[0.5]], 0.6)                           # one drug has no others
    # a sample closer to the other drugs than to its own is floored; the same answer to the drug it resembles is not
    assert grpo.contrastive_rewards([0.2, 0.2], [True, True], [0, 1], [[0.1, 0.6], [0.1, 0.6]], 0.6) == pytest.approx([0.0, 0.2 - 0.6 * 0.1])


def constant_answer_batch(rng, drugs, valid):
    """One constant answer given to every drug: per drug its similarity t, scaffold flag s and identity flag i (an exact copy of the
    reference has t = 1), the full reward, and the contrastive rewards with `valid` of 4 samples usable per drug."""
    ident = [rng.random() < 0.25 for _ in range(drugs)]
    table = [1.0 if i else rng.random() * 0.9 for i in ident]
    scaffold = [rng.random() < 0.4 for _ in range(drugs)]
    fulls = [1.0 if i else 0.05 + 0.6 * t + 0.3 * s for i, t, s in zip(ident, table, scaffold)]       # reward_v2
    owns = [d for d in range(drugs) for _ in range(4)]
    usable = [k % 4 < valid for k in range(len(owns))]
    got = grpo.contrastive_rewards([fulls[d] if u else 0.0 for d, u in zip(owns, usable)], usable, owns, [table if u else None for u in usable], 0.6)
    return ident, table, scaffold, fulls, owns, usable, got


def test_a_constant_answer_keeps_its_validity_scaffold_and_identity_terms_and_loses_its_similarity_terms():
    rng = random.Random(3)
    for drugs in (2, 3, 8, 32):
        for valid in (4, 3, 1):                                       # the same share of valid samples for every drug
            ident, table, scaffold, fulls, owns, usable, got = constant_answer_batch(rng, drugs, valid)
            share = valid / 4
            # before the floor: 0.05 + 0.3 * mean(s) + mean(i * (1 - 0.05 - 0.6 t - 0.3 s)), times the share of usable samples
            exact = 0.05 + 0.3 * sum(scaffold) / drugs + sum(i * (1 - 0.05 - 0.6 * t - 0.3 * s) for i, t, s in zip(ident, table, scaffold)) / drugs
            plain = [grpo.contrastive_reward(fulls[d], grpo.mean_others(table, d), True, 0.6) for d in range(drugs)]        # the floored values
            unfloored = [fulls[d] - 0.6 * grpo.mean_others(table, d) for d in range(drugs)]
            assert sum(unfloored) / drugs == pytest.approx(exact, abs=1e-12), (drugs, valid)           # the similarity terms cancel exactly
            assert sum(got) / len(got) == pytest.approx(share * sum(plain) / drugs, abs=1e-12)
            assert sum(got) / len(got) >= share * exact - 1e-12                                       # the floor can only add
            full_mean = sum(fulls[d] for d, u in zip(owns, usable) if u) / len(got)
            assert full_mean - share * sum(unfloored) / drugs == pytest.approx(0.6 * share * sum(table) / drugs, abs=1e-12)    # what the full reward pays
    # a plain acyclic constant (no scaffold bonus, no identity): exactly the validity term, as long as the floor does not bite
    flat = [0.2, 0.22, 0.18, 0.2]
    got = grpo.contrastive_rewards([0.05 + 0.6 * t for t in flat], [True] * 4, [0, 1, 2, 3], [flat] * 4, 0.6)
    assert sum(got) / 4 == pytest.approx(0.05, abs=1e-12) and min(got) > 0


def test_the_floor_adds_the_clipped_negative_parts_to_a_constant_answers_mean():
    sims = [0.9, 0.1, 0.1, 0.1]                                       # one drug the constant resembles, three it does not
    fulls = [0.05 + 0.6 * t for t in sims]
    got = grpo.contrastive_rewards(fulls, [True] * 4, [0, 1, 2, 3], [sims] * 4, 0.6)
    raw = [f - 0.6 * grpo.mean_others(sims, d) for d, f in enumerate(fulls)]
    assert sum(raw) / 4 == pytest.approx(0.05, abs=1e-12) and min(raw) < 0                       # unfloored: exactly the validity term
    assert got == pytest.approx([max(0.0, r) for r in raw]) and min(got) == 0.0
    assert sum(got) / 4 == pytest.approx(0.05 + sum(-r for r in raw if r < 0) / 4) and sum(got) / 4 > 0.05


def test_an_answer_specific_to_its_drug_beats_the_same_answer_given_to_another_drug():
    sims = [0.7, 0.1, 0.1, 0.1]                                       # the answer resembles drug 0's reference
    full = lambda d: 0.05 + 0.6 * sims[d]
    to_drug_0, to_drug_1 = (grpo.contrastive_rewards([full(d)], [True], [d], [sims], 0.6)[0] for d in (0, 1))
    assert to_drug_0 == pytest.approx(0.05 + 0.6 * 0.7 - 0.6 * 0.1)
    assert 0.05 + 0.6 * 0.1 - 0.6 * 0.3 < 0 and to_drug_1 == 0.0      # given to the wrong drug it would be -0.07: the floor makes it a failure
    assert to_drug_0 > 0.05 > to_drug_1


# ---------------------------------------------------------------- the Val first-valid policy
def test_usable_is_valid_closed_and_not_length_terminated():
    ok = dict(valid=True, closed=True, truncated=False)
    assert grpo.usable(ok) is True
    for damage in (dict(valid=False), dict(closed=False), dict(truncated=True)):
        assert grpo.usable(dict(ok, **damage)) is False


def test_answer_lift_is_the_own_similarity_minus_the_mean_over_the_other_references():
    similarity = lambda answer, ref: LIFT_TABLE[(answer, ref)]
    assert grpo.answer_lift("a0", 0, ["r0", "r1", "r2"], similarity) == pytest.approx(0.8 - (0.2 + 0.1) / 2)
    assert grpo.answer_lift("a0", 1, ["r0", "r1", "r2"], similarity) == pytest.approx(0.2 - (0.8 + 0.1) / 2)
    assert grpo.answer_lift("a2", 2, ["r0", "r1", "r2"], similarity) == pytest.approx(0.0)


def fv(drug, prompt, answer, tani, **flags):
    """A Val sample of `prompt` (a row of `drug`); a usable one unless `flags` damage it."""
    return dict(drug=drug, prompt=prompt, answer=answer, tanimoto=tani, ref=dict(x="r0", y="r1", z="r2")[drug],
                **dict(dict(valid=True, closed=True, truncated=False), **flags))


def test_the_first_valid_policy_takes_the_first_usable_sample_of_each_row_and_weighs_each_drug_once():
    similarity = lambda answer, ref: LIFT_TABLE[(answer, ref)]
    samples = [
        # x, row p1: an invalid answer, a length-terminated one, an unclosed one, then the first usable sample (the best one comes later)
        fv("x", "p1", None, 0.0, valid=False), fv("x", "p1", "a1", 0.9, truncated=True), fv("x", "p1", "a1", 0.9, closed=False),
        fv("x", "p1", "a0", 0.8), fv("x", "p1", "a1", 0.95),
        # x, row p2: nothing usable in all 5 samples
        *[fv("x", "p2", None, 0.0, valid=False) for _ in range(5)],
        # y, row p3: the very first sample is usable
        fv("y", "p3", "a1", 0.5), fv("y", "p3", "a0", 0.99), fv("y", "p3", None, 0.0, valid=False), fv("y", "p3", None, 0.0, valid=False),
        fv("y", "p3", None, 0.0, valid=False),
        # z, row p4: the last sample is the first usable one
        *[fv("z", "p4", None, 0.0, valid=False) for _ in range(4)], fv("z", "p4", "a2", 0.2)]
    got = grpo.first_valid_policy(samples, similarity)
    assert list(got) == ["fv_tanimoto", "fv_lift", "fv_rate", "fv_samples"] == list(grpo.FV_FIELDS)
    # x: row p1 -> tanimoto 0.8, lift 0.8 - mean(0.2, 0.1) = 0.65, found, 4 drawn. row p2 -> 0, 0, not found, 5 drawn. the drug mean of both rows
    x = dict(fv_tanimoto=0.4, fv_lift=0.325, fv_rate=0.5, fv_samples=4.5)
    y = dict(fv_tanimoto=0.5, fv_lift=0.5 - (0.3 + 0.1) / 2, fv_rate=1.0, fv_samples=1.0)
    z = dict(fv_tanimoto=0.2, fv_lift=0.0, fv_rate=1.0, fv_samples=5.0)          # a2 is 0.2 against every reference: lift 0
    for key in grpo.FV_FIELDS:
        assert got[key] == pytest.approx((x[key] + y[key] + z[key]) / 3), key
    assert all(type(value) is float for value in got.values())


def test_the_first_valid_policy_edge_cases():
    similarity = lambda answer, ref: LIFT_TABLE[(answer, ref)]
    nothing = grpo.first_valid_policy([fv("x", "p1", None, 0.0, valid=False)] * 4 + [fv("y", "p2", None, 0.0, closed=False)] * 4, similarity)
    assert nothing == dict(fv_tanimoto=0.0, fv_lift=0.0, fv_rate=0.0, fv_samples=4.0)         # k draws when none qualifies
    one_drug = grpo.first_valid_policy([fv("x", "p1", "a0", 0.8), fv("x", "p1", "a0", 0.8)], similarity)
    assert one_drug == dict(fv_tanimoto=0.8, fv_lift=0.0, fv_rate=1.0, fv_samples=1.0)         # a single drug has no others: lift 0, as in metrics.py
    # a drug with more rows does not outweigh another: 3 rows of x against 1 of y
    rows = [fv("x", f"p{i}", "a0", 1.0 if i == 0 else 0.0, valid=i == 0) for i in range(3)] + [fv("y", "q", "a1", 0.5)]
    assert grpo.first_valid_policy(rows, similarity)["fv_tanimoto"] == pytest.approx((1 / 3 + 0.5) / 2)
    with pytest.raises(ValueError):
        grpo.first_valid_policy([], similarity)

