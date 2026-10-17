"""GRPO bookkeeping for the answer-only RL pilot (gpu/RL.md, docs/plan.md sections 2 and 6). Pure numpy and stdlib: no torch, no chemistry.

    A_i = r_i - mean_g(r)                       centred on every sample of the group. A length-terminated sample is a failure with
                                                reward 0: it stays in its group and in the update, never masked
    group dropped when max(r) - min(r) <= 1e-9  no gradient, and not counted in G
    A_i /= std(all retained A) + 1e-6           one std for the whole batch (ScaleRL), never the per-group std, which would
                                                inflate groups whose rewards differ only by noise
    w_i = A_i / (T_g * G)                       coefficient of every token of completion i: T_g completion tokens in group g,
                                                G retained groups, so a prompt weighs the same however long its completions are
    loss = - sum_i w_i * sum_t log pi(y_it)     one update per rollout batch, so the importance ratio is 1 and CISPO's truncated
                                                weight is 1: REINFORCE with a group baseline

    collapse_score = distinct modal answers / groups that have a valid answer    healthy near 1; alarm below 0.7; the stop rule uses it
    collapse_all   = distinct modal answers / all groups                         logged only

The contrastive reward (--reward-mode contrastive) takes the generic-answer basin out of the full reward: for a valid, closed, finished answer
to drug d in a batch of drugs B it is max(0, reward_v2 - w * mean over the OTHER drugs d' in B of T(answer, ref_d')), w the weight of the
reward's similarity term, and 0 for every other sample. The floor at 0 stops abstention from paying: a usable answer that sits closer to the
other drugs than to its own would otherwise score below a failure (0), and the update would prefer failing. With it a bad answer ties with
failure, and the gradient separates the answers better than the generic one from the rest. The similarity terms of a constant answer cancel
exactly over a batch (before the floor), the other terms do not: it averages the validity term 0.05, plus 0.3 times the share of drugs whose
scaffold it has, plus 1 - 0.05 - 0.6 T - 0.3 s for each drug it reproduces exactly (T its Tanimoto there, s its scaffold flag), plus whatever
the floor adds. An answer specific to the profile earns more. It depends on the whole batch, so it is recomputed whenever the drug set changes.

Diagnostics of a batch cover every attempted group, before the zero-variance filter: the share of valid answers that carry the most
common key, the share of distinct keys, the share that copies a Train reference, the mean similarity between the groups' modal answers,
the batch means of the reward's terms, and the profile-specific lift of docs/plan.md section 2. The similarity is passed in, so none of
it needs a chemistry library here. A Val sweep also reports the first-valid policy of metrics.py v3 (fv_tanimoto, fv_lift, fv_rate,
fv_samples); fv_tanimoto, the endpoint, is what the patience rule follows, not the Val mean reward.

The drug schedule is a stream of seeded permutations, one per pass over the Train drugs; its cursor (a position in the stream) is
the whole state, so a resumed run draws exactly what an uninterrupted run would. Stop rules and per-step aggregates live here too,
so they are tested without a model. A step's standard deviation is the population one (ddof 0); the pooled centred advantages
have mean zero, so it is the root mean square.
"""
import math
import random
from collections import Counter

import numpy as np

TOL, EPS = 1e-9, 1e-6             # zero-variance tolerance; added to the batch std
WARMUP_STEPS = 5                  # linear learning-rate warmup, then constant
COLLAPSE_BELOW, COLLAPSE_STEPS = 0.7, 3     # stop when the collapse score is below 0.7 for 3 consecutive steps
TRUNCATION_ABOVE = 0.30           # stop when more than 30% of a step's samples hit the length cap
RULE_STOPS = ("stopped_collapse", "stopped_truncation", "stopped_patience")    # final: a resume does not override them


# ---------------------------------------------------------------- advantages and token weights
def group_advantages(rewards, tol=TOL, eps=EPS):
    """rewards: one sequence of rewards per group (every sample, length-terminated ones included as zeros).
    Returns (advantages, std). advantages[g] is None for a zero-variance group, else the centred rewards divided by std + eps,
    std being the standard deviation of all retained centred advantages (None when no group is retained)."""
    centred = []
    for r in rewards:
        r = np.asarray(r, dtype=np.float64)
        if r.size == 0 or not np.isfinite(r).all():
            raise ValueError("a group needs finite rewards")
        centred.append(None if r.max() - r.min() <= tol else r - r.mean())
    kept = [a for a in centred if a is not None]
    if not kept:
        return centred, None
    std = float(np.concatenate(kept).std())
    return [None if a is None else a / (std + eps) for a in centred], std


def token_weights(advantages, lengths):
    """Coefficient of every token of every completion: A_i / (T_g * G). advantages[g] is None for a dropped group;
    lengths[g] holds the completion token counts of group g. Returns (weights, G); weights[g] is None for dropped groups."""
    groups = sum(a is not None for a in advantages)
    weights = []
    for a, n in zip(advantages, lengths):
        if a is None:
            weights.append(None)
            continue
        n = np.asarray(n, dtype=np.float64)
        if n.shape != a.shape or (n < 1).any():
            raise ValueError("every completion needs at least one token")
        weights.append(a / (n.sum() * groups))
    return weights, groups


def plan_update(rewards, lengths):
    """Everything the update needs: weights per group (None = dropped), G, the batch std and the zero-variance count."""
    advantages, std = group_advantages(rewards)
    weights, groups = token_weights(advantages, lengths)
    return dict(weights=weights, groups=groups, std=std, zero_variance=len(rewards) - groups)


def refill_count(kept, prompts, min_groups):
    """Extra prompts to draw when fewer than min_groups groups survive the zero-variance filter: enough, at the observed survival
    rate, to reach min_groups, but never more than the original prompts (the batch at most doubles)."""
    if kept >= min_groups:
        return 0
    return min(prompts, math.ceil((min_groups - kept) * prompts / max(kept, 1)))


def skip_below(min_groups):
    """An update with fewer retained groups than this is skipped: half of min_groups rounded UP, so never below half (8 for the default 16,
    2 for 3, 3 for 5)."""
    return max(1, (min_groups + 1) // 2)


def lr_at(step, lr):
    """Learning rate of the 1-based step: linear warmup over WARMUP_STEPS, then constant."""
    return lr * min(1.0, step / WARMUP_STEPS)


# ---------------------------------------------------------------- the drug schedule
def pass_order(n, seed, k):
    """Drug indices of pass k over the data: a permutation seeded by (seed, k)."""
    order = list(range(n))
    random.Random(seed * 1_000_003 + k).shuffle(order)
    return order


def draw(sizes, seed, cursor, count, exclude=()):
    """Next `count` distinct drugs of the schedule, skipping any in `exclude` (drugs already in this step).
    sizes[d] is the number of views (rows) of drug d. The schedule is the stream pass_order(n, seed, 0) + pass_order(n, seed, 1)
    + ...; the cursor is a position in it. A drug skipped because it is already in the step is consumed, not deferred, which can
    only happen where one pass ends and the next begins. The view of the drug at stream position c is
    random.Random(f"view:{seed}:{c}").randrange(sizes[d]), so a drug can show another view in its next pass.
    Returns ([(drug, view), ...], new cursor)."""
    n = len(sizes)
    taken = set(exclude)
    count = max(0, min(count, n - len(taken)))
    picks, order, loaded = [], None, None
    while len(picks) < count:
        k, i = divmod(cursor, n)
        if loaded != k:
            order, loaded = pass_order(n, seed, k), k
        d = order[i]
        if d not in taken:
            taken.add(d)
            picks.append((d, random.Random(f"view:{seed}:{cursor}").randrange(sizes[d])))
        cursor += 1
    return picks, cursor


def rollout_seed(seed, step, refill=0):
    """Torch seed of one generation call: seed * 1_000_003 + step. The refill call of the same step adds 2**40."""
    return seed * 1_000_003 + step + (refill << 40)


def eval_seed(seed, step):
    """Torch seed of the Val sampling after `step` steps: the stream a rollout never uses (refill index 2)."""
    return rollout_seed(seed, step, 2)


# ---------------------------------------------------------------- answers of a batch: collapse and diagnostics
def _modal(group):
    """The most frequent key of one group (ties: the smallest key), or None when no sample has a valid answer."""
    counts = Counter(k for k in group if k is not None)
    if not counts:
        return None
    top = max(counts.values())
    return min(k for k, c in counts.items() if c == top)


def modal_keys(keys):
    """keys: for each group, the connectivity key of each sample's answer (None: no valid, finished answer).
    The modal key of every group that has a valid answer."""
    return [m for m in map(_modal, keys) if m is not None]


def collapse_score(keys):
    """Distinct modal answers divided by the number of groups with at least one valid answer (so a group without one does not
    count), or None when no group has one. Healthy near 1; a policy that gives every prompt the same answer scores 1 / groups."""
    modal = modal_keys(keys)
    return len(set(modal)) / len(modal) if modal else None


def collapse_all(keys):
    """Distinct modal answers divided by ALL groups: a group without a valid answer counts as a group that did not diversify."""
    return len(set(modal_keys(keys))) / len(keys) if keys else None


def modal_answers(keys, answers):
    """The answer string behind each modal key (the first sample that carries it); groups without a valid answer are skipped."""
    out = []
    for group_keys, group_answers in zip(keys, answers):
        modal = _modal(group_keys)
        if modal is not None:
            out.append(next(a for k, a in zip(group_keys, group_answers) if k == modal))
    return out


def mean_pairwise(items, similarity):
    """Mean of similarity(a, b) over all pairs of different positions; None when there are fewer than two items."""
    pairs = [(a, b) for i, a in enumerate(items) for b in items[i + 1:]]
    return math.fsum(similarity(a, b) for a, b in pairs) / len(pairs) if pairs else None


def top_answer_share(keys):
    """Share of the valid answers of the whole batch that carry its most common key; None without a valid answer."""
    valid = [k for group in keys for k in group if k is not None]
    return max(Counter(valid).values()) / len(valid) if valid else None


def distinct_answer_share(keys):
    """Distinct keys divided by valid answers over the whole batch; None without a valid answer."""
    valid = [k for group in keys for k in group if k is not None]
    return len(set(valid)) / len(valid) if valid else None


def copy_share(keys, reference_keys):
    """Share of the valid answers whose key is the key of a reference in `reference_keys` (a Train structure copied); None
    without a valid answer."""
    valid = [k for group in keys for k in group if k is not None]
    return sum(k in reference_keys for k in valid) / len(valid) if valid else None


def reward_components(samples, w_valid, w_tani, w_scaf):
    """Batch means of the reward's parts over all samples (an invalid, unclosed or length-terminated sample counts 0 in each): the
    validity term w_valid, the similarity term w_tani * tanimoto, the scaffold term w_scaf and the identity indicator. The first three
    are the reward of a non-identity answer; an identity answer's reward is 1.0."""
    n = len(samples)
    if n == 0:
        raise ValueError("no samples")
    mean = lambda f: math.fsum(f(s) for s in samples) / n
    return dict(reward_validity_term=mean(lambda s: w_valid * bool(s["valid"])),
                reward_tanimoto_term=mean(lambda s: w_tani * s["tanimoto"]),
                reward_scaffold_term=mean(lambda s: w_scaf * bool(s["scaffold_match"])),
                reward_identity=mean(lambda s: float(bool(s["conn_match"]))))


def answer_lift(answer, own, refs, similarity):
    """Profile-specific lift of one answer to the drug at position `own` of `refs` (at least two): its similarity to that reference minus
    its mean similarity to the other references."""
    sims = [similarity(answer, r) for r in refs]
    return sims[own] - (math.fsum(sims) - sims[own]) / (len(refs) - 1)


def profile_lift(refs, answers, similarity):
    """Profile-specific lift (docs/plan.md section 2). refs[d] is the reference of drug d; answers[d] lists its samples' answers, None
    for an invalid, unclosed or length-terminated one. A valid answer a to drug d scores similarity(a, refs[d]) minus the mean of
    similarity(a, refs[d']) over the OTHER drugs d' (the own reference is not in that mean); None scores 0; the mean over a drug's
    samples, then over drugs. A constant answer that is valid for the same share of every drug's samples scores 0: its own-versus-others
    terms cancel across drugs (up to rounding). None for fewer than two drugs."""
    drugs = len(refs)
    if drugs < 2:
        return None
    per_drug = []
    for d, group in enumerate(answers):
        values = [0.0 if a is None else answer_lift(a, d, refs, similarity) for a in group]
        per_drug.append(math.fsum(values) / len(values) if values else 0.0)
    return math.fsum(per_drug) / drugs


def drug_lift(samples, similarity):
    """profile_lift of samples that carry `drug`, `ref` (that drug's reference) and `answer` (None: no valid finished answer)."""
    refs, answers = {}, {}
    for s in samples:
        refs.setdefault(s["drug"], s["ref"])
        answers.setdefault(s["drug"], []).append(s["answer"])
    return profile_lift(list(refs.values()), [answers[d] for d in refs], similarity)


# ---------------------------------------------------------------- the contrastive reward
def mean_others(sims, own):
    """Mean of sims[j] over every position but `own`: the similarity of one answer to the references of the other drugs of the batch.
    ValueError with fewer than two drugs, where there are no others."""
    if len(sims) < 2:
        raise ValueError("the contrastive reward needs at least two drugs")
    return math.fsum(s for j, s in enumerate(sims) if j != own) / (len(sims) - 1)


def contrastive_reward(full, others, usable, weight):
    """max(0, reward_v2 - weight * others) for a usable (valid, closed, finished) answer; an invalid, unclosed or length-terminated one
    keeps 0. The floor is what stops abstention from paying: no usable answer scores below a failure, so a bad answer ties with
    failure and only the answers that beat the generic one rise above it."""
    return max(0.0, full - weight * others) if usable else 0.0


def contrastive_rewards(fulls, usable, owns, sims, weight):
    """The contrastive reward of every sample of a batch. fulls[i]: its full reward; usable[i]: valid, closed and finished; owns[i]: the
    index of its drug among the batch's references; sims[i]: T(answer_i, ref_j) for every reference j of the batch (None when unusable).
    The "others" of sample i are every reference but its own drug's."""
    return [contrastive_reward(f, mean_others(row, own) if ok else 0.0, ok, weight) for f, ok, own, row in zip(fulls, usable, owns, sims)]


# ---------------------------------------------------------------- per-step and per-eval aggregates
def summarize(samples):
    """Aggregate per-sample dicts (reward, valid, closed, conn_match, scaffold_match, tanimoto, n_tokens, truncated; reward_full when
    the training reward differs from the full one) into plain floats. length_p90 is the nearest-rank 90th percentile."""
    n = len(samples)
    if n == 0:
        raise ValueError("no samples")
    col = lambda key: np.array([float(s[key]) for s in samples])
    lengths = np.sort(col("n_tokens"))
    reward, full = col("reward"), np.array([float(s.get("reward_full", s["reward"])) for s in samples])
    return dict(samples=n, reward_mean=float(reward.mean()), reward_std=float(reward.std()),
                reward_full_mean=float(full.mean()), reward_full_std=float(full.std()),
                valid=float(col("valid").mean()), closed=float(col("closed").mean()), validity=float(col("valid").mean()),
                valid_closed=float(np.mean([bool(s["valid"]) and bool(s["closed"]) for s in samples])),
                truncation=float(col("truncated").mean()), length_mean=float(lengths.mean()),
                length_p90=float(lengths[math.ceil(0.9 * n) - 1]),
                core=float(np.mean([bool(s["conn_match"]) or bool(s["scaffold_match"]) for s in samples])),
                exact=float(col("conn_match").mean()), tanimoto=float(col("tanimoto").mean()))


_EVAL = (("reward", "reward_mean"), ("reward_full", "reward_full_mean"), ("validity", "validity"), ("valid_closed", "valid_closed"),
         ("core", "core"), ("exact", "exact"), ("tanimoto", "tanimoto"), ("truncation", "truncation"), ("length", "length_mean"))


def drug_balanced(samples):
    """Eval aggregates with the drug as the unit: the mean over a drug's samples (all its rows), then over drugs. Each sample carries
    `drug`. `tanimoto` counts every attempt (invalid = 0); `tanimoto_valid` is the mean over drugs of each drug's mean over its valid
    finished samples, a drug without one left out (None when no drug has one)."""
    by_drug = {}
    for s in samples:
        by_drug.setdefault(s["drug"], []).append(s)
    per = [summarize(group) for group in by_drug.values()]
    out = {name: float(np.mean([p[key] for p in per])) for name, key in _EVAL}
    valid_means = [float(np.mean([s["tanimoto"] for s in group if s["valid"]])) for group in by_drug.values() if any(s["valid"] for s in group)]
    return dict(out, tanimoto_valid=float(np.mean(valid_means)) if valid_means else None, drugs=len(per), samples=len(samples))


def usable(sample):
    """A sample a deployed system could act on: valid, closed and not length-terminated (metrics.py's `_ok`)."""
    return bool(sample["valid"] and sample["closed"] and not sample["truncated"])


FV_FIELDS = ("fv_tanimoto", "fv_lift", "fv_rate", "fv_samples")


def first_valid_policy(samples, similarity):
    """The first-valid policy of metrics.py v3 (the confirmatory endpoint, docs/plan.md) on a Val sweep of k samples per prompt row: per
    row, the first sample in list order that is usable. fv_tanimoto and fv_lift are its Tanimoto and profile-specific lift (0 when no
    sample of the row is usable), fv_rate the share of rows with one, fv_samples the mean number of samples drawn up to and including it
    (k when none). Each drug's rows are averaged, then the drugs, like every drug-balanced number. A mirror of metrics.aggregate and
    metrics._first_valid, whose equality tests pin; the lift needs two drugs (one drug: 0, as there). Samples carry drug, ref (the drug's
    reference), prompt (the row's id), answer, tanimoto, valid, closed and truncated, in sample order within their row."""
    if not samples:
        raise ValueError("no samples")
    refs = {}
    for s in samples:
        refs.setdefault(s["drug"], s["ref"])
    panel, position = list(refs.values()), {drug: i for i, drug in enumerate(refs)}
    rows = {}
    for s in samples:
        rows.setdefault((s["drug"], s["prompt"]), []).append(s)
    per_drug = {}
    for (drug, _), group in rows.items():
        value = [0.0, 0.0, 0.0, float(len(group))]
        for drawn, s in enumerate(group, 1):
            if usable(s):
                lift = answer_lift(s["answer"], position[drug], panel, similarity) if len(panel) > 1 else 0.0
                value = [float(s["tanimoto"]), lift, 1.0, float(drawn)]
                break
        per_drug.setdefault(drug, []).append(value)
    mean = np.mean([np.mean(values, axis=0) for values in per_drug.values()], axis=0)
    return dict(zip(FV_FIELDS, map(float, mean)))


def train_window(metrics, after):
    """Mean of the per-step mean rewards (the training reward) of the steps after step `after`; None when there are none."""
    values = [m["reward_mean"] for m in metrics if m["step"] > after]
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------- stop rules
def collapse_streak(streak, score):
    """Consecutive steps with a collapse score below COLLAPSE_BELOW. A step without a valid answer is no evidence: it resets."""
    return streak + 1 if score is not None and score < COLLAPSE_BELOW else 0


def patience_stop(evals, patience):
    """evals: dicts {val, train} in order; val is the Val fv_tanimoto (the endpoint; the Val mean training reward is logged beside it, not
    read here), train the mean train reward over the steps since the previous eval (None for the step-0 baseline). Stop when the best Val
    eval (the first maximum; only a strictly higher value is an improvement) is at least `patience` evals old while the train reward rose
    over that span: the latest train value is above the one at the best eval (above the first one after it when the best eval is the
    baseline)."""
    if len(evals) <= patience:
        return False
    best = int(np.argmax([e["val"] for e in evals]))
    if len(evals) - 1 - best < patience:
        return False
    reference = evals[best]["train"] if evals[best]["train"] is not None else evals[best + 1]["train"]
    latest = evals[-1]["train"]
    return reference is not None and latest is not None and latest > reference


def stop_status(streak, truncation, evals, patience, collapse_stop=True):
    """The run's final status when a stop rule fires (collapse, then truncation, then patience), else None. The collapse rule reads the
    streak of collapse_score (denominator: groups with a valid answer). Without the profile (--profile-mode none) one constant answer is
    the best policy, so the caller switches the collapse rule off there; the score is still logged."""
    if collapse_stop and streak >= COLLAPSE_STEPS:
        return "stopped_collapse"
    if truncation > TRUNCATION_ABOVE:
        return "stopped_truncation"
    if patience_stop(evals, patience):
        return "stopped_patience"
    return None
