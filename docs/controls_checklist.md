# Controls for similarity-rewarded chemistry tasks

A checklist for any text-to-structure or molecule-generation result that is scored by fingerprint similarity. Each item says what to compute, why, and what this study measured.
Numbers are on the 20 Val drugs (16 structure components) unless stated, with Morgan count-fingerprint Tanimoto; "delivered similarity" is the similarity of the first valid answer in up to 8 tries.
Intervals are wide: the constant answer's own 95% interval is 0.07 to 0.08 wide. Sources are files in this repository.

1. **Tuned constant answer.** Compute: the one structure (training set, optionally plus a public pool) with the highest mean similarity to the training targets; answer it for every input.
   Why: a score is news only above what an input-free answer earns. Here: constant 0.176 [0.145, 0.212], reward-optimal constant 0.187 [0.151, 0.229] (results/val_baselines.json),
   best constant that could have been picked with Val in view 0.195 (results/constant_ceiling.json).
2. **Random-pair calibration of the metric.** Compute: the similarity of random pairs of reference structures; place your score in that distribution.
   Why: similarity scales are compressed and a score means little without one. Here: mean 0.141, 90th percentile 0.213, 99th 0.274; 0.176 is the 73rd percentile, 0.187 the 79th,
   0.195 the 83rd (scripts/random_pair_calibration.py, docs/prereg-v1-addendum1.md section 2; results/random_pair_calibration.json is not written yet).
3. **Delivered similarity at a stated cost.** Compute: the similarity of the answer a user would get under a fixed policy (first valid of up to 8 tries), with mean tries and generated tokens per answer.
   Why: an arm can raise similarity by paying in tries. Here: untrained 4B 0.087 at 4.9 tries and 377 tokens; untrained 27B 0.186 at 1.5 tries and 100 tokens;
   selected RL checkpoint 0.195 at 1.0 tries and 32 tokens (results/val_baselines.json).
4. **Shuffled-input counterfactual.** Compute: rerun the system with each input replaced by another item's input (same task type and length) and score against the original targets.
   Why: the drop is the part of the score that depends on the input. Here: answer-only SFT 0.156 to 0.131 and RL 0.195 to 0.183 when shuffled, 0.181 for RL with a blank input;
   all three are below the constant 0.187 (results/val_baselines.json). Paired sign-flip p = 0.022 and 0.034 for the two shuffles (docs/prereg-v1-addendum1.md section 4).
5. **Constant-proof readout.** Compute: rank the true target among the split's own N references by similarity to your answer; report the top-1 rate and the mean percentile rank.
   Why: any input-independent answer scores exactly 1/N and (N+1)/2N whatever the molecule, so a constant cannot pass. Here: N = 20, chance 0.050 and 0.525; both constants and the
   blank-input RL run sit on chance; mean percentile rank (lower is better) 0.396 for answer-only SFT and 0.473 for RL (scripts/rank_readout.py, docs/prereg-v1-addendum1.md section 2; no results file yet).
6. **Held out by structure component and by date.** Compute: join molecules into components (similarity 0.4 or more, or the same ring scaffold), assign whole components to splits, and add a cohort by first approval date.
   Why: near-duplicates across splits reward memorising. Here: 302 components, no cross-split pair reaches 0.4 (largest 0.398), no scaffold is shared, and the date cohort is 41 drugs first approved
   in 2025-26 (data/processed/split_report.json). Approval is not disclosure: 40 of those 41 have a PubChem record 3 to 20 years older (median 10.5).
7. **Validity-only reward control.** Compute: train the same recipe with a reward that pays only for a parseable answer.
   Why: separates learning the output format from learning chemistry. Here: 99.5% to 99.8% valid at step 40 but delivered similarity 0.074 to 0.080, far below the constant 0.176
   (results/val_baselines.json, rows rlvalidity-s40-*); the similarity-shaped reward reached 0.189 in the same 40 steps.
8. **Profile-free (input-free) training control.** Compute: train with the same reward and the input replaced by one fixed string.
   Why: shows what the reward gives with no input at all, which is the constant the policy can converge to. Here: not run (an omission in docs/prereg-v1.md section 8).
   Its test-time counterpart, a blank input, gives RL 0.181 against 0.195 with the real input (results/val_baselines.json).
9. **Seeds and selection optimism.** State: training seeds, how many candidates were scored on the selection split, and the spread when the same checkpoint is sampled again.
   Why: the best of many candidates on 20 drugs is biased upward. Here: the reported RL checkpoint is the best of 21 Val candidates (0.149 to 0.195, median 0.183); sampling it again with two seeds
   gave 0.1952 and 0.1948 against 0.1949, answer-only SFT 0.1565 and 0.1519 against 0.1557 (results/selection.json, results/gate_dryrun.json). Two RL training runs reached 0.189 and 0.195, one seed each.
10. **The constant next to every number.** Report: the constant's score and the ceiling beside each result, and say whether the result is above them.
    Why: it is the reading that survives review. Here: Val constant 0.187, ceiling 0.195, selected RL 0.195 (results/val_baselines.json, results/constant_ceiling.json):
    the trained system is at the ceiling, not above it, and no system recovers a structure (exact match is 0 for all).
