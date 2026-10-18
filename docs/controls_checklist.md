# Controls for similarity-rewarded chemistry tasks

A checklist for any text-to-structure or molecule-generation result that is scored by fingerprint similarity. Each item says what to compute, what it costs, why, and what this study measured on its three
held-out sets: Val (20 drugs in 16 components; the selection data), Test-R (30 drugs in 27 components; random held-out components) and Test-T (41 drugs; a temporal cohort first approved in 2025 and 2026).
Similarity is Morgan count-fingerprint Tanimoto; "delivered similarity" is the similarity of the first valid answer in up to 8 tries. Intervals are wide at these sizes. Sources are files in this repository.
Cost: "CPU" is minutes to a day of CPU work on your own data; "inference" is one more sampling pass; "training" is one more training run.

1. **Tuned constant answer** (CPU). Compute: the one structure (training set, optionally plus a public pool) with the highest mean similarity to the training targets; answer it for every input.
   Why: a score is news only above what an input-free answer earns. Here: reward-optimal constant 0.187 [0.151, 0.229] on Val, 0.199 on Test-T (the best of 1,828 candidate constants there) and 0.159 on
   Test-R (21st of 1,828; the best is 0.171); the Train-only constant 0.176, 0.184 and 0.160. The constant's level moves by 0.04 between splits (results/val_baselines.json, test_t_baselines.json, constant_ceiling*.json).
2. **Random-pair calibration** (CPU). Compute: the similarity of random pairs of reference structures; place each score in that distribution.
   Why: similarity scales are compressed and a score means little without one. Here: mean 0.141, 90th percentile 0.213, 99th 0.274 over 332 structures; 0.187 is the 79th percentile, 0.199 the 85th and the RL
   checkpoint's 0.186 on Test-T the 78th. A scale over the study's drugs, not a null distribution (scripts/random_pair_calibration.py, results/random_pair_calibration.json).
3. **Delivered similarity at a stated cost** (the evaluation pass). Compute: the similarity of the answer a user would get under a fixed policy (first valid of up to 8 tries), with mean tries and generated tokens per answer.
   Why: an arm can raise similarity by paying in tries. Here, Test-T: untrained 4B 0.073 at 5.5 tries and 761 tokens; answer-only SFT 0.131 at 2.0 tries and 92 tokens; RL checkpoint 0.186 at 1.0 tries and
   34 tokens; every constant 0 tokens. Prompt tokens are not counted, which flatters retrieval (results/test_t_baselines.json).
4. **Shuffled-input counterfactual** (inference). Compute: rerun the system with each input replaced by another item's input (same task type and length), score against the original targets, and read an input-deleted
   run through mean similarity, not first-valid similarity. Why: the drop is the part of the score that depends on the input. Here: real minus shuffled delivered similarity for the RL checkpoint +0.012 (Val, p = 0.034),
   +0.004 (Test-T, p = 0.31), +0.013 (Test-R, p = 0.010, which is 0.080 after Holm over the eight Test cells); untrained 9B and 27B +0.040 and +0.049 (Val, p = 0.045 and 0.009). Deleting the input costs the RL checkpoint
   0.018 [0.009, 0.027] of its 0.184 mean similarity on Test-T, 60% of which another drug's input recovers (results/paired_*.json, paired_blank_*.json).
5. **Constant-proof readout** (CPU on the answers you already have). Compute: rank the true target among the split's own N references by similarity to your answer; report the top-1 rate, the mean percentile rank, the share of rows with
   an answer and the number of distinct answers. Why: one fixed answer valid on every row scores exactly 1/N and (N+1)/2N whatever the molecule (failures and ties only lower it; a policy whose answers vary scores them in
   expectation), so a constant cannot pass. Here: chance 0.525 (N=20), 0.512 (N=41), 0.517 (N=30); RL checkpoint 0.473, 0.475, 0.463; nearest-profile copy 0.439, 0.427, 0.396; constants and input-deleted runs sit on chance (results/rank_*.json).
6. **Held out by structure component and by date** (CPU). Compute: join molecules into components (similarity 0.4 or more, or the same ring scaffold), assign whole components to splits, and add a cohort by first approval date.
   Why: near-duplicates across splits reward memorising. Here: 302 components, no cross-split pair reaches 0.4 (largest 0.398), no scaffold is shared, and the date cohort is 41 drugs first approved
   in 2025-26 (data/processed/split_report.json). Approval is not disclosure: 40 of those 41 have a PubChem record 3 to 20 years older (median 10.5).
7. **Validity-only reward control** (training). Compute: train the same recipe with a reward that pays only for a parseable answer. Why: separates learning the output format from learning chemistry.
   Here: 99.5% to 99.8% valid at step 40 but delivered similarity 0.074 to 0.080 on Val, far below the constant 0.187 (results/val_baselines.json, rows rlvalidity-s40-*).
8. **Input-free training control** (training; 0.2 GPU-hours here). Compute: train with the same reward and the input replaced by one fixed string. Why: it shows what the reward gives with no input at all, which is the constant the
   policy can converge to. Here: in two runs the training reward matches the run with the input (0.159 and 0.152 against 0.158 over steps 31 to 40) and Val similarity at step 40 is 0.178 and 0.196 against 0.183 with the input
   (0.195 for the selected checkpoint), by policies that concentrate on few answers (results/paired_val_r4*.json, runs/rl/R4, R4b).
9. **Seeds and selection optimism** (CPU, bookkeeping). State: training seeds, how many candidates were scored on the selection split, and the spread when the same checkpoint is sampled again.
   Why: the best of many candidates on 20 drugs is biased upward. Here: the reported RL checkpoint is the best of 21 Val candidates (0.149 to 0.195, median 0.183) and delivers 0.186 on Test-T and 0.168 on
   Test-R; re-sampling it gives 0.1952 and 0.1948 against 0.1949; two RL training runs, one seed each (results/selection.json, results/gate_dryrun.json).
10. **The constant next to every number** (CPU). Report: the constant's score and the ceiling beside each result, and say whether the result is above them. Why: it is the reading that survives review.
    Here: Test-T constant 0.199 = ceiling, RL 0.186 below it; Test-R constant 0.159, ceiling 0.171, RL 0.168; Val constant 0.187, ceiling 0.195, RL 0.195 (selected on it). No system recovers a
    structure: exact match is 0 for every system on every split.
