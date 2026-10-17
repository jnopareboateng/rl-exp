# Pre-registration v1 (draft until the git tag `prereg-v1`)

Status: DRAFT 2026-10-17. Everything below is fixed at the tag; no Test-T or Test-R sample has been generated or scored by any arm
(the CPU controls C0 to C2 have been scored on Val only). Any change after the tag is a logged deviation under a new tag.
Plan, contracts and memory: `docs/plan.md` (sections 2, 5, 6), `docs/contracts.md`, `docs/memory/`.

## 1. Question and claims allowed
Does online RL (GRPO family, LoRA, answer-only) deliver higher similarity to a drug's true structure than cheaper alternatives, at a
stated inference cost, when the drug is described by its pharmacology profile and is held out by structure and by approval date?
Allowed: "higher delivered similarity at a stated cost, against specified tuned controls, on this split and cohort"; for the word
"profile" in a claim, see section 7. Not claimed: structure recovery (unless exact@1 or core@1 also rise), novel-drug discovery,
biological efficacy, universal RL superiority or failure, molecules unseen in pretraining. Non-significance is not "no effect".

## 2. Data and splits (frozen files)
`data/processed/{train,val,test_r,test_t}.jsonl` and `split_report.json` as listed in `data/MANIFEST.sha256` (rebuilt and committed at
the tag). Train 241 drugs / 687 rows; Val 20 / 58; Test-R 30 / 80; Test-T 41 / 100 (first approved anywhere in 2025-26 after a regulator
audit; a temporal cohort, not proof of unseen molecules). Components join drugs with bit Tanimoto >= 0.4 or a shared non-empty Murcko
scaffold; no pair across splits reaches either. Test-T consists of 41 singleton components.

## 3. Arms (the full set; nothing else is scored on Test)
CPU controls (no parameters tuned on Test): C0 constant (best Train reference), C1 copy the structure of the TF-IDF-nearest Train
profile, C2 constant scaffold answer. Model arms at Qwen3.5-4B: A0n untrained no-think direct prompt, A4 untrained plus 3 retrieved Train
examples, A1 trace SFT, A2 answer-only SFT, A5 RL arms R1 (standard reward), R3 (contrastive reward) and the controls R4 (validity-only
reward) and, if run, the profile-free RL. Size/generation reference arms (exploratory comparisons only, not in the confirmatory set):
untrained Qwen3.5-9B, Qwen3.5-27B-FP8, Qwen3.8-27B-FP8. All use the direct-answer system text `rlexp.arms.DIRECT_SYSTEM`
(A1 keeps the dataset's system turn: it was trained with it). Diagnostics: identification (profile to name), name-given, shuffled-profile.

## 4. Endpoint and statistics
Confirmatory endpoint **fv_tanimoto**: per prompt row, the count-vector Morgan (r=2, 2048) Tanimoto of the first valid, closed, finished
answer among the first 8 samples (0 if none), mean over rows per drug, mean over drugs. Cost fv_samples (mean generations per answer)
and tokens per answer are reported for every arm. A length-terminated sample is a failure for every arm. Reported, never the primary:
expected Tanimoto over all attempts, validity, valid-only similarity, core@1, exact@1, stereo-exact, maj@8, pass@8 (oracle), lift and
fv_lift. Confirmatory contrast (one): the Val-selected A5 checkpoint (R1 or R3, chosen by Val fv_tanimoto before Test) against the best
arm by Val fv_tanimoto in the comparator set {C0, C1, A0n, A1, A2, A4 at their Val-selected decoding}, on Test-T; exact two-sided
cluster-level sign-flip test on the per-drug differences (components are the exchangeable units; Test-T has 41 singletons; Monte Carlo
100,000 flips above 20 non-tied units), alpha 0.05, wins, losses, ties and the studentized component-bootstrap line shown. The sign-flip
null is symmetry of the differences, not merely a zero mean; the studentized line is descriptive. Test-R and every other comparison are
exploratory and labelled so. One training seed per arm: seed variance is unmeasured.

## 5. Decoding and selection (frozen before Test)
Each stochastic arm: temperature chosen on Val from {0.3, 0.6, 1.0} (plus near-greedy 0.01 reported at k=1) by Val fv_tanimoto at k=8,
then frozen; top-p 1.0; max 256 new tokens for answer-only arms (A1: 8192, think mode); mode (think/no-think) as trained. RL
checkpoint: among the saved eval steps, the one with the highest Val fv_tanimoto when re-sampled with vLLM at k=8 and the arm's
chosen temperature; ties go to the earlier step. Merged exports are made from the adapters with `gpu/merge_lora.py` at the pinned
base revision. Searches guided by Val are capped: at most 3 learning rates for RL, 3 SFT settings, 3 temperatures per arm.

## 6. What is not allowed after the tag
Looking at Test results to change an arm, decoding, endpoint, selection rule or comparator set; re-scoring an arm on Test with a different
setting; adding arms. A bug fix that changes a number is a deviation reported next to both values.

## 7. The word "profile"
A claim that an arm uses the profile requires all of: fv_lift (or lift) above zero with a component-bootstrap interval excluding zero;
above the nearest-profile control C1 on the same split; and a matched shuffled-profile run (each drug shown another held-out drug's
profile, so length, section order and format are preserved) with fv_tanimoto lower than on the correct profile (paired sign-flip).
The lift's constant-invariance holds only if the completion process does not depend on the drug, and nuisance cues can raise it.

## 8. Freeze checklist (all must be true at the tag)
[ ] Independent reviews of reward/metrics, RL pack and SFT pack resolved (`docs/plan.md` review log).
[ ] R1, R3, R4 finished on Val; checkpoint and temperature selections recorded in `results/selection.json`.
[ ] `results/val_baselines.json` regenerated with metrics v3; manifest and package versions recorded.
[ ] Comparator set and temperatures written to `results/frozen_arms.json` with adapter and merged-export hashes.
[ ] `docs/contracts.md` and this file match the code at the tagged commit.
