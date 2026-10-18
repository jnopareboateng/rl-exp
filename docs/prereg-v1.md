# Pre-registration v1

Status: FROZEN by the annotated git tag `prereg-v1` (set on 2026-10-18, before any language model generated or scored a Test-T or Test-R sample; the CPU controls C0 to C2 were scored on all splits before the lock: section 3; the reward-optimal constant C3 and the
k=8 copies of the controls are made for Test after the tag, from Train alone). The tag message is `prereg-v1 spec=<sha256> selection=<sha256> manifest=<sha256>`: the frozen spec `results/frozen_arms.json`, the Val selection `results/selection.json` and the data manifest
`data/MANIFEST.sha256`; the locked scripts read these hashes from it. Any change after the tag is a logged deviation under a new tag. Plan, contracts and memory: `docs/plan.md` (sections 2, 5, 6), `docs/contracts.md`, `docs/memory/`.

<!-- frozen: comparator_set=C0,C1,C3,A0n,A2,A4; rl_set=RL-full,RL-contrastive -->

## 1. Question and claims allowed
Does online RL (GRPO family, LoRA, answer-only) deliver higher similarity to a drug's true structure than cheaper alternatives, at a
stated inference cost, when the drug is described by its pharmacology profile and is held out by structure and by approval date?
Allowed: "higher delivered similarity at a stated cost, against specified tuned controls, on this split and cohort"; for the word
"profile" in a claim, see section 7. Not claimed: structure recovery (unless exact@1 or core@1 also rise), novel-drug discovery,
biological efficacy, universal RL superiority or failure, molecules unseen in pretraining. Non-significance is not "no effect".

## 2. Data and splits (frozen files)
`data/processed/{train,val,test_r,test_t}.jsonl`, `split_report.json`, the arm prompt files and the other inputs as listed in `data/MANIFEST.sha256`
(rebuilt and committed at the tag; the locked driver refuses a data file whose hash differs from it). Train 241 drugs / 687 rows; Val 20 / 58;
Test-R 30 / 80; Test-T 41 / 100 (first approved anywhere in 2025-26 after a regulator audit; a temporal cohort, not proof of unseen molecules).
Components join drugs with bit Tanimoto >= 0.4 or a shared non-empty Murcko scaffold; no pair across splits reaches either. The 41 Test-T drugs
lie in 41 different components (33 hold one drug, 8 also hold excluded buffer drugs), so within Test-T every drug is its own exchangeable unit.

## 3. Arms (the full set; nothing else is scored on Test)
**CPU controls.** Deterministic functions of Train (C1 also reads the prompt it answers; C3 also reads a public ChEMBL approved-drug pool from which every
structure that is in, or within the split rule of, Val, Test-R or Test-T is removed). The controls CLI wrote scored files for C0 to C2 on all splits
(Val, Test-R, Test-T; `runs/control_*/scored.jsonl`) before the lock; no choice (arms, decoding, endpoint, selection, comparator set) used or looked at
their Test values. The Test oracle ceilings quoted in `docs/plan.md` section 5 are computed from reference structures only (no model output).
- C0 constant: the Train reference with the highest mean bit Tanimoto to the other Train references.
- C1 copy: the structure of the TF-IDF-nearest Train profile.
- C2 constant scaffold: the shortest Train structure among those with the most common scaffold.
- C3 reward-optimal constant: the one structure, chosen from the Train references plus the pool, with the highest mean count-vector Tanimoto to the Train
  references (a Train candidate is scored without its own entry). It is what an RL policy that ignores the profile can at best learn. Added 2026-10-17 (section 9, D1).

**Model arms at Qwen3.5-4B** (answer-only arms use the direct-answer system text `rlexp.arms.DIRECT_SYSTEM`): A0n untrained, no-think; A4 untrained plus 3
retrieved Train examples (no-think); A2 answer-only SFT, epoch 1 (think mode, empty block); A5 online RL started from A2. A1 (trace SFT) and A3 (rejection-sampling
FT) are Val-only and are not scored on Test (section 9, D2). A5 comes in reward modes, named by reward and start, each made by a `gpu/RL.md` command:

| RL arm | `gpu/RL.md` command | Reward, start | Role |
|---|---|---|---|
| RL-full | 1 (`rl_full`) | standard reward, from A2, real profile; runs R1 (pilot, seed 0, 40 steps) and R5 (replicate, seed 1, up to 100 steps) | A5 candidate |
| RL-contrastive | 5 (`rl_contrastive`, `REWARD_MODE=contrastive`) | contrastive reward, from A2 | A5 candidate |
| RL-validity | 2 (`rl_validity`, `REWARD_MODE=validity`) | validity-only reward, from A2 | mechanism control, Val only |
| RL-profilefree | 3 (`rl_noprofile`, `PROFILE_MODE=none`) | standard reward, profile replaced by a fixed string | mechanism control, Val only |
| RL-cold | 4 (`rl_zero`, `INIT_ADAPTER=none`) | standard reward, fresh LoRA, real profile | mechanism control, Val only |

The A5 candidates are RL-full and RL-contrastive. Only the one with the highest Val fv_tanimoto (section 5) is scored on Test; the other stays a Val result.
The mechanism controls are Val-only runs, made before or after the tag, and never scored on Test.

**Size and generation references** (untrained Qwen3.5-9B, Qwen3.5-27B-FP8 and Qwen3.8-27B-FP8 on the direct-answer text, no-think): Val only (Table 2 of the paper). They are not run on Test: the Test run is limited to the 4B arms (D4).

**Diagnostics (exploratory).** Identification (profile in, drug name out, normalised name match) and name-given (the name replaces the profile) for the untrained
4B (the 9B and 27B probes are Val only). Shuffled-profile: every row is shown the profile of a donor row of another drug (same task type, same
prompt-length tercile, another component when the stratum allows; `rlexp.shuffle`, seed 0), scored against its own reference. Blank-profile: the profile is replaced by
"Drug: unknown." (`rlexp.arms blank`), the model's answer with no drug information. Shuffled and blank runs are made for A2 and for the Val-selected A5 only.

Test run list (spec ids in `results/frozen_arms.json`; ids in the confirmatory pair are fixed by `results/selection.json`):

| id | what | splits |
|---|---|---|
| C0, C1, C2, C3 | CPU controls, replicated to k=8 | both |
| A0n, A4, A2, RL-full | 4B arms at their Val-selected decoding | both |
| A2-shuffle, A2-blank, RL-full-shuffle, RL-full-blank | profile diagnostics | both |
| name-4B, ident-4B | name-given and identification of the untrained 4B | both |

The Test run is one stage (`gpu/run_final.py`: the confirmatory pair first, then the other 4B arms, the diagnostics and the probes); an arm that cannot be sampled is reported as an omission.
Planned only if budget allows, otherwise reported as omissions (`docs/plan.md` section 2(d) wants them before any RL claim): profile-free RL, validity-only RL, cold-start RL,
continued SFT with a small learning-rate and step search, dense-reward-ranked rejection-sampling FT (A3 as specified keeps only identity hits and would starve), and the
profile-plus-name variant of the name-given diagnostic.

## 4. Endpoint and statistics
Confirmatory endpoint **fv_tanimoto**: per prompt row, the count-vector Morgan (r=2, 2048) Tanimoto of the first valid, closed, finished
answer among the first 8 samples (0 if none), mean over rows per drug, mean over drugs. It compares arms at a stated cost, not a matched one: fv_samples (mean
generations per answer) and fv_tokens (generated tokens per answer, with prompt_tokens for the prompt) are reported for every arm, and an arm that invalidates
weak answers can raise the endpoint by paying in draws. A length-terminated sample is a failure for every arm. Reported, never the primary:
expected Tanimoto over all attempts (mean_tanimoto), validity, valid-only similarity, core@1, exact@1, stereo-exact, maj@8, pass@8 (oracle), lift and fv_lift.
Confirmatory contrast (one): the Val-selected A5 (RL-full or RL-contrastive, chosen by Val fv_tanimoto before Test) against the best arm by Val fv_tanimoto in the
comparator set {C0, C1, C3, A0n, A2, A4} at their Val-selected decoding, on Test-T; a tie in the Val score goes to the smaller arm id. The test is an exact two-sided
cluster-level sign-flip on the per-drug differences (components are the exchangeable units; the 41 Test-T drugs lie in 41 components, so each drug is a unit; Monte Carlo
100,000 flips above 20 non-tied units), alpha 0.05, wins, losses and ties and the studentized component-bootstrap line shown. The sign-flip null is symmetry of the
differences, not merely a zero mean; the studentized line is descriptive. Test-R and every other comparison are exploratory and labelled so. The paired report asserts
its cohort (Test-T 41 drugs and 100 rows, Test-R 30 and 80). One training seed per arm: seed variance is unmeasured.

## 5. Decoding and selection (frozen before Test)
Each stochastic arm: temperature chosen on Val from {0.3, 0.6, 1.0} at k=8 by Val fv_tanimoto, then frozen; the score of an arm is its Val fv_tanimoto only (a candidate
whose Val run has fewer than 8 samples per row is not a candidate). Near-greedy decoding (0.01) is reported at k=1 as "near-greedy" and is not selectable. Top-p 1.0; max 256 new
tokens for answer-only arms; mode (think/no-think) as trained. RL checkpoint: among the saved steps of RL-full (R1: 10, 20, 30, 40; R5: every saved step from 40 on, because the pilot's Val curve was still rising at step 40), the one with the highest Val
fv_tanimoto when re-sampled with vLLM at k=8, at each grid temperature; an exact tie goes to the lower temperature, then the earlier step, then the run name (R1 before R5). RL runs evaluate Val at EVAL_K=8 in training from the second pilot on (the first used 4).
Merged exports are made from the adapters with `gpu/merge_lora.py`
at the pinned base revision and checked against the adapter hashes in `results/frozen_arms.json`; before any Test sampling, each merged adapter arm is sampled on Val at its frozen settings
(seed 0) and must reproduce its Val fv_tanimoto in `results/selection.json` within 0.005 (the sanity gate of `gpu/run_final.py`; every attempt is recorded in `runs/final/sanity.json`, and the tolerance is a constant of the frozen code,
not a flag). The gate runs before any Test sample exists, so repeating it after fixing a cause (wrong files, a stale merge) is not a Test choice. A failure stops the whole run (nothing is sampled on Test until every adapter arm has passed, the RL arm first among
the causes to look at), and the confirmatory contrast requires the latest attempt of the RL arm to have passed; the paper reports the number of attempts. Searches guided by Val are capped: at most 3 learning rates for RL, 3 SFT
settings, 3 temperatures per arm (RL-contrastive was sampled at 0.6 and 1.0 only; it is not selected).

## 6. What is not allowed after the tag
The tag `prereg-v1` is annotated and its message is `prereg-v1 spec=<sha256> selection=<sha256> manifest=<sha256>`; the locked scripts read these hashes from it. Its time is the local clock of the committing machine unless the tag is pushed to a remote at once.
The first complete locked sampling run (`runs/final`, made by `gpu/run_final.py` under the frozen spec) is the result. A second sampling run of any frozen arm is a deviation, reported with both results;
vLLM batching is not bit-reproducible, so a re-run can differ and reporting the better one would be a forking path. A run killed and resumed through the same command is not a second run.
Looking at Test results to change an arm, decoding, endpoint, selection rule or comparator set; re-scoring an arm on Test with a different
setting; adding arms. A bug fix that changes a number is a deviation reported next to both values. Val-only mechanism controls (section 3) may be run after the tag; they
cannot change anything above.

## 7. The word "profile"
A claim that an arm uses the profile requires all of: fv_lift (or lift) above zero with a component-bootstrap interval excluding zero;
above the nearest-profile control C1 on the same split; and a matched shuffled-profile run (above) with fv_tanimoto lower than on the correct profile (paired sign-flip).
The lift's constant-invariance holds only if the completion process does not depend on the drug, and nuisance cues can raise it. The blank-profile run is reported, not a
condition. Identical blank prompts give identical draws for every row, so its first-valid answer is one shared molecule: read it through mean_tanimoto and validity.

## 8. Freeze checklist (all true at the tag)
[x] Independent reviews of reward/metrics, RL pack, SFT pack and the evaluation pipeline (four rounds) resolved (`docs/plan.md` review log).
[x] Every RL candidate finished on Val (R1 steps 10 to 40, R5 steps 40 to 60, RL-contrastive steps 10 and 20), and the checkpoint and temperature selections recorded in `results/selection.json`. Each control of `docs/plan.md` section 2(d) is
    either finished on Val (validity-only RL) or listed as an omission (profile-free RL may finish after the tag, Val only; continued SFT, dense RFT and the profile-plus-name variant are omissions).
[x] The answer-reader decision (plan section 4: more than 1% of outputs differ; `results/extractor_audit.json`, 29 Val runs, 10,304 samples): the 1% rule fired (2.1%), but only 5 samples read a different
    molecule; 190 are sentence-embedded answers that the stricter rule would drop and 26 are tokens under the reader's 8-atom guard. The reader is KEPT because the RL arms were trained on it, and a
    strict-reader rescoring of the Test samples is a sensitivity analysis. T5 (plan section 4) fails for every arm (best 15.5% against 30%): reported as a finding.
[x] `results/val_baselines.json` regenerated with metrics v3; the manifest lists the Test and arm prompt files; package versions are in each run's meta.json.
[x] `results/frozen_arms.json` written by `scripts/make_frozen_spec.py` (10 arms: the 4B arms, their diagnostics and the 4B probes), its sha256 in `results/frozen_arms.sha256` and in the tag message, with adapter hashes and base revision; `gpu/final_eval.py` accepts it against `results/selection.json`.
[x] `docs/contracts.md` and this file match the code at the tagged commit.

## 9. Deviations from the plan recorded before the tag
- **D1 (comparator C3).** The first draft of this section named the comparator set {C0, C1, A0n, A1, A2, A4}. C3 was built on 2026-10-17, after the class-information ceiling
  showed that a constant over a larger pool beats class retrieval, and it is a stronger "best constant" than C0 on Train alone. It is added to the set before any model has
  generated a Test sample; its Val score (0.1865) is the highest of the set, so it is the registered comparator. The change makes the contrast harder for RL, not easier.
- **D2 (A1 and A3).** A1 is removed from the comparator set and from Test: its mean Tanimoto over valid answers on Val is 0.152 (k=4, 33% valid), which bounds its first-valid
  similarity near 0.15, below A2 and C3, so it could not be the best comparator, and its Test cost (8,192 thinking tokens per sample) was not budgeted. A3 was not run (it
  needs a model with Train hits; the dense variant is an omission).
- **D3 (RL candidates).** The RL-contrastive run was cut short by the loss of the Colab runtime at about step 28 of 40; steps 10 and 20 were saved. Their Val fv_tanimoto
  (0.12 at best) is far below RL-full and below the A2 start, so RL-full is the Val-selected A5. RL-validity, profile-free RL and cold-start RL were not finished before the tag.
- **D4 (scope).** The size references (9B, Qwen3.5-27B-FP8, Qwen3.8-27B-FP8) and their name-given and identification probes stay Val-only: the Test run is limited to the 4B arms, to put the compute and the analysis on the confirmatory contrast and its controls (decided with the project owner before the tag, no Test sample existed). The Val size trend is exploratory and reported as such.
- **D5 (shuffle control).** The shuffled-profile control keeps task type and prompt-length tercile, not format beyond that; it replaced a first version that only paired
  donors at random, before any Test file was built.
- **D6 (replicate run).** The pilot R1 was still improving at step 40 (Val 0.163, 0.175, 0.178, 0.186 at steps 10 to 40, T=0.3), so a replicate R5 with another seed and a budget of 100 steps was added before the freeze (it stopped itself at step 63 on its collapse rule); its checkpoints from step 40 are RL-full candidates. R1 ran an earlier
  version of the RL pack (patience followed the Val mean reward and the resume identity had no objective hash); the training maths is the same, and R5 runs the pack committed in `1f44e4d` (objective hash 8f4d26b5; its rollouts also stop at the endoftext token). Both choices use Val only.
