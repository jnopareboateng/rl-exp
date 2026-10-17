# Plan v5: does online RL add value for structure recovery? A controls-first study (Qwen3.5-4B, 9B, Qwen3.8-27B)

Status: v5, 2026-10-17 (v4 plus the first measured baselines, sections 5 and 6, and two open protocol questions in section 2). Supersedes v3 after two independent reviews (Fable 5.1, Codex GPT-6.1 Sol). Freeze point: git tag `prereg-v1`, set before any Test-split sampling.
Labels: **measured** (scripts in `scripts/`), **recalled** (papers from memory), **inferred** (reasoning, untested). Contracts for every file: `contracts.md`.

## 0. Review log

| Source | Finding | Response |
|---|---|---|
| Fable | 0.2 similarity floor erases the within-group signal (98.5% of wrong pairs are at or below 0.2) | Floor removed from the training reward; scaffold tier added |
| Fable | 2023-24 approvals are not recall-proof; need a pass@8 gate, an SFT arm, a step budget | 2025-26 cohort; G0; SFT arms; budget section |
| Fable | "Qwen3.8-27B is really 3.5-27B" | Rejected: separate repos and model cards, configs differ in 8 keys. Size and generation are confounded, so a 3.5-27B baseline is kept |
| Codex | NO-GO for RL as written; compare against answer-only SFT and rejection-sampling FT first | Adopted: this plan is controls-first (section 2) |
| Codex | Test-R used as a development set; primary-metric rule allows selective claims; bootstrap too optimistic at 43 drugs | Decisions use Train-baseline and Val only; one endpoint, one confirmatory contrast, exact sign-flip test (section 2) |
| Codex | Labels and "exact" ignore stereo; FDA date helper can miss earlier approvals; approval date is not disclosure date | Stereo-exact reported; helper fixed and audited by hand; Test-T is a temporal cohort, with PubChem/PubMed exposure covariates |
| Codex | Split units; retrieval control is answer-copying; LoRA parity at step 0 proves nothing; T5 passes on format noise; collapse metric undefined | Scaffold edges and cluster bootstrap; renamed control plus in-context arm; parity after a nonzero update; valid-only spread; defined collapse score (sections 3 to 6) |
| Codex | D, p90 and census wording overclaimed | Corrected below |
| Opus 5.5 (SFT pack) | A1 length cap dropped a third of Train and biased to short traces; missing fast kernel silent on A100; best-by-loss checkpoint only; tests missed six mutations | Caps 8192/2048, kernel gate fatal, every epoch adapter kept, 50 tests with mutation proofs |
| Codex (SFT pack) | Epoch export could merge against the wrong base revision and bypassed the run lock and committed hashes; A3 rows could share ids; padding and clipping mutations survived | `revision.json` pin, shared lock plus committed-hash check, atomic swap, `#sample_idx` ids, new tests (round 2) |
| Codex (plan v5) | Valid-only Tanimoto is selectable; decoding tuning is fair if frozen; missing controls (validity-only RL, profile-free RL, continued SFT, dense RFT, shuffled profile); collapse score needs diagnostics; claims language | Endpoint fv_tanimoto, per-arm temperature grid, controls added, diagnostics logged, claim rules (section 2) |
| Codex (metrics v3, class ceiling) | Lift is not constant-invariant when completion depends on the drug; NaN silently dropped drugs; partial cluster maps could flip p-values; pairwise "split-matched" neighbours are weaker than the real component split; claims overreached | Lift demoted to a diagnostic, input checks, component-matched ceiling with component bootstrap, wording corrected (round 2) |
| Codex (RL pack) | Contrastive reward paid for abstention; resume identity did not pin the objective; Val did not measure the endpoint; six mutations survived; odd min-groups threshold | Reward floored at 0, objective hash, fv metrics and patience on fv_tanimoto, tests and mutants, ceiling threshold (round 3) |
| Opus 5.5 (SFT fixes, metrics v3 round 2, RL pack) | Cross-vendor credits exhausted 2026-10-17; same-vendor independent reviews from here | Reports in `.agent-runs/reviews/opus_*.md`; findings and responses added below as they land |

## 1. What the task is (measured, pooled train + val)

| Finding | Number | Consequence |
|---|---|---|
| The SFT answer is the real approved drug | Same connectivity as ChEMBL in 98-99% of the 942 rows whose name resolved (94-95% of all 980 single-drug rows); stereo-exact in about 89% (842/942) | Ground truth is exact and parse-free. The traces are rationalisations written with the answer known. Report stereo separately. |
| Prompts rarely mention checkable properties | A keyword survey finds none of formula, MW, logP, TPSA, HBD/HBA terms in 75-93% of prompts (word presence, an upper bound on usable numeric constraints); 88-96% of A/B/MO discuss PK | A constraint-satisfaction reward can apply to a minority at most. Not the primary reward. |
| Few independent examples | 354 single-drug groups, 62 with one view; about 300 structure components | The effective sample is the molecule. After buffer and exclusions Train has about 235-241 drugs (final after hand resolution). |
| Low analog leakage | Nearest other drug: median Tanimoto 0.26; 11% have a neighbour of 0.5 or more | Component split is cheap insurance. |
| Similarity landscape | Random pairs: mean 0.11, p99 0.21; best constant answer 0.14; random pairs share a Murcko scaffold 0.12% of the time | Raw Tanimoto carries rank signal; a scaffold hit is informative |
| Approval years (drugs) | up to 2014: 55; 2015-21: 182; 2022: 15; 2023: 27; 2024: 29; 2025: 28; 2026: 16 | 41 single drugs (100 rows) first approved anywhere in 2025-26 after a regulator audit (5 of 43 candidates had earlier approvals in Japan or China) form the temporal cohort |
| Name leakage | 0 of 991 prompts contain the drug's own name | No trivial lookup |
| Lengths | Retained completions p50 2.7k, p90 5.5k tokens (train+val); prompts p50 0.56k | Cap 8k tokens |
| Models (config.json) | 4B, 9B, 3.5-27B, 3.8-27B are dense, 3:1 linear/full attention, vocab 248,320 | Cost scales with N; logits are the memory hazard |

The task is structure elucidation from a pharmacology profile, scored against a known structure. Success on unseen drugs mixes recall and reasoning.

## 2. Question, arms, endpoint, statistics, locking

**Question.** Does online RL add value for held-out structure recovery beyond cheaper alternatives, at matched inference cost?

**Arms at 4B** (each trained on Train only, selected on Val; token cost per answer reported for every arm):

| Arm | What | Cost |
|---|---|---|
| C0 constant, C1 copy-nearest | CPU controls; copy-nearest copies a Train structure (answer copying, not retrieval-augmented prompting) | none |
| A0 / A0n | Base model, thinking on / off | inference |
| A4 | Base + in-context retrieval: the 3 nearest Train prompts with their reference SMILES in the prompt | inference |
| A2 | Answer-only SFT (LoRA; empty think block, then the SMILES) | small |
| A1 | Trace SFT (LoRA; reference traces cut after the answer line) | small |
| A3 | Rejection-sampling FT: keep self-generated Train rollouts that reach conn_match, SFT on them | sampling + small |
| A5 | Online RL (GRPO-family, init from the best of A0/A1/A2 on Val), answer-only mode first (about 100 tokens per rollout); thinking-mode RL only if compute allows | large |

**Endpoint (one, for every arm and size):** core@1 = conn_match or non-empty Murcko-scaffold match, mean over 8 samples, drug as the unit. Secondary and reported, never primary: exact@1, stereo_exact@1, mean Tanimoto, maj@8 (deployable plurality vote), pass@8 (oracle coverage, an upper bound that needs the answer to select).
**Confirmatory contrast (one):** A5 at its Val-selected checkpoint versus the best non-RL arm by Val, on Test-T, decided by the exact two-sided sign-flip permutation test on per-drug differences at alpha 0.05, with wins, losses and ties shown. Test-R and every other comparison are exploratory, labelled so, with no adjustment claimed.
**Statistics.** The sign-flip p-value decides; cluster-bootstrap intervals are descriptive. One training seed per arm, so seed variance is unmeasured and stated as a limitation. At 43 drugs only large effects can be detected; no power claim beyond that.
**Locking.** Reward weights, extractor, decoding, arms and hyperparameters, checkpoint-selection rule, endpoint and contrast are fixed and committed under tag `prereg-v1` before any Test-split sampling. Decisions (G0, G1, reward tuning, model selection) use Train rollouts of the untrained model and Val only. Each frozen arm is scored on Test once; any later change gets a new tag and is reported as a deviation.
**If A5 does not win:** the deliverable is the benchmark, the controls, and the SFT/RFT results (a negative result for RL is a result). 9B and 27B are entered only after 4B A5 wins; the 27B then needs its own matched SFT comparator.
**Revised before the lock (2026-10-17, after the Val baselines of section 5 and an independent review of this plan; the lift metric itself is still to be checked independently before `prereg-v1`).** These supersede the endpoint, contrast and decoding paragraphs above where they differ.
(a) *Endpoint.* core@1 is near 0 for every arm, and Test-T drugs approved in 2025-26 cannot be recalled, so exact and core recovery stay near 0 for every arm; a sign-flip test on them is decided by ties. After a second independent review, the single confirmatory endpoint is **fv_tanimoto**: the mean count-vector Tanimoto of the **first valid, finished answer among up to 8 samples** of each prompt row (0 when none is valid), drug-balanced. It is the deployable "resample until RDKit accepts the SMILES" policy, so arms are compared at matched validity without conditioning away hard cases, and its cost (`fv_samples`, mean generations per answer, retrieval prompts and retries included) is reported beside it. Reported, never the primary: expected Tanimoto over all attempts (invalid = 0), validity, valid-only similarity, core@1, exact@1, stereo-exact, maj@8, pass@8 (oracle) and the **profile-specific lift** (answer-to-own-reference Tanimoto minus its mean to the other drugs' references of the split; `fv_lift` for the first-valid policy). The lift is a diagnostic of association between the drug and the policy's output, not a proof of profile use: a constant answer has lift exactly 0 only if the completion process does not depend on the drug (a fixed molecule scored +0.18 when only one drug finished), and a policy keyed on nuisance cues such as prompt length can score high. Claim language: a positive fv_tanimoto difference is "higher delivered similarity at a stated cost"; the word "profile" requires a positive lift above the nearest-profile control C1 AND a matched shuffled-or-removed-profile comparison that keeps prompt length, section order and format.
(b) *Contrast.* Unchanged in form: A5 at its Val-selected checkpoint against the best non-RL arm by Val on Test-T, cluster-level exact sign-flip on the per-drug fv_tanimoto differences, plus a studentized component-bootstrap line as a sensitivity check; the p-value is not assumption-free (components are the exchangeable units). Test-R is exploratory.
(c) *Decoding.* Every stochastic arm gets the same tuning opportunity: temperature from {0.01, 0.3, 0.6, 1.0} chosen on Val by fv_tanimoto with the same k (8) at each temperature, then frozen. Thinking mode, prompt, top-p, token cap and stopping are frozen; a length-terminated answer is a failure for every arm; the cost report includes retrieval prompts, failed attempts and retries.
(d) *Controls required before any RL claim* (additions to the arm table): the exact checkpoint A5 starts from, with identical inference; **validity-only RL** from the same checkpoint (isolates syntax); **profile-free RL** (the profile replaced by a fixed string, same reward: what the reward gives without the profile); continued SFT with a small learning-rate and step search; dense-reward-ranked RFT from a matched rollout pool (A3 as specified keeps only identity hits and would starve); first-valid and best-of-k selection; the identification and name-given diagnostics with fixed aliases and a profile-plus-name variant; and a frozen shuffled-profile diagnostic (the lift is its dual and needs no extra sampling).
(e) *Order.* The zero-RL size and cost sweep (4B, 9B, 3.5-27B, 3.8-27B) is the main study; RL is a bounded case study. The RL size is chosen by predeclared Train/Val validity, headroom and compute rules, not by a positive 4B RL effect. Prefer repeat training seeds for the final RL and comparator pair over a wide RL size ladder. Cap the number of Val-guided searches.
(f) *Allowed claims.* Positive: improved expected similarity or profile-specific similarity under a fixed budget, against specified tuned controls, on this split and cohort. Negative: a bound for the tested recipe and budget (non-significance is not "no effect"). Not claimed: novel-drug discovery, biological efficacy, universal RL superiority or failure, molecules unseen in pretraining.

## 3. Phase 0: data foundation (CPU)

1. `rlexp.data_prep` on the pooled train+val rows: drop PD (155 rows) and the 11 combination rows; references from ChEMBL cross-checked against the SFT answers, hand-resolved where they disagree; final approval year = manual > audit > ChEMBL/openFDA.
2. Splits (seed 0, built): Test-T = first approved anywhere in 2025-26 (41 drugs, 100 rows); buffer = their component neighbours (19 drugs); Test-R 30 drugs (80 rows) and Val 20 (58 rows) drawn by whole components; Train = rest (241 drugs, 687 rows); 3 excluded. No pair across splits reaches Tanimoto 0.4 or shares a scaffold. Components join drugs with Tanimoto >= 0.4 or an identical non-empty scaffold. Exclusions: unresolved disagreement, no year, unextractable reference.
3. Exposure covariates (PubChem creation year, PubMed count) per drug; Test-T results are also reported by exposure tercile. Approval date is not disclosure date, so Test-T is a temporal cohort, not proof of unseen molecules.
4. Data manifest (sha256 of raw and processed files) recorded in every run.

## 4. Phase 1: reward v2 and its tests (CPU)

Gate: `</think>` closed, a SMILES extracted and parseable, else 0. Reference and answer standardised identically.
`r = 0.05 + 0.6 * Tanimoto + 0.3 * [same non-empty Murcko scaffold]` with Morgan count-vector Tanimoto, `r = 1.0` when the answer's identity key (InChIKey first block) equals the reference's or a curated alternate's (5 references have one); stereo reported only. Answers over 1,500 characters or 400 atoms, or with dummy atoms, are invalid. Weights are provisional and tuned only on Train rollouts and Val.
Offline tests before any GPU RL (Train and Val only; measured: T1 745/745, T2 mean 0.136, T3 200/200, constant-answer bar 0.160 on Train and 0.147 on Val): T1 reference completion scores 1.0 on its own row for 98% or more (misses listed); T2 another drug's reference averages 0.2 or less; T3 randomised SMILES of the reference score identically;
T4 the 77 answer-format fixtures behave as recorded; T5 on real Train rollouts at least 30% of groups show reward std of 0.05 or more **among valid completions only**. The extractor's 11 known silent-wrong cases are measured on real rollouts:
if more than 1% of outputs differ between this extractor and "last SMILES-only line", switch to the latter before `prereg-v1`. Codex reviews `reward.py` before use.

## 5. Phase 2: first GPU weeks (inference first, then small training)

Done in the first A100 session (2026-10-17, Colab A100-40GB; aggregate numbers in `results/colab_session1_report.json`):
the pinned stack installs in 57 s, a single prompt decodes at 113 tokens/s and batched sampling reaches about 2,000 tokens/s (no-think) or 885 tokens/s (long think), startup is about a minute once caches are warm.
**G1 fails for the dataset prompt:** under its system turn ("Reasoning effort is set to xhigh ...") 99.8% of no-think samples hit a 512-token cap with no SMILES, 97% still hit 4,096, and only 8% of think samples close their reasoning within 8,192 tokens; none yields a valid answer.
**Prompt policy:** answer-only arms use the direct-answer system text `rlexp.arms.DIRECT_SYSTEM` = "Reply with only the SMILES string of the drug described. No explanation." (no-think); the untrained model is reported under both prompts.
Untrained Qwen3.5-4B, direct-answer prompt, Val (20 drugs, k=8): 82% of samples finish within 256 tokens, 17% are valid SMILES, exact@1 0, core@1 0.8% (CI 0 to 1.9%), core pass@8 5%, mean reward 0.03 against the constant-answer bar 0.15.
So the base model has almost no knowledge of these structures; the first lift is teaching valid SMILES output (A2), and RL starts from A2, not from the base.

**Measured on Val** (20 drugs, 58 rows; k=8, T=1.0, 256-token cap, direct-answer system text; A2 sampled in think mode, the rest no-think; `runs/colab-s1/*/report.json`):

| Arm | valid-closed | exact@1 | core@1 | core@8 (oracle) | mean Tanimoto | mean reward |
|---|---|---|---|---|---|---|
| Constant answer (C0) | 1.00 | 0 | 0 | 0 | 0.176 | 0.156 |
| 4B untrained | 0.18 | 0 | 0.008 | 0.050 | 0.028 | 0.028 |
| 4B + 3 retrieved examples (A4) | 0.54 | 0 | 0.025 | 0.117 | 0.087 | 0.087 |
| 4B A2 (answer-only SFT, epoch 1 of 3) | 0.56 | 0 | 0.004 | 0.017 | 0.079 | 0.077 |
| 9B untrained | 0.46 | 0 | 0.021 | 0.087 | 0.075 | 0.074 |
| 27B-FP8 (Qwen3.5) untrained | 0.69 | 0 | 0.048 | 0.113 | 0.124 | 0.123 |
| 4B told the drug name (control) | 0.33 | 0 | 0 | 0 | 0.043 | 0.042 |
| 9B told the drug name (control) | 0.43 | 0 | 0.021 | 0.150 | 0.070 | 0.070 |

Ceilings and floors for graded similarity (dataset properties, not model results): the best constant answer scores mean Tanimoto 0.176 on Val; an oracle that copies, for each drug, the structurally nearest Train molecule scores 0.31 (Val), 0.30 (Test-R), 0.28 (Test-T), so a method that only selects among Train structures lives between about 0.18 and 0.31.
Identification diagnostic (`rlexp.ident`; profile in, drug name out, scored by normalised name match, drug-balanced, Val): correct name for 1.0% of samples at 4B (pass@8 4.2%), 2.1% at 9B (pass@8 9.6%), 8.1% at 27B-FP8 (k=2, pass@2 12.1%). The 27B told the name (k=2) still writes no exact structure (valid 0.58, Tanimoto 0.096), so identification and SMILES recall both fail at these sizes, and identification is the part that grows with scale.
**Class-information ceiling** (`scripts/class_ceiling.py`, ChEMBL_37, `results/class_ceiling.json`, 1,786 drugs in 771 components): two retrieval rules (1-nearest-neighbour, 5-medoid) over coarsened class facts (ATC, mechanism, target, indication) predict a drug's structure from its neighbours. With neighbours allowed by the pairwise split rule (no Tanimoto >= 0.4, no shared scaffold) the 1-NN scores 0.201 against 0.169 for the best constant (+0.032, component-bootstrap [+0.015, +0.051]; Train targets +0.001, compatible with gains up to +0.014). An independent review showed the real split separates whole connected components; with the target's whole component excluded the 1-NN scores 0.163 (-0.006 [-0.011, +0.001] against the best constant, compatible with gains up to +0.001; Train targets -0.014 [-0.024, -0.004]) and beats only a Train-fixed constant over the whole pool (+0.017), not for Train targets (-0.011). Reading: class facts through these two rules do not beat a constant once neighbours respect the split. This does not bound learned predictors, richer facts or free-text profiles (the nearest-profile copy control C1 has profile-specific lift +0.022 at constant-level Tanimoto), and failing to reject zero is not equivalence. Decision: scaling the number of training drugs with synthetic profiles is deprioritised, not ruled out.
Near-greedy decoding (T=0.01, k=1) moves the numbers a lot: A2 valid 0.81, mean reward 0.114 (0.077 at T=1.0); 9B untrained valid 0.55, reward 0.093; but 14-20% of near-greedy samples run to the cap in repetition loops.
A2 training (12 minutes on the A100): teacher-forced Val loss 1.25 after epoch 1 and 1.61, 1.72 after epochs 2 and 3 while train loss fell 0.99, 0.38, 0.14, so one epoch is the best checkpoint and more epochs only memorise the 241 Train drugs.
Reading (measured, then inferred): (1) no arm clears the constant-answer bar on mean reward, and the 27B's valid answers average 0.18 Tanimoto, the constant's level, so there is no evidence yet that the profile informs the structure beyond a class prior;
(2) naming the drug does not help at 4B and 9B, so SMILES recall, not identification, is the first bottleneck at these sizes (the identification diagnostic, `rlexp.ident`, splits this properly);
(3) core@1 is 0 to 5% for every arm, so with 41 Test-T drugs the primary endpoint can only separate arms with a very large effect. Section 2 lists the two protocol questions this raises.

Remaining steps:
1. A4 (base plus in-context retrieval, direct-answer prompt, Val) and A2 / A1 LoRA SFT (scripts in progress), each scored on Val only; Test untouched.
2. A3 (rejection-sampling FT) only from a model whose Train success rate is above zero (A2).
3. G0 is read on A2 (core-match pass@8 minus pass@1 on Val and the Train subset), because the untrained model has nothing to sharpen; G1 becomes: valid-and-closed rate of 80% or more at the answer-only cap (256 tokens), measured drug-balanced (A2 near-greedy: 81% by row, 75% by drug, so it fails; the RL pilot's first job is validity).
4. Template check: the native Qwen3.5 template with thinking on versus the dataset's `<think>` placement matters only for the trace arm (A1); the answer-only arms use the empty think block.

## 6. Phase 3: RL pilot (A5)

**Implementation (decided and built 2026-10-17; supersedes the TRL/ms-swift plan; details and gates in `gpu/RL.md`).** One process on the A100: rollouts come from Hugging Face `generate` on the PEFT model being trained (no vLLM, no TRL), with an fp32 LM head in generation and in training, so there is no engine or weight-sync mismatch and the environment is the SFT one. Answer-only completions are short, so training dominates the step time.
Loss: one update per rollout batch, so the importance ratio is 1 and the update is REINFORCE with a group baseline (CISPO's clipped ratio would need old log-probs; documented in code); advantages centred per group, divided by the batch std of the retained centred advantages; zero-variance groups dropped; every token of completion i weighs `A_i / (T_g * G)` (prompt-level aggregation); no KL. A length-terminated answer is a failure (reward 0) and stays in its group; it is not masked, because with a 256-token cap it is a failure to stop, not a long chain of thought. Cap 256 tokens, group 8, temperature 1.0, AdamW without weight decay, LoRA rank 32 on all projections including the GatedDeltaNet ones, learning rate 1e-5 to 3e-5 (at most three settings, on Val), warmup 5 steps, clip 1.0.
Why not TRL: its prompt-level aggregation, batch normalisation of centred advantages, zero-variance dropping and fp32 tied head each needed a hook (`gpu/STACK.md`); a 1,000-line own loop with CPU-tested maths and a tiny-model run is simpler to audit than five hooks. TRL stays the cross-check for thinking-mode RL later.
Measured gates on the A100 (Qwen3.5-4B, A2 adapter): sampling versus training-forward log-probs, mean |diff| 0.015 to 0.018 per token, mean importance ratio 0.994 to 1.000, 8 to 11% of tokens above 0.05, the same with a prompt alone as in a heavily left-padded batch (so padding adds no error; the gap is bf16 decode-versus-prefill noise; the provisional "5% above 0.05" threshold was too strict); smoke gate: 3 real steps, the first update moves the adapter (delta norm 0.016), losses finite, resume reproduces the schedule, seeds and even the rollouts bit for bit. Step cost at 8 prompts: generation 25 s (latency-bound: 256 decode steps), update 10 s.
Batch construction: each update needs at least 16 active drugs after filtering; oversample up to 2x to refill; skip the update and log if fewer than 8 remain.
Gates before real steps: exact template match, step-0 trainer-vs-vLLM log-prob gap, then **parity after a nonzero optimizer update** (a no-op LoRA init passes step 0 trivially), adapter coverage of every projection, five finite steps, peak memory at the 8k cap, kill-and-resume with optimizer and RNG state.
Per step log: reward mean/std, zero-variance share, valid, closed, truncation, length mean and p90, clip, grad norm, and the **collapse score**: distinct modal answers (by connectivity) over distinct drugs in the batch; healthy is near 1, alarm below 0.7.
Val eval every 10 steps (k=4). Stop: Val flat for 3 evals while train reward rises (memorisation); collapse alarm; truncation above 30%. **G2:** Val gain beyond the paired interval, no collapse, resume works.

## 7. Phase 4: locked evaluation and decision

Tag `prereg-v1`. Score each frozen arm once on Test-T and Test-R (k=8). Report the confirmatory contrast, the exploratory table, exposure-tercile and 2025/2026 splits, tokens per answer, and the baseline sweep over sizes (0.8B to 27B, inference only, optional and last).
Then decide on 9B and 27B per section 2. The measurable scaling result is the zero-RL baseline across sizes plus the 3.5-27B versus 3.8-27B contrast; no fitted exponents.

## 8. Compute (measured and inferred)

Supply: Kaggle 21,600 GPU-seconds per week, refresh 2026-10-24, of which 934 s were lost to stuck sessions on 2026-10-17; the hardware is **two Tesla T4 (15 GB each), driver R580**, not an A100 (the shape request was ignored); batch runs with a hard timeout are the only safe route. 72,000 TPU-seconds exist but the Gated-DeltaNet stack cannot use them (inferred). No local GPU. Colab is now the primary route (user decision 2026-10-17); its tier is unknown until a tab connects.
Consequence (inferred): a T4 decodes a 4B model at roughly 1k tokens per second at best, so a thinking-mode RL step of about 1M tokens takes 15+ minutes and only about 15 steps fit in a Kaggle week. Answer-only RL (about 100 tokens per rollout, 256 rollouts per step) fits in about a minute per step on a T4, so it is the primary RL arm there; thinking-mode RL needs A100-class compute.
Demand (inferred, plus or minus 3x): baselines at 4B and 9B 2-6 GPU-h; SFT arms under 4 h at 4B; RFT sampling about 5 h; 4B RL about 24 h for 65-240 steps; 9B and 27B RL multiples of that. Roughly 40-60 GPU-h for the 4B study alone, so Kaggle needs 7-10 weeks without Colab or other credit.
Cheapest-first ordering and the gates exist for this reason: nothing is spent on a larger arm until the smaller one has earned it.

## 9. Risks

| Risk | Control |
|---|---|
| Recall mistaken for design | Temporal cohort + exposure covariates, copy-nearest and in-context arms, report pass@8 and maj@8 separately |
| Memorising about 240 answers | Val early stop; Train reward is not a metric |
| Reward gaming or collapse | Constant-answer bar, collapse score, valid-only spread |
| Library support for Qwen3.5/3.8 | Timeboxed stack check before any RL budget; ms-swift fallback |
| Colab or Kaggle disconnects | Checkpoint with optimizer state every 10 steps; resume test |
| Thin statistics | Sign-flip test, one confirmatory contrast, large effects only, limits stated |
| Reproducibility | Every run records git revision, data manifest hash, seed, library versions, hardware |

## 10. Defaults taken (reversible) and next actions

PD and combinations dropped; objective is held-out structure recovery; 0.8B/2B sweep is optional and last.
Next: hand-resolution and exposure files land; `data_prep` re-run; reward v2 plus tests plus Codex review; CPU controls scored; first Kaggle GPU session (stack check + 4B probe).
