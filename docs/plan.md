# Plan v4: does online RL add value for structure recovery? A controls-first study (Qwen3.5-4B, 9B, Qwen3.8-27B)

Status: v4, 2026-10-17. Supersedes v3 after two independent reviews (Fable 5.1, Codex GPT-6.1 Sol). Freeze point: git tag `prereg-v1`, set before any Test-split sampling.
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

0. Runtime inventory through the Colab or Kaggle MCP. Kaggle is authenticated (6 GPU-hours per week, measured); Colab needs the user to accept the connection.
1. Stack check, one timeboxed session, using `gpu/STACK.md` (vLLM 0.31.0, torch 2.13.0, Transformers 5.17.0, driver R575 or newer; P100 unsupported; T4 only for 4B, fp16): first GPU session = the ten ordered commands there, then a 4B sampling probe.
2. A0 and A0n on Val plus a Train subset (k=4 to 8): valid-and-closed rate, lengths, truncation, tokens per second. **G0:** core-match pass@8 minus pass@1 is 5 points or more on Val + Train subset (RL has headroom); **G1:** valid-and-closed 80% or more at the chosen cap and T5 holds.
3. A4 (inference only), A2, A1 (LoRA SFT, microbatch 1, packing off), A3 (sampling on Train, then SFT). Val scoring only; Test untouched.
4. Template check: Qwen3.5 native template with thinking on versus the dataset's `<think>` placement.

## 6. Phase 3: RL pilot (A5)

Stack: TRL 1.16.0 on an A100 first, ms-swift 4.6.0 as fallback (separate environment); choice by the first probe. Starting recipe, recalled from ScaleRL, DAPO and LoRA Without Regret and confirmed per stack in `gpu/STACK.md`:
CISPO (`loss_type="cispo"`), beta 0, prompt-level loss aggregation, batch-level advantage normalisation on centred advantages, FP32 LM head with chunked log-probs (needs a hook for tied-embedding 4B), drop zero-variance groups,
overlong samples masked not zero-scored, LoRA on all linear layers including GatedDeltaNet projections, rank 32-64, learning rate near 1e-5, group 8, temperature 1.0, cap 8k. Sweep at most 3 settings, on Val.
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
