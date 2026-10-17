# Paper scaffold

Working title: *How much of a drug's structure can language models recover from its pharmacology profile? A controls-first evaluation with deployable and profile-association metrics, and a bounded RL case study.*
The paper is written from this repository: every number in it must trace to a file in `results/` produced by a command in the README, at a recorded git revision and data hash. Write the plan section from `docs/plan.md` as frozen at tag `prereg-v1`, and list every later deviation.

## Research question and hypotheses

Question: beyond cheaper alternatives, does online RL (GRPO-family, verifiable similarity reward) improve held-out structure recovery at matched inference cost?
- H1 (confirmatory, Test-T, fv_tanimoto): the Val-selected RL checkpoint (RL-full or RL-contrastive) beats the best non-RL arm in delivered similarity (Tanimoto of the first valid answer of up to 8 samples) at a stated cost. Decided by the exact cluster-level sign-flip test at alpha 0.05. Profile language needs a positive lift above the nearest-profile control and a matched shuffled-profile comparison.
- H2 (exploratory): any gain persists on Test-R and is not concentrated in the high-exposure tercile (PubChem creation year, PubMed count).
- H3 (exploratory): zero-RL baselines improve with model size (4B, 9B, 3.5-27B, 3.8-27B) in validity, delivered similarity and lift; identification of the drug (profile to name) is the part that grows most; RL gain, if any, shrinks with baseline strength.
- Null is a result: if H1 fails, the contribution is the benchmark, the controls, the SFT/RFT results and the measured limits of RL at this data scale (about 240 training molecules).

## Claims and evidence (update as results land; status: planned, running, done, refuted)

| Claim | Evidence artifact | Produced by | Status |
|---|---|---|---|
| The reference answer in the SFT data is the real drug: same connectivity (strict key) in 98-99% of 942 rows (row level); `sft_agrees` is 338 of 339 drugs by a tolerant skeleton rule (drug level); the ChEMBL release behind the references was not recorded | `data/processed/refs.jsonl`, `split_report.json` | `scripts/resolve_refs.py`, `rlexp.data_prep` | done (rerun pending) |
| Splits are leakage-safe (no pair at Tanimoto >= 0.4 or shared scaffold across splits); Test-T is 41 drugs (100 rows) in 41 different components, 8 of which also hold buffer drugs | `split_report.json`, `refs.jsonl` | `rlexp.data_prep` | in progress |
| Test-T is a temporal cohort, not unseen molecules: PubChem records of 40 of 41 cohort drugs were created 3 to 20 years before approval (median 10.5); the audit checked approval dates, not disclosure | `data/curated/exposure.json`, `approval_audit_notes.md` | `data/processed/refs.jsonl` exposure fields (computed ad hoc, no results file yet) | done |
| Constant and copy-nearest controls set the do-nothing bar (C1 has lift +0.022 at constant-level Tanimoto on Val) | `results/val_baselines.json` | `rlexp.controls`, `rlexp.metrics`, `scripts/val_table.py` | done on Val |
| Class facts (ATC, mechanism, target, indication) predict structure little beyond a constant once neighbours respect the split | `results/class_ceiling.json` | `scripts/class_ceiling.py` | done; component-matched: class facts do not beat a constant |
| Identification (profile to name) grows with size: 1.0% (4B), 2.1% (9B), 8.1% / 4.2% (Qwen3.5 / 3.8 27B-FP8) on Val | `runs/colab-s1/ident-*` | `rlexp.ident` | done on Val |
| Base model zero-shot level by size (no model recovers a structure; lift grows from +0.005 at 4B to +0.027 at 27B) | `results/val_baselines.json` | `gpu/sample_vllm.py`, `rlexp.metrics` | done on Val (Test after the lock) |
| In-context retrieval, answer-only SFT, trace SFT levels (A1 trace SFT: 33% valid at 2,500 tokens per answer; A2: 56% valid) | `results/val_baselines.json` | training scripts, `rlexp.metrics` | done on Val; RFT planned |
| H1: RL vs best non-RL arm on Test-T (fv_tanimoto) | `results/confirmatory.json` | `rlexp.metrics report --vs` | planned |
| Answer-reader audit (fixtures plus real-rollout disagreement rate): 2.2% of 10,304 Val samples (3.0% of the 7,520 from non-RL runs, none from the RL-full checkpoints) differ between `extract_answer` and the last-SMILES-line rule, above the 1% switch threshold; 0.04% return different molecules | `tests/fixtures/answer_formats.json`, `results/extractor_audit.json` | tests, `scripts/offline_audit.py` | done on Val; keep-or-switch decision before the freeze |
| Within-prompt reward spread among valid answers is small: T5 (std >= 0.05 in 30% of rows) fails for every arm (best 15.5%; A2 at T=1.0 5.2%; RL-full checkpoints at most 5.2%) | `results/t5.json` | `scripts/offline_audit.py` | done on Val |

## Methods outline

1. Data: source, size, the finding that answers equal real drugs, exclusions (PD, combinations, unresolved), reference construction and hand-resolution sources, approval dating (ChEMBL, openFDA, regulator audit), exposure covariates.
2. Splits: component rule, temporal cohort, buffer, leakage numbers.
3. Reward and scoring: definition, weights, why no floor, metrics (fv_tanimoto primary; core@1, exact@1 and the rest reported), extractor and its measured failure modes.
4. Arms and training recipes with every hyperparameter; compute and hardware (Kaggle/Colab GPU-hours actually used).
5. Statistics: unit of analysis, cluster bootstrap (descriptive), sign-flip test (decision), single seed and what that leaves unmeasured.

## Tables and figures (generated by scripts, never by hand)

T1 data and split counts; T2 controls and baselines by size; T3 arms at 4B on Val and Test (fv_tanimoto with fv_samples, expected Tanimoto, validity, lift, core@1, exact@1, stereo-exact, maj@8, pass@8, tokens per answer); T4 paired contrast with wins, losses, ties and p-value.
F1 approval-year histogram by split; F2 fv_tanimoto versus tokens per answer for all arms; F3 RL training curves (Val fv_tanimoto, train reward, collapse score); F4 Test-T by exposure tercile.

## Threats to validity (keep current)

Approval date is not disclosure date: Test-T is temporal, not proof of unseen molecules, and PubChem records of 40 of its 41 drugs predate approval by 3 to 20 years. One training seed per arm. 41 Test-T drugs detect only large effects (minimum detectable fv_tanimoto difference about 0.015 to 0.036, `docs/plan.md` section 2, Power). Teacher traces were written with the answer known. Similarity-based reward can be gamed by plausible neighbours; stereo is not rewarded. Hardware limits constrain size and step counts. Knowledge cutoffs of the models are undocumented.

## Reporting conventions

Report negative and exploratory results; label every comparison confirmatory or exploratory; state compute spent; cite related work from `related-work.md`; keep the review log (two independent reviews of the plan) in the appendix.
