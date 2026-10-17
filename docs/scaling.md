# Scaling laws, applied to this project

Data handling and the step-by-step plan live in `docs/plan.md`. This file explains the scaling reasoning only.
Labels: **measured**, **recalled** (papers from memory; the RL-compute paper was confirmed by search summaries, not read in full), **inferred**.

## Three axes, three different laws

Held-out performance depends on model size N, RL compute C, and the number of distinct training drugs D. Do not mix the laws.

**N: pretraining scaling (recalled).** Loss falls as a power law in parameters and tokens (Kaplan et al.; Hoffmann et al.). We do not pretrain,
so this sets what each rung brings in: chemistry, SMILES syntax, recall of approved drugs. The answer here is one known drug, so a large share of
the gap between sizes will be knowledge, not RL. Measured by the zero-RL baselines. Caveat (measured): the ladder mixes generations (3.5 for 4B and 9B,
3.8 for the 27B), so a 3.5-27B baseline is included to separate size from generation.

**D: fine-tuning data (recalled, then inferred).** Hernandez et al. 2021 ("Scaling laws for transfer") fit the effective data transferred from pretraining as a power law in
fine-tuning data and in model size, with larger models transferring more. It was measured on loss in unsupervised fine-tuning (text to code), not on RL.
Applying it here is inference. What is measured: D is about 240 distinct training molecules (final after hand resolution), because rows are repeated views of the same answer.
That is small enough that memorisation, not compute, is the likely ceiling; the train-versus-val gap is the diagnostic, not a separate ablation.

**C: RL compute scaling (recalled; Khatri et al. 2025, arXiv 2510.13786, "The Art of Scaling RL Compute").** Reward against compute follows a saturating sigmoid:

    R(C) = R0 + (A - R0) / (1 + (C_mid / C)^B)

R0 is the starting reward, A the ceiling, B the steepness, C_mid the midpoint. Per the summaries I found, the loss type (CISPO) and an FP32 LM head raise the ceiling A
(the FP32 head from about 0.52 to 0.61), while loss aggregation, normalisation and curriculum mostly change speed. Their fits used thousands of GPU-hours.
Two companions (recalled): RL sharpens what the base can already sample, so headroom is roughly pass@k minus pass@1 (Yue et al. 2025); and
RL carries little information per rollout, so LoRA matches full fine-tuning in RL, with LoRA on all layers including MLP and a learning rate about 10 to 15 times higher (Thinking Machines, 2025).

## What follows for this task

- **No curve fits.** Our budget is tens to a couple of hundred steps per size (docs/plan.md section 8), two orders of magnitude below the paper's regime.
  We report one paired delta per size with intervals. Sigmoid language is background, not an analysis plan.
- **The measurable scaling law is the zero-RL baseline.** Core-match and exact-match pass@1 and pass@8 against size over 0.8B to 27B, plus the 3.5-27B versus 3.8-27B contrast.
  Expect most of the size effect there, and expect it to be steeper on older, well-known drugs than on Test-T (the 2025-2026 temporal cohort; approval date is not disclosure date, so it is not proof of unseen molecules).
- **Train reward is not a metric.** With about 240 answers to learn, expect gains from format, validity and class-level priors, then train reward rising by memorisation while held-out stalls.
- **Cost is proportional to N for dense models (measured: all four are dense).** Relative 1 : 2.3 : 6.8 for 4B : 9B : 27B. Generation dominates and is memory-bound.
- **Memory, from config.json (inferred arithmetic).** Only every 4th layer is full attention: KV cache 32 KB per token for 4B and 9B, 64 KB for 27B; 64 sequences of 8k tokens is about 17 GB and 34 GB.
  Logits are the hidden cost: 248,320 vocab x 8,192 tokens is 4 GB per sequence in bf16 and 8 GB in FP32, so a chunked log-prob loss is required.
