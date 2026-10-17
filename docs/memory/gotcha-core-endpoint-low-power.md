---
name: gotcha-core-endpoint-low-power
description: 'core@1 is near 0 for every arm at 4B-27B and untestable; the endpoint is fv_tanimoto instead'
type: gotcha
updated: 2026-10-17
---
On Val (20 drugs) exact@1 is 0 and core@1 is 0 to 5% for every arm measured: untrained 4B, 9B and 27B-FP8, 4B with retrieved examples, 4B after answer-only SFT, and 4B/9B told the drug's name. Every positive core@1 score on Val (29 runs, RL-full checkpoints included) sits in one 4-drug component, so a sign-flip test on it cannot reach p <= 0.05 (it needs six nonzero component differences). No arm beats the constant-answer bar (C0: Tanimoto 0.176, mean reward 0.156) at T=1.0; valid 27B answers average 0.18 Tanimoto, the constant's level. SMILES recall, not identification, is the first bottleneck at these sizes.

**Why:** measured 2026-10-17 (docs/plan.md section 5, results/val_baselines.json). With 41 Test-T drugs a sign-flip test on core@1 is decided by ties unless an arm wins by a huge margin; RL cannot create recall, only validity and class-level similarity.
**How to apply:** the endpoint is decided: fv_tanimoto (decision-endpoint-fv-tanimoto), locked at prereg-v1. Report core@1 and exact@1 as recovery rates, never as the tested endpoint; use C0 and C1 as the do-nothing bar; fix one decoding policy per arm on Val.

Related: [[decision-controls-first-design]], [[gotcha-base-model-verbosity]]
