---
name: gotcha-core-endpoint-low-power
description: 'core@1 is near 0 for every arm at 4B-27B; the primary endpoint cannot separate arms yet'
type: gotcha
updated: 2026-10-17
---
On Val (20 drugs) exact@1 is 0 and core@1 is 0 to 5% for every arm measured: untrained 4B, 9B and 27B-FP8, 4B with retrieved examples, 4B after answer-only SFT, and 4B/9B told the drug's name. No arm beats the constant-answer bar (mean reward 0.156) at T=1.0; valid 27B answers average 0.18 Tanimoto, the constant's level. SMILES recall, not identification, is the first bottleneck at these sizes.

**Why:** measured 2026-10-17 (docs/plan.md section 5 table). With 41 Test-T drugs a sign-flip test on core@1 is decided by ties unless an arm wins by a huge margin; RL cannot create recall, only validity and class-level similarity.
**How to apply:** do not tag prereg-v1 until the primary endpoint question in plan section 2 ("Open before the lock") is decided; compare arms on mean Tanimoto and mean reward with the C0/C1 controls as gates, and report core@1 and exact@1 as headline recovery rates; fix one decoding policy per arm on Val.

Related: [[decision-controls-first-design]], [[gotcha-base-model-verbosity]]
