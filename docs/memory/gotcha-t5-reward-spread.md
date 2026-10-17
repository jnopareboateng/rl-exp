---
name: gotcha-t5-reward-spread
description: 'T5 fails on Val: valid-answer reward std >= 0.05 in at most 15.5% of rows (A2 at T=1.0: 5.2%)'
type: gotcha
updated: 2026-10-17
---
Among the valid finished samples of a prompt, the reward std is at least 0.05 in at most 15.5% of the 58 Val rows for any non-RL arm (A2: 5.2% at T=1.0, 8.6% at 0.3, 15.5% at 0.6) and in 0% to 5.2% for the RL-full checkpoints s20, s30, s40; the plan's T5 bar is 30%. Counting all 8 samples with invalid ones at reward 0, which is what an RL update sees, 91% of A2 rows at T=1.0 reach 0.05, against 31%, 17% and 5% for RL-full s20, s30, s40: the within-group signal is mostly validity, and it shrinks as RL proceeds (inferred).

**Why:** results/t5.json from scripts/offline_audit.py on Val samples (no Train rollouts of a SMILES-answering model exist). Valid answers cluster at similar Tanimoto, and a reward std of 0.05 needs a Tanimoto std of 0.083.
**How to apply:** read an RL gain in fv_tanimoto beside the validity-only run (RL-validity) before calling it graded learning; rerun scripts/offline_audit.py on RL checkpoints' samples; the plan's T5 gate is failed by every measured arm, so it needs a decision before it is cited as passed.
