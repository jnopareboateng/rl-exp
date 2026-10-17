---
name: decision-decoding-grid-k8
description: 'Selectable temperatures {0.3, 0.6, 1.0} at k=8; 0.01 is a k=1 near-greedy report, never selected'
type: decision
updated: 2026-10-17
---
Each stochastic arm's temperature is chosen on Val from {0.3, 0.6, 1.0} at k=8 by fv_tanimoto, then frozen. Near-greedy T=0.01 is reported at k=1 as "near-greedy" and is never selectable. RL runs evaluate Val at EVAL_K=8 from now on (the first pilot used 4), so the in-training Val fv_tanimoto has the endpoint's k.

**Why:** first-valid-of-8 needs eight draws, and eight near-greedy samples of a prompt are nearly one; plan, protocol and the Test driver had disagreed (0.01 at k=8, at k=1, at k=1 to 8).
**How to apply:** docs/plan.md section 2(c) and docs/prereg-v1.md section 5 state it; keep gpu/final_eval.py (0.01 only at k=1) and the EVAL_K default in gpu/run_rl.sh, gpu/rl_grpo.py and gpu/RL.md in line with it.
