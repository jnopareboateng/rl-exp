---
name: decision-reward-v2
description: 'Reward: 1.0 on connectivity match, else 0.05+0.6*Tanimoto+0.3*scaffold match; no similarity floor'
type: decision
updated: 2026-10-17
---
reward = 0 if the think block is unclosed or the answer is missing or unparseable; 1.0 on connectivity match; otherwise 0.05 + 0.6 * Tanimoto + 0.3 * [same non-empty Murcko scaffold]. Stereo is reported, never rewarded.

**Why:** 98.5% of wrong drug pairs have Tanimoto at or below 0.2, so a floor at 0.2 leaves only {0, 0.05, 1.0} and no within-group spread; random pairs of the 351 drugs used share an identical non-empty scaffold 0.03% of the time (20 of 61,425 pairs), so a scaffold hit is informative. The weights are provisional.
**How to apply:** tune weights only on Train rollouts and Val. The do-nothing bar is C0, the best constant: mean count-vector Tanimoto 0.176 and mean reward 0.156 on Val (0.143 is the bit-Tanimoto statistic that selects it; docs/plan.md section 5, "Which best constant"). A trained model must beat it on held-out drugs.

Related: [[decision-task-framing]]
