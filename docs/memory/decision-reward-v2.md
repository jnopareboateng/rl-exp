---
name: decision-reward-v2
description: 'Reward: 1.0 on connectivity match, else 0.05+0.6*Tanimoto+0.3*scaffold match; no similarity floor'
type: decision
updated: 2026-10-17
---
reward = 0 if the think block is unclosed or the answer is missing or unparseable; 1.0 on connectivity match; otherwise 0.05 + 0.6 * Tanimoto + 0.3 * [same non-empty Murcko scaffold]. Stereo is reported, never rewarded.

**Why:** 98.5% of wrong drug pairs have Tanimoto at or below 0.2, so a floor at 0.2 leaves only {0, 0.05, 1.0} and no within-group spread; random pairs share a scaffold 0.12% of the time, so a scaffold hit is informative. The weights are provisional.
**How to apply:** tune weights only on Train rollouts and Val; the best constant answer scores about 0.14 raw Tanimoto, the do-nothing bar a trained model must beat on held-out drugs.

Related: [[decision-task-framing]]
