---
name: decision-comparator-c3-registered
description: 'C3 joins the comparator set and A1 leaves it; prereg deviations D1 and D2, before any Test sample'
type: decision
updated: 2026-10-17
---
The confirmatory comparator set is {C0, C1, C3, A0n, A2, A4}, the best by Val fv_tanimoto at its Val-selected decoding (ties to the smaller arm id). C3 is the reward-optimal constant chosen from Train plus a leakage-filtered ChEMBL pool; on Val it scores 0.1865, against RL-full's 0.1887, so the contrast is a near tie by design. A1 (trace SFT) is dropped from the set and from Test: its valid-answer similarity is 0.152 on Val, which bounds its first-valid similarity below C3 and A2, and 8,192-token thinking per sample was not budgeted.

**Why:** an independent review found that C3 was chosen as comparator but absent from the first protocol draft, and A1 was named but absent from the selection; the change is conservative (it hardens the bar for RL) but must be registered, not adopted quietly.
**How to apply:** docs/prereg-v1.md sections 3, 4 and 9 (D1, D2) state it; tests/test_prereg_consistency.py ties `comparator_set` in results/arm_candidates.json to the machine-readable line in the protocol.

Related: [[decision-controls-first-design]], [[decision-endpoint-fv-tanimoto]]
