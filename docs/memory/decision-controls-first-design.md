---
name: decision-controls-first-design
description: 'Controls-first study: one endpoint, one confirmatory contrast, sign-flip test, prereg tag'
type: decision
updated: 2026-10-17
---
The study asks whether online RL beats cheaper arms (constant and copy-nearest controls, base thinking on/off, in-context retrieval, answer-only SFT, trace SFT, rejection-sampling FT) on one endpoint (first core@1, now fv_tanimoto: decision-endpoint-fv-tanimoto).
One confirmatory contrast on Test-T, decided by an exact two-sided sign-flip test; everything else is exploratory. The design is frozen under git tag prereg-v1 before any model samples a Test split. The CPU controls C0 to C2 were scored on all splits by the controls CLI; no choice used their Test values (docs/prereg-v1.md section 3).

**Why:** an independent review judged RL as originally planned no-go: 43 candidate test drugs (41 after the regulator audit) give power only for large effects, endpoints were selectable, and no cheap comparator existed. A negative result for RL is still a publishable result.
**How to apply:** decisions use Train rollouts of untrained models and Val only; never tune on Test; report every deviation from prereg-v1.

Related: [[decision-split-design]]
