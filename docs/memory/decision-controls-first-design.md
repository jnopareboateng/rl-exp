---
name: decision-controls-first-design
description: 'Controls-first study: one endpoint (core@1), one confirmatory contrast, sign-flip test, prereg tag'
type: decision
updated: 2026-10-17
---
The study asks whether online RL beats cheaper arms (constant and copy-nearest controls, base thinking on/off, in-context retrieval, answer-only SFT, trace SFT, rejection-sampling FT) on core@1.
One confirmatory contrast on Test-T, decided by an exact two-sided sign-flip test; everything else is exploratory. The design is frozen under git tag prereg-v1 before any Test-split sampling.

**Why:** an independent review judged RL as originally planned no-go: 43 test drugs give power only for large effects, endpoints were selectable, and no cheap comparator existed. A negative result for RL is still a publishable result.
**How to apply:** decisions use Train rollouts of untrained models and Val only; never tune on Test; report every deviation from prereg-v1.

Related: [[decision-split-design]]
