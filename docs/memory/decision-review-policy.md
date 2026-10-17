---
name: decision-review-policy
description: 'Reviews are same-vendor now; reviewers list GPU checks, the lead runs them on Colab'
type: decision
updated: 2026-10-17
---
Non-trivial code (reward, metrics, splits, training packs) still gets an independent review before it is relied on, but the cross-vendor reviewer's credits ran out on 2026-10-17, so later rounds use a separate reviewer session with a fresh context. It gets the code and the first review's findings, not the author's conclusions. Five cross-vendor rounds were completed before that; the paper states the change as a limitation. Reviewers run CPU probes and mutations in scratch dirs under .agent-runs/; anything needing a GPU is listed in their report and run by the lead on the Colab A100 (there is no local GPU).

**Why:** the user ruled out further cross-vendor reviews and asked that all GPU work use Colab.
**How to apply:** brief names the first review's findings and asks to verify each; require mutation checks; require the report in .agent-runs/reviews/; add the round to docs/plan.md's review log; never let a reviewer use the Colab runtime while a training run is active.

Related: [[env-colab-a100-runtime]]
