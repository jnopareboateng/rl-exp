---
name: decision-endpoint-fv-tanimoto
description: 'Confirmatory endpoint is fv_tanimoto: Tanimoto of the first valid answer of 8; lift is a diagnostic'
type: decision
updated: 2026-10-17
---
The single confirmatory endpoint is fv_tanimoto: per prompt row, the count-vector Tanimoto of the first valid, closed, finished answer among the first 8 samples (0 if none), drug-balanced; cost fv_samples reported beside it. Reported but not tested as the primary: expected Tanimoto over all attempts, validity, core@1, exact@1, and the profile-specific lift (own-reference Tanimoto minus mean Tanimoto to the other drugs' references; fv_lift for the first-valid policy).

**Why:** exact@1 = 0 and core@1 <= 5% for every arm; mean Tanimoto rewards syntax and generic molecules; valid-only means let arms drop hard cases (rejected in review); lift looked ideal (constant = 0) but executed counterexamples show: a fixed molecule scores +0.18 when only one drug finishes and a prompt-length policy +0.71, so it measures association with the drug, nuisance cues included. First-valid-of-8 compares arms at matched validity and states the cost.
**How to apply:** the word "profile" in a claim needs lift above the nearest-profile copy control C1 (+0.022 on Val) and a matched shuffled-or-removed-profile comparison; report fv_samples; choose each arm's temperature on Val by fv_tanimoto; the endpoint is still to be locked at prereg-v1.

Supersedes: core@1 as the single primary (decision-controls-first-design, 2026-10-17). Related: [[gotcha-core-endpoint-low-power]]
