---
name: status-main
description: 'Test run done and analysed, paper has results; R4 (Val-only) running; review and polish next'
type: status
updated: 2026-10-18
---
Done and committed: tags `prereg-v1` (spec, selection, manifest hashes) and `prereg-v1-a1` (addendum: rank readout, random-pair calibration, S1 to S4, P1/P2). The locked Test run (4B arms, 10 arms, one attempt, both Val gates passed) is scored; results are in `results/` (confirmatory.json, paired_test_*.json, rank_*.json, test_*_baselines.json) and `docs/paper/draft.md` section 5.5. Confirmatory contrast on Test-T: RL-full 0.1857 against C3 0.1991, difference -0.0134, 10 wins 31 losses, p 0.061 (not rejected, point estimate for the constant). Profile dependence: Val and Test-R yes (about +0.01 RL), Test-T not detected; the untrained 9B and 27B depend on the profile more on Val (+0.040, +0.049; Val-only shuffles).
`scripts/rank_readout.py` was edited after the addendum tag on an independent review (guards, wording; no number of a complete run changes): Table 7 of the paper lists it as deviation A2.
Running or pending: profile-free RL R4 on the Colab runtime (Val only; `.agent-runs/p36.py`, output `.agent-runs/p36.out`, then merge and Val re-samples into `.agent-runs/v30/`); download into `runs/rl/R4` and `runs/colab-s1`, add rows to `scripts/val_table.py`, regenerate `results/val_baselines.json`, fill the [TBD] marks of the paper (abstract, 5.3, 6), update `docs/controls_checklist.md` item 8 and `docs/benchmark_card.md` with Test values if the Test controls are released with the paper, Fable round 2 report in `.agent-runs/advisor/fable_round2.md`.
Scope decisions by the project owner: Test is 4B only (the 9B and 27B stay Val-only); nothing is pushed.

**Why:** records where multi-session work stands so a fresh session can resume.
**How to apply:** delete this entry when the work merges into main-line results.
