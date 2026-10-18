---
name: status-main
description: 'Repo is public with the open-release scope; paper final draft after three advisor rounds'
type: status
updated: 2026-10-18
---
Done and committed: tags `prereg-v1` and `prereg-v1-a1`; the locked Test run (4B, one attempt, both Val gates passed) scored; confirmatory contrast on Test-T RL-full 0.1857 against C3 0.1991, -0.0134, 10 wins 31 losses, p 0.061 (not rejected), reproduced by `scripts/recompute_confirmatory.py`; paper in `docs/paper/draft.md` after three advisor rounds; Val-only controls after the tag (shuffled 9B and 27B; profile-free RL R4 and R4b); cohort diagnostics; release tables, benchmark card and checklist published; the Colab runtime was released (8.7 logged GPU-hours).
Open for the user: whether the private rows, prompt files, runs and adapters may be published (rights not recorded), the venue, the full Drive, record gaps (trace authorship, prompt formats A/B/C/MO).
An external review (2026-10-18; `.agent-runs/advisor/fable_publication_review.md`, local only) rates the work a solid but narrow evaluation-methods contribution: preprint now, then TMLR or J. Cheminformatics, not a main track. Upgrades by value per cost: constant bar and rank readout on a public benchmark (CPU), a title and abstract that lead with the controls and the oracle-constant ceiling, the lost contrastive reward run on Val, more RL seeds. Not yet chosen.
Test is 4B only (owner decision).

**Why:** records where multi-session work stands so a fresh session can resume.
**How to apply:** delete this entry when the work merges into main-line results.
Published at `jnopareboateng/rl-exp` (public; main plus the two prereg tags) under the scope in [[decision-open-release-scope]].
