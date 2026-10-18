---
name: status-main
description: 'Test run done and analysed, paper final-draft after three advisor rounds; handoff to the user'
type: status
updated: 2026-10-18
---
Done and committed: tags `prereg-v1` and `prereg-v1-a1`; the locked Test run (4B, one attempt, both Val gates passed) scored; confirmatory contrast on Test-T RL-full 0.1857 against C3 0.1991, -0.0134, 10 wins 31 losses, p 0.061 (not rejected); paper restructured in `docs/paper/draft.md` (Test results first, Val in Appendix A, deviations in Appendix D); Val-only controls after the tag (shuffled 9B and 27B; profile-free RL R4 and R4b, two seeds); exploratory cohort diagnostics (`scripts/cohort_diagnostics.py`); release tables, benchmark card and controls checklist built; the Colab runtime was released (A100 attached 15.1 h in two runtimes; 8.7 logged GPU-hours).
Advisor round 3 fixes are applied (title and abstract carry the cohort qualifier; blank-profile contrast, Holm line, C3 naming clause, coverage column, deviations F1 and F2); an independent numpy recomputation (`scripts/recompute_confirmatory.py`) reproduces the confirmatory contrast. Open for the user: licence and publication of `release/` (ChEMBL, openFDA, PubChem terms unchecked; `refs.jsonl` already tracks the Test structures), the Drive is full (empty the trash), unanswered record gaps (trace authorship, prompt formats A/B/C/MO, Colab unit balance).
Scope decisions by the project owner: Test is 4B only; nothing is pushed.

**Why:** records where multi-session work stands so a fresh session can resume.
**How to apply:** delete this entry when the work merges into main-line results.
