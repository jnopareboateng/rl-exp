---
name: status-main
description: 'Val phase done and committed; RL replicate R5 training on runtime 2; freeze, tag and Test run next'
type: status
updated: 2026-10-17
---
Done and committed: Val baselines for every arm with the full temperature grid, RL-full steps 10 to 40, RL-contrastive steps 10 and 20 (cut short), sampler-seed replicates, shuffled and blank profile runs, answer diversity, extractor audit pinned to 29 runs, independent reviews (SFT, metrics, RL rounds 3 and 4, evaluation pipeline round 1) applied. Selection (final on Val): comparator C3 (Val 0.1865), A5 = RL-full replicate R5 step 50 at T=0.3 (0.1949), A0n T=0.6, A4 T=0.6, A2 T=1.0.
R5 (seed 1) stopped itself on its collapse rule at step 63 (adapters for steps 10 to 60 on Drive); its steps 40, 50, 60 were sampled on Val at the three temperatures. The validity-only control R2 finished (Val 0.078: trivial valid answers). Scope decision by the project owner (2026-10-18): the locked Test run covers the 4B arms only; the 9B and 27B references stay Val-only. An external advisor (Fable) is consulted on priorities and the paper; its reports are in .agent-runs/advisor/. Drive has under 4 GB free (deleted files stay in the trash): checkpoints are excluded from the sync, so R5 cannot resume after a runtime loss, only its adapters survive.
Next: tag `prereg-v1` (message: `prereg-v1 spec=<sha256> selection=<sha256> manifest=<sha256>`), bundle at the tag, `python gpu/run_final.py` per `gpu/FINAL.md` (4B only, about 40 minutes), `scripts/final_controls.py`, `scripts/confirmatory.py`, tables (merge, Val sanity run, sampling, `scripts/final_controls.py`, `scripts/confirmatory.py`, tables). Update `docs/paper/draft.md` Results 5.4 and Discussion afterwards.
Omitted for budget unless time allows: dense RFT, continued SFT, profile-plus-name; RL-cold.

**Why:** records where multi-session work stands so a fresh session can resume.
**How to apply:** delete this entry when the work merges into main-line results.
