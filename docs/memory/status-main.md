---
name: status-main
description: 'Val phase done and committed; RL replicate R5 training on runtime 2; freeze, tag and Test run next'
type: status
updated: 2026-10-17
---
Done and committed: Val baselines for every arm with the full temperature grid, RL-full steps 10 to 40, RL-contrastive steps 10 and 20 (cut short), sampler-seed replicates, shuffled and blank profile runs, answer diversity, extractor audit pinned to 29 runs, independent reviews (SFT, metrics, RL rounds 3 and 4, evaluation pipeline round 1) applied. Selection so far: comparator C3 (Val 0.1865), A5 = RL-full step 40 at T=0.6 (0.1887), A0n T=0.6, A4 T=0.6, A2 T=1.0.
In flight: the replicate RL-full run R5 (seed 1, up to 100 steps) on Colab runtime 2 (A100, started 22:06 UTC); its checkpoints from step 40 become RL-full candidates (vLLM Val sampling at T 0.3, 0.6, 1.0 is chained after it); then the validity-only and profile-free controls (Val only, may be interrupted for the Test run). Drive has under 4 GB free (deleted files stay in the trash): checkpoints are excluded from the sync, so R5 cannot resume after a runtime loss, only its adapters survive.
Next: rerun `scripts/select_arms.py` with the R5 candidates, write `results/frozen_arms.json` (+ sha256 file), regenerate `data/MANIFEST.sha256`, independent round-2 review of the evaluation pipeline, tag `prereg-v1`, then follow `gpu/FINAL.md` (merge, Val sanity run, sampling, `scripts/final_controls.py`, `scripts/confirmatory.py`, tables). Update `docs/paper/draft.md` Results 5.4 and Discussion afterwards.
Omitted for budget unless time allows: dense RFT, continued SFT, profile-plus-name; RL-cold.

**Why:** records where multi-session work stands so a fresh session can resume.
**How to apply:** delete this entry when the work merges into main-line results.
