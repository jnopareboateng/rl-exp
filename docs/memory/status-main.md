---
name: status-main
description: 'Val phase done; Colab A100 runtime lost 2026-10-17 20:35; Test evaluation and the tag wait on a GPU'
type: status
updated: 2026-10-17
---
Done on Val: controls C0 to C3 (C3 = reward-optimal constant, Val 0.187), untrained 4B/9B/27B sweeps with a temperature grid, A1/A2/A4, identification and name-given, RL-full (40 steps: first-valid similarity 0.189 at step 40, equal to C3; lift not raised), the class-information ceiling, extractor audit (reader kept) and T5. Preliminary selection (results/selection.json): comparator C3, RL arm RL-full step 40 at T=0.6.
Lost with the A100 VM: the RL-contrastive run (about step 28 of 40; its Drive-synced evals at steps 10 and 20 gave first-valid 0.130 and 0.120, below the A2 start 0.148), the validity-only run and the queued SFT variants. Adapters (A2, A1, R1 step-10..40) should be on Drive MyDrive/rl-exp.
Next: user sets the runtime to an A100 and mounts Drive; rebuild with gpu/bootstrap_colab.sh from a bundle (scripts/make_bootstrap_bundle.py); diagnostics (shuffled and blank profile, A4 temperatures); freeze spec; tag prereg-v1; Test run through gpu/final_eval.py; scripts/confirmatory.py.
Omitted for budget unless time allows: profile-free RL, dense RFT, continued SFT, profile-plus-name.

**Why:** records where multi-session work stands so a fresh session can resume.
**How to apply:** delete this entry when the work merges into main-line results.
