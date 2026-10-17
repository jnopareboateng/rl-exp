---
name: status-main
description: 'In flight: RL pilot on Colab; prereg-v1 not tagged, answer-reader decision open'
type: status
updated: 2026-10-17
---
Done: repo, plan v5, contracts, CPU pipeline and controls, sampling and SFT packs, Val baselines (4B, 9B, 27B-FP8, ICL, name-given, identification, A2 and A1 SFT). Decided, pending the lock: endpoint fv_tanimoto, decoding grid {0.3, 0.6, 1.0} at k=8, RL arm names (docs/prereg-v1.md). Measured on Val: the answer-reader audit (2.2% of samples differ from the last-SMILES-line rule, above the 1% threshold; none in the RL-full checkpoints) and T5 (fails for every arm). Colab A100 with Drive mounted is the compute route.
In flight: the answer-only RL pilot (own loop, HF generation; gates passed on the A100).
Not done: the keep-or-switch decision on the answer reader, prereg-v1 tag, controls of plan 2(d) beyond validity-only RL (profile-free RL, continued SFT, dense RFT, profile-plus-name), A3 and A5 results, any model sampling on a Test split. The CPU controls were scored on all splits before the lock; no choice used their Test values (prereg section 3).

**Why:** records where multi-session work stands so a fresh session can resume.
**How to apply:** delete this entry when the work merges into main-line results.
