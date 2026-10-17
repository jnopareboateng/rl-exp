---
name: gotcha-rl-checkpoint-needs-drive-sync
description: 'A lost Colab VM kills RL resume unless the whole run dir, checkpoints included, is synced to Drive'
type: gotcha
updated: 2026-10-17
---
R1 (RL-full, 40 steps) and R3 (contrastive, cut at step 28) lost their resumable checkpoints when the first A100 VM disappeared at 20:35: only `adapters/step-N`, `evals.jsonl`, `metrics.jsonl` and `summary.json` had been copied to Drive, not `checkpoints/` (optimizer moments and RNG states). A fresh run can start from an adapter, but not continue the same run, and the runner refuses a resume across a changed objective hash.

**Why:** R3 could not be finished and R1 could not be extended past step 40 after the VM loss.
**How to apply:** during any RL run keep a loop that rsyncs the whole `<out>` directory (checkpoints included, about 1 GB) to Drive every few minutes; p34.py of the 2026-10-17 session does it. Before relying on a resume after a VM change, check that `<out>/checkpoint.json` exists.

Related: [[env-colab-a100-runtime]], [[gotcha-agent-restarts]]
