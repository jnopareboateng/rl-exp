---
name: reference-compute-budget
description: 'Per-stage GPU hours: scripts/compute_budget.py reads run records; runtime windows are hand-kept'
type: reference
updated: 2026-10-17
---
`scripts/compute_budget.py` writes `results/compute_budget.json` and `docs/paper/compute_budget.md`: SFT hours from each run's summary.json (`training_seconds`), RL hours from metrics.jsonl plus the Val evaluations in evals.jsonl, vLLM sampling hours from every `runs/colab-s1/*/meta.json` and `runs/final/*/meta.json`. Colab runtime windows live in `results/compute_sessions.json` (hand-kept from the session log: runtime 1 12:10 to 20:35 UTC on 2026-10-17, runtime 2 from 20:57). SFT summaries are copied from Drive into `runs/sft/{A1,A2}/summary.json`; RL summaries from `runs/rl/R*/` (copy each new run's summary, metrics and evals down from Drive or Colab before the VM goes).
Not counted: weight downloads, installs, LoRA merges (about 80 s each), CPU stages, and agent time (about 24 worker-hours by 2026-10-17 22:30, summed over parallel workers, from the session log only).

**Why:** the paper's reproducibility statement needs hours per stage, and several early runs have no meta.json, so the numbers must say what is missing rather than estimate it.
**How to apply:** rerun the script after the Test run and after copying the last RL summaries; quote the "Total logged" row with its caveats, never a rounded wall-clock total.

Related: [[env-colab-a100-runtime]], [[gotcha-rl-checkpoint-needs-drive-sync]]
