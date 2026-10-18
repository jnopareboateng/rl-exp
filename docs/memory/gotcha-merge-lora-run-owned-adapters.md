---
name: gotcha-merge-lora-run-owned-adapters
description: 'merge_lora treats <run>/adapters/step-N as run-owned: use plain adapter copies with revision.json'
type: gotcha
updated: 2026-10-18
---
`gpu/merge_lora.py` (SFT round 3) recognises `<run>/adapters/epoch-N` and `<run>/adapters/step-N` as adapters owned by a run and then requires them to be in that run's committed checkpoint state (`checkpoint.json` plus `checkpoints/*/state.json`) with matching hashes. Deleting `checkpoints/` of a finished run (to free 0.7 GB) makes every `step-N` adapter of it unmergeable. A plain copy in another directory (`.agent-runs/adapters_plain/<name>`, any name that is not `epoch-N` or `step-N`) merges without that check, and uses the base revision from its `revision.json`; without that file the merge silently takes the hub's current main.

**Why:** found 2026-10-18 while freezing the locked evaluation; the frozen spec therefore names plain copies `A2-epoch1` and `R5-step-50` (three files each, hashes identical to the run's).
**How to apply:** before deleting a run's checkpoints, copy the adapters you still need to a plain directory including `revision.json`; freeze the adapter path and the base revision in the spec; hash the two adapter files, never rely on the path.

Related: [[gotcha-rl-checkpoint-needs-drive-sync]]
