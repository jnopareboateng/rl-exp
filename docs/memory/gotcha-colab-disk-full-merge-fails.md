---
name: gotcha-colab-disk-full-merge-fails
description: 'merge_lora ''SafetensorError: Data values are omitted'' means the Colab VM disk is full (27B cache)'
type: gotcha
updated: 2026-10-18
---
`gpu/merge_lora.py` failed with `SafetensorError: Data values are omitted` inside `save_pretrained` after the Qwen3.5-27B-FP8 run: the VM disk was 96% full because the Hugging Face cache still held the 27B weights (38 GB) beside the 4B and 9B ones, and a merged 4B export needs about 9 GB. The Drive is also full (30 GB; deleted files stay in the trash), so nothing new syncs to it.

**Why:** a write error from a full disk surfaces as a safetensors error with no mention of space; the merge was the first large write after the 27B job.
**How to apply:** before a merge, check `df -h /content` and delete `~/.cache/huggingface` (it is re-downloaded); a queue script that runs the 27B should clear the cache when it ends, as `g1.py` did before each job. Empty the Drive trash to get quota back.
