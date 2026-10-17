---
name: reference-gpu-stack
description: 'gpu/STACK.md: sourced Qwen3.5/3.8 GPU stack memo (vLLM 0.31.0, torch 2.13.0, TRL first), untested'
type: reference
updated: 2026-10-17
---
gpu/STACK.md is the sourced stack memo: vLLM 0.31.0 with torch 2.13.0 and Transformers 5.17.0 (cu129 wheel, driver R575 or newer), TRL 1.16.0 first and ms-swift 4.6.0 as the GRPO fallback, a model-by-GPU memory table and a ten-command first GPU session. Every GPU claim in it is untested until the first session.

**Why:** a sourced research pass with source checks; there was no GPU to run anything.
**How to apply:** update STACK.md and mark claims verified after each GPU session; do not trust version pins that a probe has not exercised.
