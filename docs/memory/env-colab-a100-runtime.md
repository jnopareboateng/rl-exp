---
name: env-colab-a100-runtime
description: 'Colab A100-40GB on Google account index 1: install 57 s, 2k tok/s batched; kill -9 stuck engines'
type: env
updated: 2026-10-17
---
The Colab account with A100 access is Google account index 1 (authuser=1). Runtime: A100-SXM4-40GB, driver 580.82, 12 vCPU, 83 GB RAM, 71 GB free disk, system python3.12 present. The pinned stack (vLLM 0.31.0, torch 2.13.0+cu129, transformers 5.17.0) installs in 57 s with uv --index-strategy unsafe-best-match. Eager-mode sampling: 113 tokens/s for one prompt (8 sequences), about 2,000 tokens/s with 128 sequences in flight, 885 tokens/s with 48 long sequences; startup is about 1 minute once caches are warm. A killed sampler can leave VLLM::EngineCore alive holding the GPU: kill -9 its pids before the next run.

**Why:** measured in the first session; the MCP cannot change the runtime type, so the user selects the GPU in the Colab UI, and the browser launcher must open Microsoft Edge (see wsl-xdg-open-shim).
**How to apply:** open the Colab connection with notebook_url ending ?authuser=1; check nvidia-smi after connecting; batch many prompts per call (--prompts-per-batch auto).

Related: [[env-compute-supply]]
