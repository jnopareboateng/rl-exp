---
name: env-colab-a100-runtime
description: 'Colab A100-40GB (account 1): stack, throughput, disk traps, 27B FP8 needs memory fraction 0.93'
type: env
updated: 2026-10-17
---
The Colab account with A100 access is Google account index 1 (authuser=1). Runtime: A100-SXM4-40GB, driver 580.82, 12 vCPU, 83 GB RAM, 113 GB disk (about 40 GB used by the image), system python3.12. Pinned inference stack (vLLM 0.31.0, torch 2.13.0+cu129, transformers 5.17.0) installs in 57 s with uv --index-strategy unsafe-best-match; the SFT/RL venv (torch 2.10.0+cu128, peft 0.21.2, fla, causal-conv1d, rdkit) is .agent-runs/sft-venv. vLLM eager: 2,000 tokens/s with 128 short sequences, 885 tokens/s long think; a Qwen3.5-27B-FP8 Val run (464 samples) takes 24 min; 27B FP8 needs --gpu-memory-utilization 0.93 on 40 GB. HF generate (RL path) is latency bound: about 25 s for 256 decode steps whatever the batch, and 192 sequences of 1.8k-token prompts run out of memory (token-budgeted sub-batches fix it).
Disk traps: model weights live in ~/.cache/huggingface/hub/blobs, shared by repos, so deleting models--* directories frees nothing (rm -rf the whole hub dir); 4B 9 GB, 9B 19 GB, 27B FP8 31 GB; a full disk stalls running jobs. Drive is mounted (30 GB; MyDrive/rl-exp holds adapters and logs). A killed sampler can leave VLLM::EngineCore holding the GPU: kill -9 its pids (not pkill -f).

**Why:** measured over the first sessions; the MCP cannot change the runtime type, so the user selects the GPU in the Colab UI, and the browser launcher must open Microsoft Edge (see wsl-xdg-open-shim).
**How to apply:** there is no local GPU: every GPU step (training, sampling, kernel gates) runs here; open the Colab connection with notebook_url ending ?authuser=1; check nvidia-smi and df -h /content before big runs; delete caches after each model; run long jobs with nohup and poll in calls of at most 60 s; keep adapters on Drive.

Related: [[env-compute-supply]], [[gotcha-colab-mcp-hang]]
