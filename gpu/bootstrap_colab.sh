#!/bin/sh
# Fresh Colab GPU runtime -> ready to sample (vLLM venv) and to train (SFT/RL venv). Run from the repository root after unpacking the bundle:
#   tar xzf bootstrap.tgz -C /content/rl-exp && cd /content/rl-exp && sh gpu/bootstrap_colab.sh
# Idempotent. Pins follow gpu/STACK.md and gpu/run_session1.sh / run_sft.sh / run_rl.sh. Exits non-zero if the GPU or an import is missing.
set -eu
if [ ! -f gpu/sample_vllm.py ]; then printf '%s\n' 'Run from the repository root.' >&2; exit 1; fi
command -v nvidia-smi >/dev/null 2>&1 || { printf '%s\n' 'No GPU: set the runtime type to an A100 first.' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if ! command -v uv >/dev/null 2>&1; then python3 -m pip install --quiet uv==0.13.1; fi
mkdir -p .agent-runs runs
# inference venv: vLLM 0.31.0 + torch 2.13.0 (CUDA 12.9)
uv venv gpu/.venv --python 3.12 --allow-existing
uv pip install --python gpu/.venv/bin/python \
    'https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
    'torch==2.13.0+cu129' 'transformers==5.17.0' --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
uv pip uninstall --python gpu/.venv/bin/python torchcodec >/dev/null 2>&1 || true      # needs FFmpeg libraries; text-only runs do not use it
# training venv: torch 2.10 + peft + fast kernels + the scorer's dependencies
uv venv .agent-runs/sft-venv --python 3.12 --allow-existing
PY=.agent-runs/sft-venv/bin/python
uv pip install --python "$PY" 'torch==2.10.0+cu128' 'transformers==5.17.0' 'peft==0.21.2' \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install --python "$PY" --only-binary :all: 'flash-linear-attention==0.5.2' 'fla-core==0.5.2' || printf '%s\n' 'FLA install failed: training will stop at its kernel check unless --allow-slow-kernels is set.'
uv pip install --python "$PY" --only-binary :all: \
    'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' || printf '%s\n' 'causal-conv1d wheel failed.'
uv pip install --python "$PY" rdkit numpy pytest
# checks
gpu/.venv/bin/python -c 'import torch, vllm; print("vllm venv: torch", torch.__version__, "cuda", torch.cuda.is_available(), "vllm", vllm.__version__)'
"$PY" -c 'import torch, transformers, peft, rdkit; print("train venv: torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__, "peft", peft.__version__, "rdkit", rdkit.__version__)'
printf '%s\n' 'Bootstrap complete. Adapters live on Drive (MyDrive/rl-exp/adapters); mount Drive from the Colab sidebar if it is missing.'
