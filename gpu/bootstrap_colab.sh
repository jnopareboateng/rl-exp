#!/bin/sh
# Fresh Colab GPU runtime -> ready to sample (vLLM venv) and to train (SFT/RL venv). Run from the repository root after unpacking the bundle:
#   tar xzf bootstrap.tgz -C /content/rl-exp && cd /content/rl-exp && sh gpu/bootstrap_colab.sh
# Idempotent. Pins follow gpu/STACK.md and gpu/run_session1.sh / run_sft.sh / run_rl.sh. Exits non-zero if the GPU or an import is missing, if CUDA
# is not usable in BOTH venvs (torch.cuda.is_available() and at least one device), if the training venv's rdkit is not the pinned one, or if a Test
# split file is in the tree: this is the pre-lock bootstrap, the Test files are uploaded only afterwards, for the locked evaluation.
set -eu
RDKIT_VERSION=2026.9.1      # the scorer's chemistry library: the version the Val numbers and the tests ran on (RL rewards and the CPU scoring use it); the one place the pin lives
if [ ! -f gpu/sample_vllm.py ]; then printf '%s\n' 'Run from the repository root.' >&2; exit 1; fi
leak=$(find data -name '*test_*' 2>/dev/null | head -n 1)
if [ -n "$leak" ]; then printf '%s\n' "A Test file is in the tree ($leak): bootstrap before the Test files are uploaded, from a bundle without them." >&2; exit 1; fi
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
uv pip install --python "$PY" "rdkit==$RDKIT_VERSION" numpy pytest
# checks: a driver with no visible device passes nvidia-smi but not these
gpu/.venv/bin/python -c 'import sys, torch, vllm; n = torch.cuda.device_count(); ok = torch.cuda.is_available() and n >= 1; print("vllm venv: torch", torch.__version__, "cuda", ok, "devices", n, "vllm", vllm.__version__); sys.exit(0 if ok else 1)' \
    || { printf '%s\n' 'CUDA is not usable in the vLLM venv (no device, or a driver/torch mismatch).' >&2; exit 1; }
"$PY" -c 'import sys, torch, transformers, peft; n = torch.cuda.device_count(); ok = torch.cuda.is_available() and n >= 1; print("train venv: torch", torch.__version__, "cuda", ok, "devices", n, "transformers", transformers.__version__, "peft", peft.__version__); sys.exit(0 if ok else 1)' \
    || { printf '%s\n' 'CUDA is not usable in the training venv (no device, or a driver/torch mismatch).' >&2; exit 1; }
"$PY" -c 'import sys, rdkit; from importlib.metadata import version; v = version("rdkit"); print("rdkit", v); sys.exit(0 if v == sys.argv[1] else 1)' "$RDKIT_VERSION" \
    || { printf '%s\n' "rdkit in the training venv is not the pinned $RDKIT_VERSION." >&2; exit 1; }
printf '%s\n' 'Bootstrap complete. Adapters live on Drive (MyDrive/rl-exp/adapters); mount Drive from the Colab sidebar if it is missing.'
