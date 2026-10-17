#!/bin/sh
# First Kaggle/Colab session. GPU installation and execution are UNTESTED.
# Run: NOTHINK_SECONDS=900 THINK_SECONDS=2400 sh gpu/run_session1.sh
# Each split gets its own limit, including startup. A complete call can overrun.
set -u

if [ ! -f gpu/env_check.py ] || [ ! -f gpu/sample_vllm.py ]; then
    printf '%s\n' 'Run this command from the repository root.' >&2
    exit 1
fi
NOTHINK_SECONDS=${NOTHINK_SECONDS:-900}
THINK_SECONDS=${THINK_SECONDS:-2400}
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-python3}
K_NOTHINK=${K_NOTHINK:-8}   # cheap answers: k=8 so pass@8 (gate G0) can be read off
K_THINK=${K_THINK:-4}
scratch=.agent-runs/session1
mkdir -p "$scratch" || exit 1
exec 9> "$scratch/session.lock"
if ! flock -n 9; then
    printf '%s\n' 'Cannot lock this session. Stop another session if it is active. This runtime needs flock (util-linux).' >&2
    exit 1
fi
LOG=gpu/session1.log
: > "$scratch/steps.tsv"
rm -f "$scratch/before.json" "$scratch/after.json"

log() {
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG"
}

step() {
    step_name=$1
    shift
    log "START $step_name"
    step_start=$(date +%s)
    "$@" > "$scratch/$step_name.log" 2>&1
    step_code=$?
    step_seconds=$(($(date +%s) - step_start))
    cat "$scratch/$step_name.log" | tee -a "$LOG"
    printf '%s\t%s\t%s\n' "$step_name" "$step_code" "$step_seconds" >> "$scratch/steps.tsv"
    log "END $step_name exit=$step_code seconds=$step_seconds"
    return "$step_code"
}

finish() {
    session_code=$?
    trap - 0
    log 'START report'
    "$BOOTSTRAP_PYTHON" gpu/session1.py report "$scratch" "$session_code" >> "$LOG" 2>&1
    report_code=$?
    log "END report exit=$report_code; results/session1_report.json"
    if [ "$report_code" -ne 0 ]; then
        exit "$report_code"
    fi
    exit "$session_code"
}
trap finish 0
trap 'exit 130' INT
trap 'exit 143' TERM

log 'First session started. GPU checks run before any download.'
# env_check uses only the standard library and does not initialize CUDA.
step inventory "$BOOTSTRAP_PYTHON" gpu/env_check.py || exit 1
cp "$scratch/inventory.log" "$scratch/before.json" || exit 1
step gpu-check "$BOOTSTRAP_PYTHON" gpu/session1.py preflight "$scratch" || exit 1
# Reject invalid budgets before spending time on installation.
step budgets "$BOOTSTRAP_PYTHON" -c 'import math,sys; from pathlib import Path; values=[float(s) for s in sys.argv[1:3]]; assert all(math.isfinite(s) and s>0 for s in values), "Budgets must be positive finite seconds"; Path(sys.argv[3]).write_text("\n".join(str(math.ceil(s)+600) for s in values))' "$NOTHINK_SECONDS" "$THINK_SECONDS" "$scratch/hard-budgets" || exit 1
NOTHINK_HARD_SECONDS=$(sed -n '1p' "$scratch/hard-budgets")
THINK_HARD_SECONDS=$(sed -n '2p' "$scratch/hard-budgets")
step uv-install "$BOOTSTRAP_PYTHON" -m pip install --quiet uv==0.13.1 || exit 1
UV="$BOOTSTRAP_PYTHON -m uv"
step venv $UV venv gpu/.venv --python 3.12 --allow-existing || exit 1
PY=gpu/.venv/bin/python
step stack-install $UV pip install --python "$PY" \
    'https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
    'torch==2.13.0+cu129' 'transformers==5.17.0' \
    --extra-index-url https://download.pytorch.org/whl/cu129 || exit 1
step pip-check $UV pip check --python "$PY" || exit 1
step installed-inventory "$PY" gpu/env_check.py || exit 1
cp "$scratch/installed-inventory.log" "$scratch/after.json" || exit 1
step prepare "$PY" gpu/session1.py prepare "$scratch" || exit 1

step smoke-short timeout -k 30 600 "$PY" gpu/sample_vllm.py --model Qwen/Qwen3.5-4B \
    --data "$scratch/smoke-data" --splits val --mode nothink --k 8 \
    --max-tokens 128 --limit-drugs 1 --out "$scratch/smoke-short" || log 'Short smoke failed; continue to collect diagnostics.'
step smoke-resume timeout -k 30 600 "$PY" gpu/sample_vllm.py --model Qwen/Qwen3.5-4B \
    --data "$scratch/smoke-data" --splits val --mode nothink --k 8 \
    --max-tokens 128 --limit-drugs 1 --out "$scratch/smoke-short" || log 'Resume smoke failed; continue.'
step smoke-long timeout -k 30 600 "$PY" gpu/sample_vllm.py --model Qwen/Qwen3.5-4B \
    --data "$scratch/smoke-data" --splits val --mode think --k 8 \
    --max-tokens 8192 --limit-drugs 1 --out "$scratch/smoke-long" || log 'Long smoke failed or timed out; this does not pass the 8k check.'
step memory-after-smoke nvidia-smi --query-gpu=uuid,memory.used,memory.free --format=csv,noheader,nounits || log 'Could not measure GPU memory after smoke.'

for mode in nothink think; do
    if [ "$mode" = nothink ]; then
        tokens=512
        k=$K_NOTHINK
        seconds=$NOTHINK_SECONDS
        hard_seconds=$NOTHINK_HARD_SECONDS
    else
        tokens=8192
        k=$K_THINK
        seconds=$THINK_SECONDS
        hard_seconds=$THINK_HARD_SECONDS
    fi
    for split in val train; do
        step "$mode-$split" timeout -k 30 "$hard_seconds" "$PY" gpu/sample_vllm.py --model Qwen/Qwen3.5-4B \
            --data "$scratch/probe-data" --splits "$split" --mode "$mode" --k "$k" \
            --max-tokens "$tokens" --temperature 1.0 --prompts-per-batch auto \
            --prompt-subset-seed 0 --stop-after-seconds "$seconds" --out "$scratch/$mode-$split" || log "$mode $split probe failed; continue. Completed prompts remain on disk."
        step "memory-after-$mode-$split" nvidia-smi --query-gpu=uuid,memory.used,memory.free --format=csv,noheader,nounits || log 'Could not measure GPU memory after probe.'
    done
done
log 'Sampling steps ended. The exit trap will write the report.'
