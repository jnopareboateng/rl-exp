#!/bin/sh
# Colab A100 SFT. GPU install and execution are UNTESTED here. See gpu/SFT.md.
# ARM=A2 STOP_AFTER_SECONDS=3600 sh gpu/run_sft.sh
set -u

if [ ! -f gpu/sft_lora.py ] || [ ! -f gpu/merge_lora.py ]; then
    printf '%s\n' 'Run from the repository root.' >&2
    exit 1
fi
ARM=${ARM:-A2}
case "$ARM" in
    A1) prefix=a1_trace; default_len=8192; default_tokens=16384 ;;
    A2) prefix=a2_answer; default_len=2048; default_tokens=8192 ;;
    *) printf '%s\n' 'ARM must be A1 or A2.' >&2; exit 1 ;;
esac
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-python3}
UV=${UV:-uv}
TRAIN_VENV=${TRAIN_VENV:-.agent-runs/sft-venv}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
TRAIN_FILE=${TRAIN_FILE:-data/processed/arms/${prefix}_train.jsonl}
VAL_FILE=${VAL_FILE:-data/processed/arms/${prefix}_val.jsonl}
OUT=${OUT:-.agent-runs/sft_${ARM}}
MERGED=${MERGED:-.agent-runs/sft_${ARM}_merged}
MAX_LEN=${MAX_LEN:-$default_len}
MAX_TOKENS_PER_BATCH=${MAX_TOKENS_PER_BATCH:-$default_tokens}
BATCH_SIZE=${BATCH_SIZE:-16}
OVERLENGTH=${OVERLENGTH:-drop}
RANK=${RANK:-32}
LR=${LR:-0.0002}
EPOCHS=${EPOCHS:-3}
SEED=${SEED:-0}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
SAVE_STEPS=${SAVE_STEPS:-10}
LOG_STEPS=${LOG_STEPS:-5}
STOP_AFTER_SECONDS=${STOP_AFTER_SECONDS:-3600}
ALLOW_SLOW_KERNELS=${ALLOW_SLOW_KERNELS:-0}
DRY_RUN=${DRY_RUN:-0}
scratch=.agent-runs/sft-runner
mkdir -p "$scratch" || exit 1
exec 9> "$scratch/session.lock"
if ! flock -n 9; then
    printf '%s\n' 'Another SFT runner is active. This runtime needs flock.' >&2
    exit 1
fi
LOG="$scratch/sft.log"
: > "$scratch/steps.tsv"
rm -f "$scratch/versions.json"

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
    "$BOOTSTRAP_PYTHON" gpu/sft_report.py report "$scratch" "$ARM" "$OUT" "$MERGED" "results/sft_${ARM}_report.json" "$session_code" >> "$LOG" 2>&1
    report_code=$?
    log "END report exit=$report_code"
    if [ "$report_code" -ne 0 ]; then exit "$report_code"; fi
    exit "$session_code"
}
# Set up the command once. Dry runs use the same validated options, without installs.
set -- gpu/sft_lora.py --arm "$ARM" --model "$MODEL" --train "$TRAIN_FILE" --val "$VAL_FILE" --out "$OUT" \
    --max-len "$MAX_LEN" --max-tokens-per-batch "$MAX_TOKENS_PER_BATCH" --batch-size "$BATCH_SIZE" \
    --overlength "$OVERLENGTH" --rank "$RANK" --lr "$LR" --epochs "$EPOCHS" --seed "$SEED" \
    --warmup-ratio "$WARMUP_RATIO" --save-steps "$SAVE_STEPS" --log-steps "$LOG_STEPS" --stop-after-seconds "$STOP_AFTER_SECONDS"
if [ "$ALLOW_SLOW_KERNELS" = 1 ]; then set -- "$@" --allow-slow-kernels; fi
if [ "$DRY_RUN" = 1 ]; then
    step dry-run "$BOOTSTRAP_PYTHON" "$@" --dry-run
    exit $?
fi
trap finish 0
trap 'exit 130' INT
trap 'exit 143' TERM
step plan "$BOOTSTRAP_PYTHON" "$@" --dry-run || exit 1
step gpu-check "$BOOTSTRAP_PYTHON" gpu/sft_report.py preflight "$STOP_AFTER_SECONDS" || exit 1
HARD_SECONDS=$("$BOOTSTRAP_PYTHON" -c 'import math,sys; print(math.ceil(float(sys.argv[1]))+600)' "$STOP_AFTER_SECONDS") || exit 1
if ! command -v "$UV" >/dev/null 2>&1; then
    log 'uv is required. Install uv with the official installer, then run again.'
    exit 1
fi
step venv "$UV" venv "$TRAIN_VENV" --python 3.12 --allow-existing || exit 1
PY="$TRAIN_VENV/bin/python"
step stack-install "$UV" pip install --python "$PY" 'torch==2.10.0+cu128' 'transformers==5.17.0' 'peft==0.21.2' \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match || exit 1
step fla-install "$UV" pip install --python "$PY" --only-binary :all: 'flash-linear-attention==0.5.2' 'fla-core==0.5.2' || log 'FLA install failed. On compute capability 8.0+ (A100) the train step stops at its kernel check unless ALLOW_SLOW_KERNELS=1 (--allow-slow-kernels) is set.'
# This wheel matches Python 3.12, torch 2.10, CUDA 12, x86_64 and the C++11 ABI.
step conv-install "$UV" pip install --python "$PY" --only-binary :all: \
    'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=8e81f8c76435ad31aa41edc6c0c9d26de971a293b190183d2816da71547614d7' \
    || log 'Convolution wheel install failed. On compute capability 8.0+ (A100) the train step stops at its kernel check unless ALLOW_SLOW_KERNELS=1 (--allow-slow-kernels) is set.'
step pip-check "$UV" pip check --python "$PY" || exit 1
step versions "$PY" gpu/sft_report.py versions || exit 1
cp "$scratch/versions.log" "$scratch/versions.json" || exit 1
step train timeout --foreground -k 30 "$HARD_SECONDS" "$PY" "$@" || exit 1
step best "$BOOTSTRAP_PYTHON" gpu/sft_report.py best "$OUT"
best_code=$?
if [ "$best_code" = 3 ]; then
    log 'No full validation epoch yet. The checkpoint can resume. Merge is deferred.'
    exit 0
fi
if [ "$best_code" -ne 0 ]; then exit "$best_code"; fi
BEST_PATH=$(cat "$scratch/best.log")
MERGED="$MERGED/$(basename "$BEST_PATH")"
merge_step=merge
if [ -d "$MERGED" ]; then merge_step=merge-reused; fi
step "$merge_step" timeout --foreground -k 30 900 "$PY" gpu/merge_lora.py --base "$MODEL" --adapter "$OUT" --out "$MERGED" || exit 1
log 'SFT steps ended. The exit trap will write the report.'
