#!/bin/sh
# Colab A100 answer-only RL (arm A5). GPU install and execution are UNTESTED here. See gpu/RL.md.
# INIT_ADAPTER=.agent-runs/sft_A2/adapters/epoch-1 STOP_AFTER_SECONDS=3600 sh gpu/run_rl.sh
# GATE=parity sh gpu/run_rl.sh      GATE=smoke sh gpu/run_rl.sh      DRY_RUN=1 sh gpu/run_rl.sh
# REWARD_MODE=validity (validity-only RL control)   PROFILE_MODE=none (profile-free RL control)
# INIT_ADAPTER=none REVISION=<base commit of the A2 adapter's revision.json> (zero-RL ablation: no adapter records the base, so REVISION is required)
set -u

if [ ! -f gpu/rl_grpo.py ] || [ ! -f src/rlexp/grpo.py ]; then
    printf '%s\n' 'Run from the repository root.' >&2
    exit 1
fi
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-python3}
UV=${UV:-uv}
TRAIN_VENV=${TRAIN_VENV:-.agent-runs/sft-venv}
PLAN_PYTHON=${PLAN_PYTHON:-$TRAIN_VENV/bin/python}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
INIT_ADAPTER=${INIT_ADAPTER:-.agent-runs/sft_A2/adapters/epoch-1}
TRAIN_FILE=${TRAIN_FILE:-data/processed/train.jsonl}
VAL_FILE=${VAL_FILE:-data/processed/val.jsonl}
OUT=${OUT:-.agent-runs/rl_A5}
STEPS=${STEPS:-100}
PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-32}
GROUP_SIZE=${GROUP_SIZE:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-1.0}
LR=${LR:-1e-5}
EVAL_EVERY=${EVAL_EVERY:-10}
EVAL_K=${EVAL_K:-8}
SAVE_EVERY=${SAVE_EVERY:-10}
PATIENCE=${PATIENCE:-3}
GEN_BATCH=${GEN_BATCH:-128}
GEN_TOKEN_BUDGET=${GEN_TOKEN_BUDGET:-49152}
MAX_TOKENS_PER_BATCH=${MAX_TOKENS_PER_BATCH:-8192}
MIN_GROUPS=${MIN_GROUPS:-16}
REVISION=${REVISION:-}
REWARD_MODE=${REWARD_MODE:-full}
PROFILE_MODE=${PROFILE_MODE:-real}
SEED=${SEED:-0}
STOP_AFTER_SECONDS=${STOP_AFTER_SECONDS:-3600}
ALLOW_SLOW_KERNELS=${ALLOW_SLOW_KERNELS:-0}
GATE=${GATE:-}
DRY_RUN=${DRY_RUN:-0}
case "$GATE" in
    ''|parity|smoke) ;;
    *) printf '%s\n' 'GATE must be empty, parity or smoke.' >&2; exit 1 ;;
esac
if [ "$INIT_ADAPTER" = none ] && [ -z "$REVISION" ]; then
    printf '%s\n' 'INIT_ADAPTER=none needs REVISION: the base commit of the A2 adapter (its revision.json). Without it the Hub'"'"'s current revision would load.' >&2
    exit 1
fi
scratch=.agent-runs/rl-runner
mkdir -p "$scratch" || exit 1
exec 9> "$scratch/session.lock"
if ! flock -n 9; then
    printf '%s\n' 'Another RL runner is active. This runtime needs flock.' >&2
    exit 1
fi
LOG="$scratch/rl.log"
: > "$scratch/steps.tsv"
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}

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
    {
        printf 'exit_code=%s\n' "$session_code"
        cat "$scratch/steps.tsv"
        if [ -f "$OUT/checkpoint.json" ]; then
            "$BOOTSTRAP_PYTHON" -c 'import json, sys; root = sys.argv[1]; latest = json.load(open(root + "/checkpoint.json"))["latest"]; state = json.load(open(root + "/" + latest + "/state.json")); print(json.dumps({key: state.get(key) for key in ("status", "step", "cursor", "skipped_updates", "training_seconds", "peak_memory_bytes", "first_update_delta_norm")})); print(json.dumps({"evals [step, val = fv_tanimoto, val_reward]": [[e["step"], round(e["val"], 4), None if "val_reward" not in e else round(e["val_reward"], 4)] for e in state.get("evals", [])]}))' "$OUT"
        fi
    } 2>&1 | tee -a "$LOG"
    exit "$session_code"
}
# Set up the command once. The plan step uses the same validated options.
set -- gpu/rl_grpo.py --model "$MODEL" --init-adapter "$INIT_ADAPTER" --train "$TRAIN_FILE" --val "$VAL_FILE" --out "$OUT" \
    --steps "$STEPS" --prompts-per-step "$PROMPTS_PER_STEP" --group-size "$GROUP_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" --lr "$LR" --eval-every "$EVAL_EVERY" --eval-k "$EVAL_K" --save-every "$SAVE_EVERY" \
    --patience "$PATIENCE" --gen-batch "$GEN_BATCH" --gen-token-budget "$GEN_TOKEN_BUDGET" --max-tokens-per-batch "$MAX_TOKENS_PER_BATCH" --min-groups "$MIN_GROUPS" \
    --reward-mode "$REWARD_MODE" --profile-mode "$PROFILE_MODE" --stop-after-seconds "$STOP_AFTER_SECONDS" --seed "$SEED"
if [ -n "$REVISION" ]; then set -- "$@" --revision "$REVISION"; fi
if [ "$ALLOW_SLOW_KERNELS" = 1 ]; then set -- "$@" --allow-slow-kernels; fi
if [ -n "$GATE" ]; then set -- "$@" --gate "$GATE"; fi
if [ "$DRY_RUN" = 1 ]; then
    # The plan imports numpy and rdkit: use the training venv once it exists, else PLAN_PYTHON or the bootstrap python.
    if [ ! -x "$PLAN_PYTHON" ]; then PLAN_PYTHON=$BOOTSTRAP_PYTHON; fi
    log "COMMAND $*"
    step dry-run "$PLAN_PYTHON" "$@" --dry-run
    exit $?
fi
if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    printf '%s\n' 'The processed Train and Val files are missing. Copy data/processed/train.jsonl and val.jsonl first.' >&2
    exit 1
fi
if [ "$INIT_ADAPTER" != none ] && [ ! -f "$INIT_ADAPTER/adapter_model.safetensors" ]; then
    printf '%s\n' 'INIT_ADAPTER must be the A2 adapter directory (adapter_model.safetensors, adapter_config.json) or none.' >&2
    exit 1
fi
trap finish 0
trap 'exit 130' INT
trap 'exit 143' TERM
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log 'nvidia-smi was not found: connect a GPU runtime (an A100) first.'
    exit 1
fi
step gpu-check "$BOOTSTRAP_PYTHON" -c 'import subprocess, sys; q = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap,driver_version,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True); rows = [l.split(",") for l in q.stdout.splitlines() if l.strip()]; bad = [r for r in rows if float(r[0]) < 8.0 or int(r[1].strip().split(".")[0]) < 570]; print(len(rows), "GPU(s), memory MiB", [r[2].strip() for r in rows]); sys.exit("RL needs compute capability 8.0+ (bf16) and driver R570+" if bad or not rows else 0)' || exit 1
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
# The reward needs rdkit and numpy in the training venv (the SFT venv has neither). rdkit is pinned: a new release can change standardisation
# or fingerprints and with them the reward (2026.9.1 scored the baselines); rdkit and numpy versions are part of the resume identity.
step rdkit-install "$UV" pip install --python "$PY" 'rdkit==2026.9.1' numpy || exit 1
step pip-check "$UV" pip check --python "$PY" || exit 1
log "COMMAND $*"
step plan "$PY" "$@" --dry-run || exit 1
step train timeout --foreground -k 30 "$HARD_SECONDS" "$PY" "$@" || exit 1
log 'RL steps ended. The exit trap will write the report.'
