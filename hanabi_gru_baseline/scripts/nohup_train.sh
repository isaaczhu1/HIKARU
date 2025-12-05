#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Hyperparameters (tweak here when launching)
# -----------------------------------------------------------------------------
DEVICE="${DEVICE:-cuda}"
TOTAL_UPDATES="${TOTAL_UPDATES:-10000}"
LR="${LR:-1e-4}"
CKPT="${CKPT:-}"
SAVE_DIR="${SAVE_DIR:-runs/hanabi}"
DEBUG="${DEBUG:-0}"        # 1 to pass --debug
ASYNC_ENV="${ASYNC_ENV:-0}" # 1 to pass --async-env
VARIANT="${VARIANT:-twoxtwo}" # twoxtwo | standard

# Additional config visibility (defined in config.py; change there to affect runs)
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-64}"
UNROLL_T="${UNROLL_T:-128}"
OBS_MODE="${OBS_MODE:-minimal}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

TS="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="runs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_${TS}.log"
PID_FILE="${LOG_DIR}/train_${TS}.pid"

CMD=(python -u train.py
  --device "$DEVICE"
  --total-updates "$TOTAL_UPDATES"
  --lr "$LR"
  --save-dir "$SAVE_DIR"
  --variant "$VARIANT"
)

[[ -n "$CKPT" ]] && CMD+=(--ckpt "$CKPT")
[[ "$DEBUG" -eq 1 ]] && CMD+=(--debug)
[[ "$ASYNC_ENV" -eq 1 ]] && CMD+=(--async-env)

# Log launch configuration for reproducibility
{
  echo "=== Launch $(date +"%Y-%m-%dT%H:%M:%S%z") ==="
  echo "DEVICE=$DEVICE"
  echo "TOTAL_UPDATES=$TOTAL_UPDATES"
  echo "LR=$LR"
  echo "CKPT=$CKPT"
  echo "SAVE_DIR=$SAVE_DIR"
  echo "VARIANT=$VARIANT"
  echo "DEBUG=$DEBUG"
  echo "ASYNC_ENV=$ASYNC_ENV"
  echo "SEED=$SEED"
  echo "NUM_ENVS=$NUM_ENVS"
  echo "UNROLL_T=$UNROLL_T"
  echo "OBS_MODE=$OBS_MODE"
  echo "LOG_INTERVAL=$LOG_INTERVAL"
  echo "SAVE_INTERVAL=$SAVE_INTERVAL"
  echo "Command: ${CMD[*]}"
} >>"$LOG_FILE"

nohup "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Started training (PID $(cat "$PID_FILE"))"
echo "Log: $LOG_FILE"
echo "PID file: $PID_FILE"
