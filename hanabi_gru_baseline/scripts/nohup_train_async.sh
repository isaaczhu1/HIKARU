#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Hyperparameters (tweak here when launching)
# -----------------------------------------------------------------------------
DEVICE="${DEVICE:-cuda}"
TOTAL_UPDATES="${TOTAL_UPDATES:-50000}"
LR="${LR:-3e-5}"
LR_FINAL="${LR_FINAL:-3e-5}"
CKPT="${CKPT:-}"
SAVE_DIR="${SAVE_DIR:-runs/hanabi_async}"
DEBUG="${DEBUG:-0}"          # 1 to pass --debug
ASYNC_ENV=1                  # always async in this script
VARIANT="${VARIANT:-standard}" # twoxtwo | standard
NUM_ENVS="${NUM_ENVS:-24}"  # number of parallel envs/workers (the parameter you asked about)
SEQ_LEN="${SEQ_LEN:-}"      # optional PPO seq_len override
START_UPDATE="${START_UPDATE:-}" # optional override for starting update counter

# Additional config visibility (defined in config.py; change there to affect runs)
SEED="${SEED:-67}"
UNROLL_T="${UNROLL_T:-128}"
OBS_MODE="${OBS_MODE:-minimal}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"

TS="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="runs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_async_${TS}.log"
PID_FILE="${LOG_DIR}/train_async_${TS}.pid"

CMD=(python -u train.py
  --device "$DEVICE"
  --total-updates "$TOTAL_UPDATES"
  --lr "$LR"
  --lr-final "$LR_FINAL"
  --save-dir "$SAVE_DIR"
  --variant "$VARIANT"
  --async-env
)

[[ -n "$CKPT" ]] && CMD+=(--ckpt "$CKPT")
[[ "$DEBUG" -eq 1 ]] && CMD+=(--debug)
[[ -n "$SEQ_LEN" ]] && CMD+=(--seq-len "$SEQ_LEN")
[[ -n "$NUM_ENVS" ]] && CMD+=(--num-envs "$NUM_ENVS")
[[ -n "$UNROLL_T" ]] && CMD+=(--unroll-T "$UNROLL_T")
[[ -n "$OBS_MODE" ]] && CMD+=(--obs-mode "$OBS_MODE")
[[ -n "$SAVE_INTERVAL" ]] && CMD+=(--save-interval "$SAVE_INTERVAL")
[[ -n "$LOG_INTERVAL" ]] && CMD+=(--log-interval "$LOG_INTERVAL")
[[ -n "$SEED" ]] && CMD+=(--seed "$SEED")
[[ -n "$START_UPDATE" ]] && CMD+=(--start-update "$START_UPDATE")

# Log launch configuration for reproducibility
{
  echo "=== Launch $(date +"%Y-%m-%dT%H:%M:%S%z") ==="
  echo "DEVICE=$DEVICE"
  echo "TOTAL_UPDATES=$TOTAL_UPDATES"
  echo "LR=$LR"
  echo "LR_FINAL=$LR_FINAL"
  echo "CKPT=$CKPT"
  echo "SAVE_DIR=$SAVE_DIR"
  echo "VARIANT=$VARIANT"
  echo "DEBUG=$DEBUG"
  echo "ASYNC_ENV=$ASYNC_ENV"
  echo "NUM_ENVS=$NUM_ENVS"
  echo "SEED=$SEED"
  echo "UNROLL_T=$UNROLL_T"
  echo "OBS_MODE=$OBS_MODE"
  echo "LOG_INTERVAL=$LOG_INTERVAL"
  echo "SAVE_INTERVAL=$SAVE_INTERVAL"
  echo "Command: ${CMD[*]}"
} >>"$LOG_FILE"

nohup "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Started ASYNC training (PID $(cat "$PID_FILE"))"
echo "Log: $LOG_FILE"
echo "PID file: $PID_FILE"
