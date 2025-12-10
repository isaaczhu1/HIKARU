#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Hyperparameters (tweak here when launching)
# -----------------------------------------------------------------------------
TS="$(date +"%Y%m%d_%H%M%S")"
DEVICE="${DEVICE:-cuda}"
TOTAL_UPDATES="${TOTAL_UPDATES:-1000}"
LR="${LR:-1e-4}"
LR_FINAL="${LR_FINAL:-3e-5}"
# CKPT="${CKPT:-runs/hanabi/isaacs_first_run/ckpt_010000.pt}"
# CKPT="${CKPT:-runs/hanabi/standard_train/ckpt_000200.pt}"
CKPT="${CKPT:-}"
SAVE_DIR="${SAVE_DIR:-runs/hanabi/standard_train/$TS}"
DEBUG="${DEBUG:-0}"        # 1 to pass --debug
ASYNC_ENV="${ASYNC_ENV:-0}" # 1 to pass --async-env
VARIANT="${VARIANT:-standard}" # twoxtwo | standard
SEQ_LEN="${SEQ_LEN:-1}"     # optional PPO seq_len override
START_UPDATE="${START_UPDATE:-0}" # optional override for starting update counter

# Additional config visibility (defined in config.py; change there to affect runs)
SEED="${SEED:-67}"
NUM_ENVS="${NUM_ENVS:-64}"
UNROLL_T="${UNROLL_T:-128}"
OBS_MODE="${OBS_MODE:-minimal}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

LOG_DIR="runs/hanabi/standard_train/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_${TS}.log"
PID_FILE="${LOG_DIR}/train_${TS}.pid"

CMD=(python -u hanabi_gru_baseline/train.py
  --device "$DEVICE"
  --total-updates "$TOTAL_UPDATES"
  --lr "$LR"
  --lr-final "$LR_FINAL"
  --save-dir "$SAVE_DIR"
  --variant "$VARIANT"
)

[[ -n "$CKPT" ]] && CMD+=(--ckpt "$CKPT")
[[ "$DEBUG" -eq 1 ]] && CMD+=(--debug)
[[ "$ASYNC_ENV" -eq 1 ]] && CMD+=(--async-env)
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
  echo "SEQ_LEN=$SEQ_LEN"
  echo "START_UPDATE=$START_UPDATE"
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
