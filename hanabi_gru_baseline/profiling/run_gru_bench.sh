#!/usr/bin/env bash
set -euo pipefail

# Editable parameters
DEVICE="${DEVICE:-cpu}"          # leave empty to auto-select; use "cpu" or "cuda"
GPU="${GPU:-0}"               # set to 1 to force CUDA
CPU="${CPU:-1}"               # set to 1 to force CPU
BATCH="${BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-1}"
OBS_DIM="${OBS_DIM:-658}"
NUM_MOVES="${NUM_MOVES:-20}"
HIDDEN="${HIDDEN:-256}"
ACTION_EMB="${ACTION_EMB:-32}"
SEAT_EMB="${SEAT_EMB:-8}"
INCLUDE_PREV_SELF="${INCLUDE_PREV_SELF:-0}"
N_QUERIES="${N_QUERIES:-100}"
WARMUP="${WARMUP:-10}"
OUT="${OUT:-}"                # optional path to write results

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CMD=(python -m hanabi_gru_baseline.profiling.gru_bench
  --batch "$BATCH"
  --seq-len "$SEQ_LEN"
  --obs-dim "$OBS_DIM"
  --num-moves "$NUM_MOVES"
  --hidden "$HIDDEN"
  --action-emb "$ACTION_EMB"
  --seat-emb "$SEAT_EMB"
  --n-queries "$N_QUERIES"
  --warmup "$WARMUP"
)

[[ -n "$DEVICE" ]] && CMD+=(--device "$DEVICE")
[[ "$GPU" -eq 1 ]] && CMD+=(--gpu)
[[ "$CPU" -eq 1 ]] && CMD+=(--cpu)
[[ "$INCLUDE_PREV_SELF" -eq 1 ]] && CMD+=(--include-prev-self)
[[ -n "$OUT" ]] && CMD+=(--out "$OUT")

echo "Running: ${CMD[*]}"
conda run -n hanabi "${CMD[@]}"
