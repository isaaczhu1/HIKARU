#!/usr/bin/env bash
set -euo pipefail

# Editable parameters
CKPT="${CKPT:-/workspace/HIKARU/hanabi_gru_baseline/runs/hanabi/standard_train/20251208_012329/ckpt_020000.pt}"               # required: path to checkpoint (.pt)
EPISODES="${EPISODES:-20}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"           # empty to auto; or "cpu"/"cuda"
GREEDY="${GREEDY:-0}"          # 1 to use argmax actions

if [[ -z "$CKPT" ]]; then
  echo "CKPT must be set to a checkpoint path" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CMD=(python -m hanabi_gru_baseline.eval_policy
  --ckpt "$CKPT"
  --episodes "$EPISODES"
  --seed "$SEED"
)

[[ -n "$DEVICE" ]] && CMD+=(--device "$DEVICE")
[[ "$GREEDY" -eq 1 ]] && CMD+=(--greedy)

echo "Running: ${CMD[*]}"
conda run -n hanabi "${CMD[@]}"
