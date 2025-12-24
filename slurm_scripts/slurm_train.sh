#!/usr/bin/env bash
#SBATCH --job-name=hanabi_gru
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH -p mit_normal_gpu
#SBATCH -G 1

#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err


export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"


set -euo pipefail

mkdir -p slurm_outputs

# ---- Activate conda env ----
# Adjust if your conda init path differs.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hanabi

# -----------------------------------------------------------------------------
# Hyperparameters (same overrides as your nohup script)
# -----------------------------------------------------------------------------
TS="${TS:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
DEVICE="${DEVICE:-cuda}"
TOTAL_UPDATES="${TOTAL_UPDATES:-50000}"
LR="${LR:-1e-4}"
LR_FINAL="${LR_FINAL:-3e-5}"
CKPT="${CKPT:-}"
SAVE_DIR="${SAVE_DIR:-runs/hanabi/standard_train/$TS}"
DEBUG="${DEBUG:-0}"
ASYNC_ENV="${ASYNC_ENV:-0}"
VARIANT="${VARIANT:-standard}"
SEQ_LEN="${SEQ_LEN:-1}"
START_UPDATE="${START_UPDATE:-0}"

SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-64}"
UNROLL_T="${UNROLL_T:-128}"
OBS_MODE="${OBS_MODE:-rich_card_knowledge}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

export PYTHONUNBUFFERED=1

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

echo "=== SLURM JOB START ==="
echo "JobID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "CWD: $(pwd)"
echo "Command: ${CMD[*]}"
echo "======================="




echo "=== GPU CHECK ==="
nvidia-smi -L || true
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
PY
echo "================="




# Use srun so Slurm properly tracks the process
srun "${CMD[@]}"
