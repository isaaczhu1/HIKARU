#!/usr/bin/env bash
#SBATCH --job-name=hanabi_gru
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH -p mit_normal_gpu
#SBATCH -G 1
#SBATCH --signal=B:USR1@300          # signal batch shell 5 min before walltime
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

set -euo pipefail

# --------- you can override these at submission: VAR=... sbatch ... ----------
TOTAL_UPDATES="${TOTAL_UPDATES:-50000}"
LR="${LR:-3e-4}"
LR_FINAL="${LR_FINAL:-9e-5}"
VARIANT="${VARIANT:-standard}"
OBS_MODE="${OBS_MODE:-rich_card_knowledge}"
NUM_ENVS="${NUM_ENVS:-64}"
UNROLL_T="${UNROLL_T:-128}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"   # keep small to minimize lost work on segment stop
LOG_INTERVAL="${LOG_INTERVAL:-10}"
DEVICE="${DEVICE:-cuda}"

# Seed policy: "seed = SEED_OFFSET + start_update"
SEED_OFFSET="${SEED_OFFSET:-0}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

mkdir -p slurm_outputs

# ---- Activate conda env ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hanabi

# ---- Stable run id across segments ----
RUN_ID="${RUN_ID:-$SLURM_JOB_ID}"  # first job sets it; later jobs inherit it
SAVE_DIR="runs/hanabi/standard_train/${RUN_ID}"
mkdir -p "${SAVE_DIR}"

latest_ckpt() {
  # ckpt_000123.pt are zero-padded so lexical sort is fine
  ls -1 "${SAVE_DIR}"/ckpt_*.pt 2>/dev/null | sort | tail -n 1 || true
}

ckpt_update() {
  local p="$1"
  python - <<PY
import torch
st = torch.load("${p}", map_location="cpu")
print(int(st.get("update", 0)))
PY
}

CKPT="$(latest_ckpt)"
START_UPDATE=0
if [[ -n "${CKPT}" ]]; then
  START_UPDATE="$(ckpt_update "${CKPT}")"
fi

SEED=$(( SEED_OFFSET + START_UPDATE ))

echo "=== SEGMENT SETUP ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "RUN_ID=${RUN_ID}"
echo "SAVE_DIR=${SAVE_DIR}"
echo "CKPT=${CKPT:-none}"
echo "START_UPDATE=${START_UPDATE}"
echo "TOTAL_UPDATES=${TOTAL_UPDATES}"
echo "SEED=${SEED} (SEED_OFFSET=${SEED_OFFSET})"
echo "====================="

echo "=== GPU CHECK ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
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

CMD=(python -u hanabi_gru_baseline/train.py
  --device "${DEVICE}"
  --total-updates "${TOTAL_UPDATES}"
  --lr "${LR}"
  --lr-final "${LR_FINAL}"
  --save-dir "${SAVE_DIR}"
  --variant "${VARIANT}"
  --obs-mode "${OBS_MODE}"
  --num-envs "${NUM_ENVS}"
  --unroll-T "${UNROLL_T}"
  --save-interval "${SAVE_INTERVAL}"
  --log-interval "${LOG_INTERVAL}"
  --seed "${SEED}"
)
if [[ -n "${CKPT}" ]]; then
  CMD+=(--ckpt "${CKPT}")
fi

echo "Command: ${CMD[*]}"

GRACEFUL_STOP=0
SRUN_PID=""

graceful_stop() {
  # Called on USR1 (~5 min before walltime) or TERM/INT
  if (( GRACEFUL_STOP == 1 )); then return; fi
  GRACEFUL_STOP=1
  echo "[signal] received; stopping training step so job exits cleanly and chains."
  if [[ -n "${SRUN_PID}" ]] && kill -0 "${SRUN_PID}" 2>/dev/null; then
    kill -TERM "${SRUN_PID}" 2>/dev/null || true
  fi
}

trap graceful_stop USR1 TERM INT

# Run training under srun so Slurm tracks it
srun --kill-on-bad-exit=1 "${CMD[@]}" &
SRUN_PID=$!

set +e
wait "${SRUN_PID}"
RC=$?
set -e

if (( RC != 0 )) && (( GRACEFUL_STOP == 0 )); then
  echo "[error] training exited nonzero (RC=${RC}); not chaining."
  exit "${RC}"
fi

# Figure out where we got to (based on latest saved ckpt)
CKPT2="$(latest_ckpt)"
if [[ -z "${CKPT2}" ]]; then
  echo "[error] no checkpoint found in ${SAVE_DIR}; cannot safely chain."
  exit 1
fi
LAST_UPDATE="$(ckpt_update "${CKPT2}")"

echo "=== SEGMENT DONE ==="
echo "Latest ckpt: ${CKPT2}"
echo "Last saved update: ${LAST_UPDATE}"
echo "===================="

if (( LAST_UPDATE >= TOTAL_UPDATES )); then
  echo "[done] reached total_updates=${TOTAL_UPDATES}. Not chaining."
  exit 0
fi

# Submit continuation that will start only if THIS job exits 0 (afterok)
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
echo "[chain] submitting next segment (afterok:${SLURM_JOB_ID}) using ${SCRIPT_PATH}"

NEXT_JOB_ID="$(
  sbatch --parsable \
    --dependency=afterok:${SLURM_JOB_ID} \
    --export=ALL,RUN_ID=${RUN_ID},TOTAL_UPDATES=${TOTAL_UPDATES},LR=${LR},LR_FINAL=${LR_FINAL},VARIANT=${VARIANT},OBS_MODE=${OBS_MODE},NUM_ENVS=${NUM_ENVS},UNROLL_T=${UNROLL_T},SAVE_INTERVAL=${SAVE_INTERVAL},LOG_INTERVAL=${LOG_INTERVAL},DEVICE=${DEVICE},SEED_OFFSET=${SEED_OFFSET} \
    "${SCRIPT_PATH}"
)"
echo "[chain] queued next job: ${NEXT_JOB_ID}"
