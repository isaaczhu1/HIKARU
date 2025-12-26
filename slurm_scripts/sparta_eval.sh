#!/usr/bin/env bash
#SBATCH --job-name=sparta_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

set -euo pipefail
mkdir -p slurm_outputs sparta_eval_results

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hanabi

cd ~/HIKARU
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CKPT="${CKPT:-gru_checkpoints/ckpt_020000.pt}"
SEED="${SEED:-0}"
EPISODES="${EPISODES:-20}"

# 0=minimal, 1=card_knowledge, 2=seer, 3=rich_card_knowledge
OBS_TYPE="${OBS_TYPE:-1}"

MODE="${MODE:-both}"      # baseline | sparta | both
DEVIATOR="${DEVIATOR:-0}"

OUT_JSON="sparta_eval_results/${SLURM_JOB_ID}.json"

echo "JobID=${SLURM_JOB_ID:-none}"
echo "CKPT=${CKPT}"
echo "SEED=${SEED}"
echo "EPISODES=${EPISODES}"
echo "OBS_TYPE=${OBS_TYPE}"
echo "MODE=${MODE} DEVIATOR=${DEVIATOR}"
echo "OUT_JSON=${OUT_JSON}"

srun python -u sparta_wrapper/sparta_eval_chunk.py \
  --episodes "${EPISODES}" \
  --start-ep 0 \
  --seed "${SEED}" \
  --ckpt "${CKPT}" \
  --mode "${MODE}" \
  --deviator "${DEVIATOR}" \
  --obs-type "${OBS_TYPE}" \
  --out-json "${OUT_JSON}" \
  --quiet