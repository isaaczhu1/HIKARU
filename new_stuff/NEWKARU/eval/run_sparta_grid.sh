#!/usr/bin/env bash
set -euo pipefail

EPISODES=10
SEED=0
ROLLOUTS_LIST=(2 4 8 16)
EPSILONS=(0.0 0.05 0.1 0.2)
OUTDIR=""

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --episodes N        Episodes per grid point (default: $EPISODES)
  --seed N            Base RNG seed (default: $SEED)
  --rollouts "a b"    Space-separated rollout counts (default: ${ROLLOUTS_LIST[*]})
  --epsilons "x y"    Space-separated epsilon values (default: ${EPSILONS[*]})
  --outdir PATH       Directory to write per-episode score files/logs
  -h, --help          Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)
      EPISODES="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --rollouts)
      IFS=' ' read -r -a ROLLOUTS_LIST <<< "$2"; shift 2 ;;
    --epsilons)
      IFS=' ' read -r -a EPSILONS <<< "$2"; shift 2 ;;
    --outdir)
      OUTDIR="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -n "$OUTDIR" ]]; then
  mkdir -p "$OUTDIR"
fi

for rollout in "${ROLLOUTS_LIST[@]}"; do
  for eps in "${EPSILONS[@]}"; do
    label="r${rollout}_eps${eps//./p}"
    dump_arg=()
    if [[ -n "$OUTDIR" ]]; then
      dump_file="$OUTDIR/scores_${label}.txt"
      dump_arg=(--dump-scores "$dump_file")
    fi
    echo "==> Running rollouts=$rollout epsilon=$eps"
    cmd=(conda run -n hanabi python -m experiments.search_full \
      --episodes "$EPISODES" \
      --seed "$SEED" \
      --rollouts "$rollout" \
      --epsilon "$eps" \
      "${dump_arg[@]}" )
    # shellcheck disable=SC2128
    echo "Command: ${cmd[*]}"
    "${cmd[@]}"
    if [[ -n "$OUTDIR" ]]; then
      echo "Scores written to $dump_file"
    fi
  done
done
