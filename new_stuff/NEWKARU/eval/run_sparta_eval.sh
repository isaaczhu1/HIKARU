#!/usr/bin/env bash
set -euo pipefail

EPISODES=500
SEED=0
ROLLOUTS=8
EPSILON=0.1
OUTPUT="logs/sparta_heuristic_eval_scores_overnight.txt"

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --episodes N     Number of episodes to run (default: $EPISODES)
  --seed N         Base RNG seed (default: $SEED)
  --rollouts N     Rollouts per action (default: $ROLLOUTS)
  --epsilon X      Deviation threshold (default: $EPSILON)
  --output PATH    File to dump per-episode scores (optional)
  -h, --help       Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)
      EPISODES="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --rollouts)
      ROLLOUTS="$2"; shift 2 ;;
    --epsilon)
      EPSILON="$2"; shift 2 ;;
    --output)
      OUTPUT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

CMD=(conda run -n hanabi python -m experiments.search_full \
  --episodes "$EPISODES" \
  --seed "$SEED" \
  --rollouts "$ROLLOUTS" \
  --epsilon "$EPSILON")

if [[ -n "$OUTPUT" ]]; then
  CMD+=(--dump-scores "$OUTPUT")
fi

printf 'Running: %s\n' "${CMD[*]}"
"${CMD[@]}"
