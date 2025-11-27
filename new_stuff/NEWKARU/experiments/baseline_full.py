"""Run heuristic blueprint self-play for 2-player Hanabi."""

from __future__ import annotations

import argparse

from collections import Counter

from blueprints.heuristic_blueprint import HeuristicBlueprint
from blueprints.random_baseline import RandomHintDiscardBlueprint
from search.evaluation import run_self_play


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate heuristic blueprint self-play")
    parser.add_argument("--episodes", type=int, default=100, help="number of episodes to run")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    parser.add_argument("--policy", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--log-episodes", type=int, default=0, help="print per-step logs for first N episodes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.policy == "heuristic":
        factory = lambda: HeuristicBlueprint()
    else:
        factory = lambda: RandomHintDiscardBlueprint(seed=args.seed)

    result = run_self_play(
        factory,
        num_episodes=args.episodes,
        seed=args.seed,
        log_episodes=args.log_episodes,
    )
    print(f"Episodes: {len(result.scores)}")
    print(f"Mean score: {result.mean_score:.2f}")
    print(f"Stddev: {result.stddev_score:.2f}")
    print(f"Perfect rate: {result.perfect_rate * 100:.1f}%")

    if args.policy == "heuristic":
        totals = Counter()
        for stats in result.blueprint_stats:
            totals.update(stats)
        print("Rule usage counts:")
        for rule, count in sorted(totals.items()):
            print(f"  {rule}: {count}")


if __name__ == "__main__":
    main()
