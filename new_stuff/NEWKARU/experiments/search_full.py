"""Evaluate SPARTA wrapper on top of heuristic blueprint."""

from __future__ import annotations

import argparse

from datetime import datetime

from blueprints.heuristic_blueprint import HeuristicBlueprint
from envs.full_hanabi_env import FullHanabiEnv
from search.evaluation import self_play_action
from search.sparta_single import SpartaSingleAgentWrapper, SpartaConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heuristic vs SPARTA evaluation")
    parser.add_argument("--episodes", type=int, default=200, help="number of episodes")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--rollouts", type=int, default=32, help="rollouts per action")
    parser.add_argument("--epsilon", type=float, default=0.1, help="minimum gain to deviate")
    parser.add_argument("--dump-scores", type=str, default="", help="path to write per-episode scores")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = FullHanabiEnv()
    agents = [
        SpartaSingleAgentWrapper(
            lambda: HeuristicBlueprint(),
            SpartaConfig(num_rollouts=args.rollouts, epsilon=args.epsilon, seed=args.seed),
        ),
        HeuristicBlueprint(),
    ]

    score_file = None
    if args.dump_scores:
        score_file = open(args.dump_scores, "w", encoding="utf-8")

    scores = []
    for episode in range(args.episodes):
        env.reset(seed=args.seed + episode)
        done = False
        info = {"score": 0}
        while not done:
            pid = env.current_player()
            obs = env.observation_for_player(pid)
            action = self_play_action(agents[pid], env, pid, obs)
            _, _, done, info = env.step(action)
        score = info["score"]
        scores.append(score)
        if score_file is not None:
            ts = datetime.now().isoformat(timespec="seconds")
            score_file.write(f"{ts}\t{score}\n")
            score_file.flush()

    mean_score = sum(scores) / len(scores)
    print(f"Episodes: {len(scores)}")
    print(f"Mean score: {mean_score:.2f}")
    print(f"Scores: {scores}")

    if score_file is not None:
        score_file.close()
        print(f"Wrote per-episode scores to {args.dump_scores}")


if __name__ == "__main__":
    main()
