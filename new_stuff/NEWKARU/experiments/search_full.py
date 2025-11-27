"""Evaluate SPARTA wrapper on top of heuristic blueprint."""

from __future__ import annotations

import argparse

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
        scores.append(info["score"])

    mean_score = sum(scores) / len(scores)
    print(f"Episodes: {len(scores)}")
    print(f"Mean score: {mean_score:.2f}")


if __name__ == "__main__":
    main()
