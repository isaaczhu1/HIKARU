"""Smoke test to ensure hanabi_learning_environment is installed and usable."""
from __future__ import annotations

import argparse
import random
from typing import Tuple

from hanabi_learning_environment import pyhanabi


DEFAULT_CONFIG = {
    "players": 2,
    "hand_size": 5,
    "max_information_tokens": 8,
    "max_life_tokens": 3,
    "colors": 5,
    "ranks": 5,
    "seed": 0,
}


def run_random_episode(seed: int = 0) -> Tuple[int, int]:
    """Runs one random-play Hanabi episode and returns (score, turns)."""
    rng = random.Random(seed)
    config = dict(DEFAULT_CONFIG)
    config["seed"] = seed
    game = pyhanabi.HanabiGame(config)
    state = game.new_initial_state()
    turns = 0

    while not state.is_terminal():
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            state.deal_random_card()
            continue
        legal_moves = state.legal_moves()
        move = rng.choice(legal_moves)
        state.apply_move(move)
        turns += 1

    return state.score(), turns


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple Hanabi smoke test")
    parser.add_argument("--episodes", type=int, default=3, help="number of random episodes to simulate")
    parser.add_argument("--seed", type=int, default=0, help="base random seed")
    args = parser.parse_args()

    for episode in range(args.episodes):
        score, turns = run_random_episode(seed=args.seed + episode)
        print(f"Episode {episode}: score={score}, turns={turns}")

    print("hanabi_learning_environment import and basic simulation succeeded.")


if __name__ == "__main__":
    main()
