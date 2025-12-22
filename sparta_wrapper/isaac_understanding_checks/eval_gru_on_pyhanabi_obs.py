# eval_gru_on_pyhanabi_obs.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import random
import numpy as np
import torch
from hanabi_learning_environment import rl_env, pyhanabi

# Ensure repo root importability (so sparta_wrapper / hanabi_gru_baseline import works)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparta_wrapper.gru_blueprint import GRUBlueprint
from hanabi_gru_baseline.config import CFG as GRU_CFG
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG


def make_env(seed: int) -> rl_env.HanabiEnv:
    cfg = dict(HANABI_GAME_CONFIG)

    # rl_env wants these keys (it passes them into pyhanabi.HanabiGame)
    # Keep things explicit to avoid relying on backend defaults.
    cfg["seed"] = int(seed)
    cfg["observation_type"] = pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value

    return rl_env.HanabiEnv(config=cfg)


def run_one_episode(ckpt_path: Path, seed: int, render: bool = False) -> float:
    env = make_env(seed)
    obs = env.reset()

    # Two blueprints: one per seat (keeps GRU hidden state separate per player)
    bp0 = GRUBlueprint(ckpt_path=ckpt_path, model_cfg=GRU_CFG, hanabi_cfg=HANABI_GAME_CONFIG)
    bp1 = GRUBlueprint(ckpt_path=ckpt_path, model_cfg=GRU_CFG, hanabi_cfg=HANABI_GAME_CONFIG)
    bps = [bp0, bp1]

    done = False
    total_score_delta = 0.0

    # seed random, torch, and numpy
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    while not done:
        seat = int(obs["current_player"])
        pov = obs["player_observations"][seat]

        # Native pyhanabi observation object (the thing GRUBlueprint expects)
        han_obs = pov["pyhanabi"]

        # Blueprint returns a pyhanabi.HanabiMove
        move = bps[seat].act(han_obs)

        # Convert to DeepMind-native move UID int and step.
        uid = env.game.get_move_uid(move)
        if uid < 0:
            assert False, f"Invalid move UID {uid} for move {move} at seat {seat}"

        obs, rew, done, info = env.step(int(uid))
        true_reward = max(float(rew), 0.0) # bombing doesn't contribute negative
        # rew should be either 0 or 1
        assert true_reward in (0.0, 1.0), f"Unexpected reward {rew} at seat {seat} for move {move}"
        total_score_delta += true_reward

        if render:
            # Minimal render: just print reward/score-ish info.
            # (rl_env's info dict is usually empty; score is the sum of deltas.)
            print(f"seat={seat} uid={uid} rew={float(rew):+.1f} running={total_score_delta:.1f}")

    # In rl_env, sum of score deltas over the whole episode equals final score.
    return float(total_score_delta)


def evaluate(ckpt_path: Path, episodes: int, seed0: int, render: bool = False) -> np.ndarray:
    scores: List[float] = []
    for ep in range(episodes):
        seed = seed0 + ep
        score = run_one_episode(ckpt_path=ckpt_path, seed=seed, render=render)
        scores.append(score)
    return np.asarray(scores, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="Base seed; episode k uses seed+ k.")
    ap.add_argument(
        "--ckpt",
        type=str,
        default="gru_checkpoints/ckpt_020000.pt",
        help="Checkpoint path (default matches hanabi_gru_blueprint_eval.py).",
    )
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Keep torch deterministic-ish (optional)
    torch.manual_seed(args.seed)

    scores = evaluate(ckpt_path=ckpt_path, episodes=args.episodes, seed0=args.seed, render=args.render)
    print(
        f"GRU (pyhanabi obs + CANONICAL) over n={len(scores)} episodes: "
        f"mean={scores.mean():.3f} std={scores.std():.3f} min={scores.min():.1f} max={scores.max():.1f}"
    )


if __name__ == "__main__":
    main()
