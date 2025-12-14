"""Run a single SPARTA Hanabi game and log moves to stdout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import sys
import random
import torch

from hanabi_learning_environment import pyhanabi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch
    sys.path.insert(0, str(ROOT))

from hanabi_gru_baseline.config import CFG as GRU_CFG
from sparta_wrapper.hanabi_utils import build_observation, HanabiLookback1, move_to_dict
from sparta_wrapper.sparta_config import CKPT_PATH, HANABI_GAME_CONFIG, SPARTA_CONFIG
from sparta_wrapper.sparta_search import SpartaGRUWrapper


def _format_card(card: pyhanabi.HanabiCard) -> str:
    color = pyhanabi.COLOR_CHAR[card.color()] if card.color() >= 0 else "?"
    rank = card.rank() + 1
    return f"{color}{rank}"

def _format_hand(hand) -> str:
    if not hand:
        return "-"
    return " ".join(f"{idx}:{_format_card(c)}" for idx, c in enumerate(hand))


def _format_state(state: pyhanabi.HanabiState) -> List[str]:
    parts: List[str] = []
    parts.append(f"Score: {state.score()} | Deck: {state.deck_size()} | Info: {state.information_tokens()} | Lives: {state.life_tokens()}")
    fireworks = " ".join(f"{pyhanabi.COLOR_CHAR[i]}:{v}" for i, v in enumerate(state.fireworks()))
    parts.append(f"Fireworks: {fireworks}")
    discard = " ".join(_format_card(c) for c in state.discard_pile())
    parts.append(f"Discard: {discard if discard else '-'}")
    for pid, hand in enumerate(state.player_hands()):
        parts.append(f"P{pid} hand: {_format_hand(hand)}")
    return parts


def _render_step(step: int, pid: int, action_dict, state: pyhanabi.HanabiState) -> None:
    move_desc = action_dict.get("action_type", "UNKNOWN")
    if "card_index" in action_dict:
        move_desc += f" idx={action_dict['card_index']}"
    if "color" in action_dict:
        move_desc += f" color={action_dict['color']}"
    if "rank" in action_dict:
        move_desc += f" rank={action_dict['rank']}"
    print(f"\n--- Turn {step} | Player {pid} plays {move_desc} ---", flush=True)
    for line in _format_state(state):
        print(line, flush=True)


def run_episode(seed: int, ckpt_path: Path, render: bool = True) -> float:
    rng = random.Random(seed)
    # Seed global RNGs to reduce nondeterminism inside helpers.
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    game = HanabiLookback1(HANABI_GAME_CONFIG, seed)
    sparta_cfg = dict(SPARTA_CONFIG)
    sparta_cfg["rng_seed"] = seed
    agent = SpartaGRUWrapper(ckpt_path, GRU_CFG, sparta_cfg, game)

    turn = 0
    state = game.cur_state

    if render:
        print("\n=== New Game ===", flush=True)
        for line in _format_state(state):
            print(line, flush=True)

    while not state.is_terminal():
        pid = state.cur_player()
        obs = build_observation(state, pid)
        print("Getting move...", flush=True)
        move = agent.act(state, pid)
        print("Done.", flush=True)
        action_dict = move_to_dict(move)

        game.apply_move(move)
        state = game.cur_state
        turn += 1

        if render:
            _render_step(turn, pid, action_dict, state)

    if render:
        print(f"\n=== Terminal | Score {state.score()} ===", flush=True)
    return float(state.score())


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SPARTA-vs-SPARTA Hanabi and log moves.")
    ap.add_argument("--episodes", type=int, default=1, help="Number of games to run.")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed for Hanabi and sampling.")
    ap.add_argument("--ckpt", type=str, default=CKPT_PATH, help="GRU checkpoint path for the blueprint backbone.")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-turn logs.")
    args = ap.parse_args()

    # Respect absolute paths; otherwise resolve relative to CWD.
    ckpt_path = Path(args.ckpt).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = ckpt_path.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    scores = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        score = run_episode(seed, ckpt_path, render=not args.quiet)
        scores.append(score)

    mean_score = sum(scores) / len(scores)
    print(f"\nSPARTA mean score over {args.episodes} episodes: {mean_score:.3f}", flush=True)


if __name__ == "__main__":
    main()
