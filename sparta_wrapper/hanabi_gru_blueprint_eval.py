"""Evaluate the GRU blueprint (no SPARTA) against the Hanabi environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, List

import torch
from hanabi_learning_environment import rl_env
from hanabi_learning_environment import pyhanabi
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch
    sys.path.insert(0, str(ROOT))

from sparta_wrapper.hanabi_utils import _move_to_action_dict, build_observation
from sparta_wrapper.naive_gru_blueprint import GRU_CFG, HanabiGRUBlueprint


def _format_card(card: pyhanabi.HanabiCard) -> str:
    color = pyhanabi.COLOR_CHAR[card.color()] if card.color() >= 0 else "?"
    rank = card.rank() + 1  # display 1-based rank
    return f"{color}{rank}"


def _format_state(state: pyhanabi.HanabiState) -> str:
    parts = []
    parts.append(f"Score: {state.score()} | Deck: {state.deck_size()} | Info: {state.information_tokens()} | Lives: {state.life_tokens()}")
    fireworks = " ".join(
        f"{pyhanabi.COLOR_CHAR[i]}:{v}" for i, v in enumerate(state.fireworks())
    )
    parts.append(f"Fireworks: {fireworks}")
    discard = " ".join(_format_card(c) for c in state.discard_pile())
    parts.append(f"Discard: {discard if discard else '-'}")
    for pid, hand in enumerate(state.player_hands()):
        hand_s = " ".join(_format_card(c) for c in hand)
        parts.append(f"P{pid} hand: {hand_s}")
    return "\n".join(parts)


def _render_step(step: int, pid: int, action_dict, state: pyhanabi.HanabiState) -> None:
    move_desc = action_dict.get("action_type", "UNKNOWN")
    if "card_index" in action_dict:
        move_desc += f" idx={action_dict['card_index']}"
    if "color" in action_dict:
        move_desc += f" color={action_dict['color']}"
    if "rank" in action_dict:
        move_desc += f" rank={action_dict['rank']}"
    print(f"\n--- Turn {step} | Player {pid} plays {move_desc} ---")
    print(_format_state(state))


def _run_episode(env: rl_env.HanabiEnv, actors: List[Callable], render: bool = False) -> float:
    env.reset()
    state = env.state
    if render:
        print("\n=== New Game ===")
        print(_format_state(state))

    turn = 0
    while not state.is_terminal():
        pid = state.cur_player()
        player_obs = build_observation(state, pid)
        action_move = actors[pid](player_obs)
        action_dict = _move_to_action_dict(action_move)
        _, _, done, info = env.step(action_dict)
        state = env.state
        turn += 1
        if render:
            _render_step(turn, pid, action_dict, state)
        if done:
            return float(info.get("score", state.score()))
    return float(state.score())


def evaluate(episodes: int, blueprint_factory: Callable[[], object], writer: SummaryWriter | None = None, prefix: str = "gru_blueprint", render: bool = False) -> float:
    env = rl_env.HanabiEnv({"players": 2})
    scores = []
    for ep in range(episodes):
        bps = [blueprint_factory(), blueprint_factory()]
        actors = [lambda obs, bp=bps[0]: bp.act(obs), lambda obs, bp=bps[1]: bp.act(obs)]
        score = _run_episode(env, actors, render=render)
        scores.append(score)
        if writer is not None:
            writer.add_scalar(f"{prefix}/score", score, ep + 1)
            writer.add_scalar(f"{prefix}/running_mean", sum(scores) / len(scores), ep + 1)
    return sum(scores) / len(scores)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a Hanabi GRU blueprint (no SPARTA).")
    ap.add_argument("--episodes", type=int, default=20, help="Number of episodes to average.")
    ap.add_argument("--ckpt", type=str, default="gru_checkpoints/ckpt_020000.pt", help="Checkpoint path.")
    ap.add_argument("--device", type=str, default="cpu", help="Device for GRU blueprint (cpu|cuda).")
    ap.add_argument("--logdir", type=str, default="", help="TensorBoard log directory (empty to disable).")
    ap.add_argument("--render", action="store_true", help="Print an omniscient, human-readable game log.")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    blueprint_factory = lambda: HanabiGRUBlueprint(GRU_CFG(), ckpt_path, device=args.device)

    writer: SummaryWriter | None = None
    if args.logdir:
        writer = SummaryWriter(args.logdir)

    mean_score = evaluate(
        episodes=args.episodes,
        blueprint_factory=blueprint_factory,
        writer=writer,
        prefix="gru_blueprint",
        render=args.render,
    )

    if writer is not None:
        writer.add_scalar("summary/mean", mean_score, 0)
        writer.flush()
        writer.close()

    print(f"GRU blueprint mean over {args.episodes} episodes: {mean_score:.3f}")


if __name__ == "__main__":
    main()
