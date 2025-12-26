"""spam imports"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, List
from tqdm import tqdm

print("importing torch")
import torch
print("imported torch")
import random
from hanabi_learning_environment import rl_env
from hanabi_learning_environment import pyhanabi
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch
    sys.path.insert(0, str(ROOT))

from sparta_wrapper.gru_blueprint import GRUBlueprint

import sys
from pathlib import Path

# Ensure the repo root is importable so sibling package hanabi_gru_baseline can be used.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hanabi_learning_environment import pyhanabi
from hanabi_gru_baseline.config import CFG as GRU_CFG
from sparta_wrapper.hanabi_utils import *
from sparta_wrapper.sparta_config import *
from sparta_wrapper.sparta import *


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


def run_episode(seed: int, ckpt_path: Path, deviator = None, render: bool = True) -> float:
    print("Running episode with seed", seed, "for ckpt", ckpt_path)
    random.seed(seed)
    torch.manual_seed(seed)
    
    actor_blueprints = [lambda: GRUBlueprint(ckpt_path, GRU_CFG, HANABI_GAME_CONFIG) for i in range(HANABI_GAME_CONFIG["players"])]

    if deviator is not None: # deviator is None <-> no sparta
        actor_blueprints[deviator] = lambda: SpartaAgent(GRUBlueprint(ckpt_path, GRU_CFG, HANABI_GAME_CONFIG), HANABI_GAME_CONFIG, SPARTA_CONFIG)

    seeded_game_config = dict(HANABI_GAME_CONFIG)
    seeded_game_config["seed"] = seed
    seeded_game_config["random_start_player"] = False # just in case
    
    # _game = pyhanabi.HanabiGame(seeded_game_config) # wut
    # _state = _game.new_initial_state()
    
    game = SimulatedGame(
        history=[], 
        set_deck=None, 
        actor_blueprints=actor_blueprints, 
        hanabi_game_config=seeded_game_config
    )
    
    while not game.terminal():
        game.step()
        
    return game.peak_score


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SPARTA-vs-SPARTA Hanabi and log moves.")
    ap.add_argument("--episodes", type=int, default=20, help="Number of games to run.")
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

    baseline_scores = []
    
    print(f"====== Running non-SPARTA evaluation over {args.episodes} episodes ======", flush=True)
    for ep in tqdm(range(args.episodes)):
        manual_seed = args.seed + ep
        score = run_episode(seed=manual_seed, ckpt_path=ckpt_path, deviator=None, render=(not args.quiet))
        baseline_scores.append(score)

    print(f"Baseline scores: {baseline_scores}", flush=True)

    print(f"====== Running SPARTA evaluation over {args.episodes} episodes ======", flush=True)
    sparta_scores = []
    for ep in tqdm(range(args.episodes)):
        manual_seed = args.seed + ep
        deviator = 0 # for now, player 0 is always the SPARTA agent
        score = run_episode(seed=manual_seed, ckpt_path=ckpt_path, deviator=deviator, render=(not args.quiet))
        sparta_scores.append(score)

    print(f"SPARTA scores: {sparta_scores}", flush=True)

    pointwise_diffs = [s - b for s, b in zip(sparta_scores, baseline_scores)]
    print(f"Pointwise score differences (SPARTA - Baseline): {pointwise_diffs}", flush=True)



if __name__ == "__main__":
    main()
