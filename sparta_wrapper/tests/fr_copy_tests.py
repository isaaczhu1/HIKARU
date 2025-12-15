from __future__ import annotations

import random
import sys
import os

# Ensure repo root is on sys.path so `sparta_wrapper` is importable when running directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dataclasses import dataclass
from typing import List

from hanabi_learning_environment import pyhanabi

from sparta_wrapper.belief_models import sample_world_state
from sparta_wrapper.hanabi_utils import (
    FabricateRollout,
    _debug,
    build_observation,
    HanabiLookback1,
    _advance_chance_events
)
from sparta_wrapper.gru_blueprint import (
    FabricationPrimerFactoryFactory,
    NaiveGRUBlueprint,
    SamplerGRUFactoryFactory,
)
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG, DEBUG


cfg = HANABI_GAME_CONFIG
cfg["seed"]=0

game = pyhanabi.HanabiGame(cfg)
state = game.new_initial_state()
print(state)
    
while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
    _advance_chance_events(state)
    continue

print(state)

state.apply_move(build_observation(state, 0).legal_moves[0])
print(state)

while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
    _advance_chance_events(state)
    continue




fr_orig = FabricateRollout(state, 1, [(1, 1), (1, 1), (1, 3), (1, 3), (2, 2)])


fr_copy1 = FabricateRollout(state, 1, [(1, 1), (1, 1), (1, 3), (1, 3), (2, 2)], fr_orig.remaining_deck)

fr_copy2 = FabricateRollout(state, 1, [(1, 1), (1, 1), (1, 3), (1, 3), (2, 2)], fr_orig.remaining_deck)
