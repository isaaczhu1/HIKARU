from __future__ import annotations

import random

# add ./new_stuff/NEWKARU to sys.path for imports
import os
import sys
path_string = "/Users/isaaczhu/MIT/25-26/HIKARU/new_stuff/NEWKARU"
sys.path.append(os.path.abspath(path_string))

from envs.full_hanabi_env import FullHanabiEnv
from search.belief_models import sample_world_state


def test_sample_world_state_shapes():
    env = FullHanabiEnv()
    obs = env.current_observation()
    world = sample_world_state(obs, random.Random(0))
    assert len(world.hands) == env.num_players()
    assert len(world.hands[0]) == len(obs.card_knowledge[0])
    assert isinstance(world.deck, list)
