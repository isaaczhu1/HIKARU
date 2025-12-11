from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import List

from hanabi_learning_environment import pyhanabi

from sparta_wrapper.belief_models import sample_world_state
from sparta_wrapper.hanabi_utils import (
    FabricateRollout,
    _debug,
    build_observation,
    HanabiLookback1,
)
from sparta_wrapper.gru_blueprint import (
    FabricationPrimerFactoryFactory,
    NaiveGRUBlueprint,
    SamplerGRUFactoryFactory,
)
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG, DEBUG

class SpartaGRUWrapper:
    def __init__(self, ckpt_path, model_config, sparta_config, game: HanabiLookback1):
        self.blueprint_factory = SamplerGRUFactoryFactory(model_config, ckpt_path)
        self.ckpt_path = ckpt_path
        self.model_config = model_config
        self.sparta_config = sparta_config
        self.rng_seed = getattr(sparta_config, "rng_seed", None)
        self.rng = random.Random(self.rng_seed)
        self.game = game

    def act(self, state, player_id):
        obs = build_observation(state, player_id)
        legal_moves = list(obs.legal_moves)

        # Baseline blueprint action.
        blueprint = NaiveGRUBlueprint(self.model_config, self.ckpt_path)
        blueprint_move = blueprint.act(obs)

        values = {}
        for move in legal_moves:
            values[move] = self._estimate_value(state, player_id, move)

        # Choose best Monte Carlo value.
        best_move = max(values, key=values.get)
        best_val = values[best_move]
        bp_val = values.get(blueprint_move)
        if bp_val is None:
            bp_val = self._estimate_value(state, player_id, blueprint_move)

        if best_val - bp_val < self.sparta_config["epsilon"]:
            return blueprint_move
        return best_move

    def _estimate_value(self, state, player_id, action):
        """
        Monte Carlo estimate of Q(state, action) under the blueprint after the first ply.

        Steps (per rollout):
          1) Sample a hidden hand/world consistent with observation using sample_world_state
             (which primes the GRU blueprint when given a blueprint_factory).
          2) Clone/apply sampled world to a copy of state.  # pseudocode placeholder
          3) Apply the candidate action for player_id.        # pseudocode placeholder
          4) Roll forward with blueprint actions until terminal; record final score.  # pseudocode placeholder
        """
        if DEBUG:
            _debug(f"Called _estimate_value with params  {state} {player_id} {action}")
        obs = build_observation(state, player_id)
        values: List[float] = []
        if DEBUG:
            _debug("Collecting samples...")
        samples = sample_world_state(
            lagging_state=self.game.prev_state,
            obs=obs,
            rng=self.rng,
            blueprint_factory=self.blueprint_factory,
            takes=self.sparta_config["num_rollouts"],
            upstream_factor=self.sparta_config["upstream_factor"],
            max_attempts=self.sparta_config["max_attempts"],
        )
        if DEBUG:
            _debug("Done with samples, priming factory...")
        primer_factory = FabricationPrimerFactoryFactory(self.model_config, self.ckpt_path)
        if DEBUG:
            _debug("Iterating hands...")
            _debug(str(state))
        from tqdm import tqdm

        for hand_guess in tqdm(samples):
            _debug(f"Shit {hand_guess} {player_id}")
            fabrication = FabricateRollout(state, player_id, hand_guess)
            _debug("Done fabricating...")
            actors = [
                primer_factory(fabrication.fabricated_move_history, pid)
                for pid in range(HANABI_GAME_CONFIG["players"])
            ]
            _debug("Primed factories...")

            # Apply the candidate action from the root player, then proceed with chance draws.
            fabrication.apply_move(action)
            fabrication.advance_chance_events()

            _debug(fabrication.state)

            _debug("stabilizing outputs...")
            # import time
            # time.sleep(0.1)


            while not fabrication.is_terminal():
                pid = fabrication.cur_player()
                if DEBUG:
                    _debug(f"One turn of rollout, current player {fabrication.cur_player()}")

                if pid == pyhanabi.CHANCE_PLAYER_ID:
                    fabrication.advance_chance_events()
                    continue

                _debug("making obs")
                _debug(f"{fabrication.state} {pid}")
                rollout_obs = build_observation(fabrication.state, pid)
                _debug("made")
                _debug(str(rollout_obs))
                _debug("acting")
                move = actors[pid].act(rollout_obs)
                _debug(f"actor wants {move}")
                fabrication.apply_move(move)
                fabrication.advance_chance_events()

            values.append(float(fabrication.state.score()))

        return sum(values) / len(values) if values else 0.0

__all__ = ["SpartaConfig", "SpartaGRUWrapper"]
