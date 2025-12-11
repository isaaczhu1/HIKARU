"""Pseudocode sketch for SPARTA 1-ply search wrapper.

Outline:
    - Imports: random/dataclasses/typing; pyhanabi/rl_env if needed; from sparta_wrapper
      import sample_world_state, build_observation, _move_to_action_dict,
      _action_dict_to_move, _advance_chance_events; blueprint factories
      (e.g., CkptGuardFactoryFactory), and SpartaConfig dataclass for rollout params.

    - Config dataclass SpartaConfig:
        fields: num_rollouts (int), epsilon (float deviation threshold), rng_seed,
        max_attempts, etc.

    - Helper _rollout_return(state_clone, first_move, blueprint_factory):
        apply first_move to state_clone; advance chance nodes.
        while not terminal:
            pid = state_clone.cur_player()
            obs = build_observation(state_clone, pid)
            move = blueprint_factory().act(obs)  # follow blueprint after first ply
            state_clone.apply_move(move); _advance_chance_events(state_clone)
        return final score (or accumulated reward).

    - Class SpartaSingleAgent:
        __init__(blueprint_factory, config): store factory/cfg, seed RNG.
        act(state, player_id):
            obs = build_observation(state, player_id)
            bp = blueprint_factory(); bp_move = bp.act(obs)
            legal = obs.legal_moves
            For each move in legal:
                Monte Carlo estimate Q:
                    repeat cfg.num_rollouts:
                        sample hidden world via sample_world_state(
                            lagging_state=state, obs=obs, rng=self.rng,
                            blueprint_factory=CkptGuardFactoryFactory(ckpt_path))
                        # sample_world_state already applies sampled hands and primes GRU,
                        # so no manual apply_sampled_hands or priming needed.
                        state_clone = state.copy() consistent with sample
                        G = _rollout_return(state_clone, move, blueprint_factory)
                    Q[move] = mean(G)
            best_move = argmax Q
            if Q[best_move] - Q[bp_move] < cfg.epsilon:
                return bp_move
            return best_move

Notes:
    - This is single-ply improvement: only the first action varies; rest follow blueprint.
    - Optional UCB/pruning/variance checks can be added before stopping rollouts per action.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from hanabi_learning_environment import pyhanabi

from sparta_wrapper.belief_models import sample_world_state
from sparta_wrapper.hanabi_utils import FabricateRollout, build_observation, HanabiLookback1
from sparta_wrapper.gru_blueprint import (
    FabricationPrimerFactoryFactory,
    NaiveGRUBlueprint,
    SamplerGRUFactoryFactory,
)
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG

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
        print("Called _estimate_value with params ", state, player_id, action)
        obs = build_observation(state, player_id)
        values: List[float] = []
        print("Collecting samples...")
        samples = sample_world_state(
            lagging_state=self.game.prev_state,
            obs=obs,
            rng=self.rng,
            blueprint_factory=self.blueprint_factory,
            takes=self.sparta_config["num_rollouts"],
            upstream_factor=self.sparta_config["upstream_factor"],
            max_attempts=self.sparta_config["max_attempts"],
        )
        primer_factory = FabricationPrimerFactoryFactory(self.model_config, self.ckpt_path)
        # print("Iterating hands...")
        from tqdm import tqdm
        for hand_guess in tqdm(samples):
            fabrication = FabricateRollout(state, player_id, hand_guess)
            # print("Done fabricating...")
            actors = [
                primer_factory(fabrication.fabricated_move_history, pid)
                for pid in range(HANABI_GAME_CONFIG["players"])
            ]
            # print("Primed factories...")

            # Apply the candidate action from the root player, then proceed with chance draws.
            fabrication.apply_move(action)
            fabrication.advance_chance_events()

            while not fabrication.is_terminal():
                pid = fabrication.cur_player()
                # print("One turn of rollout, current player", fabrication.cur_player())

                if pid == pyhanabi.CHANCE_PLAYER_ID:
                    fabrication.advance_chance_events()
                    continue

                rollout_obs = build_observation(fabrication.state, pid)
                move = actors[pid].act(rollout_obs)
                fabrication.apply_move(move)
                fabrication.advance_chance_events()

            values.append(float(fabrication.state.score()))

        return sum(values) / len(values) if values else 0.0

__all__ = ["SpartaConfig", "SpartaGRUWrapper"]
