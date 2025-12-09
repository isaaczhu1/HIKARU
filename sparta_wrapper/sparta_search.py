"""SPARTA-lite single-agent search wrapper for Hanabi."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Sequence

from hanabi_learning_environment import pyhanabi

from sparta_wrapper.belief_models import apply_sampled_hands, sample_world_state
from sparta_wrapper.hanabi_utils import HanabiObservation, _advance_chance_events, build_observation


@dataclass
class SpartaConfig:
    num_rollouts: int = 8
    epsilon: float = 0.1
    seed: int = 0
    max_depth: int | None = 25
    max_actions: int | None = 6


class SpartaSingleAgent:
    """One-ply Monte Carlo search wrapper around a blueprint policy."""

    def __init__(self, blueprint_factory: Callable[[], object], config: SpartaConfig | None = None) -> None:
        self.blueprint_factory = blueprint_factory
        self.config = config or SpartaConfig()
        self.rng = random.Random(self.config.seed)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def act(self, state: pyhanabi.HanabiState, player_id: int) -> pyhanabi.HanabiMove:
        """Choose an action for ``player_id`` given the current state."""
        root_obs = build_observation(state, player_id)
        legal_moves = self._candidate_actions(root_obs)
        if not legal_moves:
            raise RuntimeError("No legal moves available for SPARTA.")

        baseline_policy = self.blueprint_factory()
        baseline_action = baseline_policy.act(root_obs)
        baseline_value = self._estimate_action_value(state, root_obs, baseline_action)

        best_action = baseline_action
        best_value = baseline_value

        for move in legal_moves:
            if move == baseline_action:
                continue
            value = self._estimate_action_value(state, root_obs, move)
            if value > best_value:
                best_value = value
                best_action = move

        if best_value >= baseline_value + self.config.epsilon:
            return best_action
        return baseline_action

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _estimate_action_value(
        self, state: pyhanabi.HanabiState, obs: HanabiObservation, action: pyhanabi.HanabiMove
    ) -> float:
        returns: List[float] = []
        for _ in range(self.config.num_rollouts):
            returns.append(self._simulate_rollout(state, obs, action))
        return sum(returns) / len(returns)

    def _simulate_rollout(
        self, state: pyhanabi.HanabiState, obs: HanabiObservation, action: pyhanabi.HanabiMove
    ) -> float:
        sim_state = state.copy()
        sampled = sample_world_state(sim_state, obs, self.rng, blueprint_factory=self.blueprint_factory)
        apply_sampled_hands(sim_state, sampled)

        blueprints = [self.blueprint_factory() for _ in range(sim_state.num_players())]
        self._apply_move(sim_state, action)
        _advance_chance_events(sim_state, deck_override=sampled.deck)

        depth = 0
        while not sim_state.is_terminal():
            if self.config.max_depth is not None and depth >= self.config.max_depth:
                break
            pid = sim_state.cur_player()
            obs = build_observation(sim_state, pid)
            move = blueprints[pid].act(obs)
            self._apply_move(sim_state, move)
            _advance_chance_events(sim_state, deck_override=sampled.deck)
            depth += 1

        return float(sim_state.score())

    def _apply_move(self, state: pyhanabi.HanabiState, move: pyhanabi.HanabiMove) -> None:
        state.apply_move(move)

    def _candidate_actions(self, obs: HanabiObservation) -> List[pyhanabi.HanabiMove]:
        legal_moves = list(obs.legal_moves)
        if self.config.max_actions is None or len(legal_moves) <= int(self.config.max_actions):
            return legal_moves
        # Always consider the blueprint's first choice, then fill deterministically.
        return legal_moves[: int(self.config.max_actions)]


__all__ = ["SpartaSingleAgent", "SpartaConfig"]
