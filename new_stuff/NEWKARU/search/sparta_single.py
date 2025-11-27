"""SPARTA-lite single-agent search wrapper."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable, List, Optional

from hanabi_learning_environment import pyhanabi

from envs.full_hanabi_env import FullHanabiEnv, HanabiObservation
from search.belief_models import WorldState, sample_world_state


@dataclass
class SpartaConfig:
    num_rollouts: int = 32
    epsilon: float = 0.1
    seed: int = 0


class SpartaSingleAgentWrapper:
    """One-ply Monte Carlo search wrapper around a blueprint policy."""

    def __init__(self, blueprint_factory: Callable[[], object], config: Optional[SpartaConfig] = None) -> None:
        self.blueprint_factory = blueprint_factory
        self.config = config or SpartaConfig()
        self.rng = random.Random(self.config.seed)

    def act_with_env(self, env: FullHanabiEnv, player_id: int, observation: HanabiObservation) -> pyhanabi.HanabiMove:
        legal_moves = list(observation.legal_moves)
        baseline_action = legal_moves[0]
        baseline_value = self._estimate_action_value(env, player_id, observation, baseline_action)

        best_action = baseline_action
        best_value = baseline_value

        for move in legal_moves:
            if move == baseline_action:
                continue
            value = self._estimate_action_value(env, player_id, observation, move)
            if value > best_value:
                best_value = value
                best_action = move

        if best_value >= baseline_value + self.config.epsilon:
            return best_action
        return baseline_action

    def _estimate_action_value(
        self,
        env: FullHanabiEnv,
        player_id: int,
        observation: HanabiObservation,
        action: pyhanabi.HanabiMove,
    ) -> float:
        returns: List[float] = []
        for _ in range(self.config.num_rollouts):
            returns.append(self._simulate_rollout(env, player_id, observation, action))
        return sum(returns) / len(returns)

    def _simulate_rollout(
        self,
        env: FullHanabiEnv,
        player_id: int,
        observation: HanabiObservation,
        action: pyhanabi.HanabiMove,
    ) -> float:
        world = sample_world_state(observation, self.rng)
        sim_env = env.clone()
        sim_env.step(action)

        blueprints = [self.blueprint_factory() for _ in range(sim_env.num_players())]

        done = sim_env.is_terminal()
        info_score = sim_env.score()
        while not done:
            pid = sim_env.current_player()
            obs = sim_env.observation_for_player(pid)
            move = blueprints[pid].act(obs)
            _, _, done, info = sim_env.step(move)
            info_score = info["score"]
        return float(info_score)


__all__ = ["SpartaSingleAgentWrapper", "SpartaConfig"]
