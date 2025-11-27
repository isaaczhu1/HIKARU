"""Utilities for evaluating blueprint policies in Hanabi."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, Dict, List

from envs.full_hanabi_env import FullHanabiEnv, HanabiObservation, _move_to_action_dict


@dataclass
class EvaluationResult:
    scores: List[int]
    blueprint_stats: List[Dict[str, int]]

    @property
    def mean_score(self) -> float:
        return mean(self.scores) if self.scores else 0.0

    @property
    def stddev_score(self) -> float:
        if len(self.scores) <= 1:
            return 0.0
        return pstdev(self.scores)

    @property
    def perfect_rate(self) -> float:
        if not self.scores:
            return 0.0
        perfect = sum(1 for score in self.scores if score >= 25)
        return perfect / len(self.scores)


def run_self_play(
    blueprint_factory: Callable[[], object],
    *,
    num_episodes: int = 10,
    seed: int = 0,
    log_episodes: int = 0,
) -> EvaluationResult:
    """Runs blueprint self-play and returns aggregate scores."""

    env = FullHanabiEnv()
    num_players = env.num_players()
    blueprints = [blueprint_factory() for _ in range(num_players)]

    scores: List[int] = []
    for episode in range(num_episodes):
        env.reset(seed=seed + episode)
        done = False
        info = {"score": 0}
        step_idx = 0
        while not done:
            pid = env.current_player()
            observation = env.observation_for_player(pid)
            action = self_play_action(blueprints[pid], env, pid, observation)
            _, _, done, info = env.step(action)
            if episode < log_episodes:
                rule = getattr(blueprints[pid], "last_rule", lambda: None)
                if callable(rule):
                    rule_name = rule()
                else:
                    rule_name = rule
                print(
                    f"[ep {episode} step {step_idx}] player={pid} info={env.information_tokens()} "
                    f"life={env.life_tokens()} fireworks={env.fireworks()}"
                )
                print(f"  rule={rule_name} action={_move_to_action_dict(action)}")
            step_idx += 1
        scores.append(int(info["score"]))

    stats: List[Dict[str, int]] = []
    for bp in blueprints:
        if hasattr(bp, "get_rule_counts"):
            stats.append(bp.get_rule_counts())
        else:
            stats.append({})

    return EvaluationResult(scores=scores, blueprint_stats=stats)


def self_play_action(agent, env: FullHanabiEnv, player_id: int, observation: HanabiObservation):
    if hasattr(agent, "act_with_env"):
        return agent.act_with_env(env, player_id, observation)
    return agent.act(observation)


__all__ = ["run_self_play", "EvaluationResult"]
