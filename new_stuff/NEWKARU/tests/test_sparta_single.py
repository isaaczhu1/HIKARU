from __future__ import annotations

from blueprints.heuristic_blueprint import HeuristicBlueprint
from envs.full_hanabi_env import FullHanabiEnv
from search.sparta_single import SpartaSingleAgentWrapper, SpartaConfig


def test_sparta_returns_legal_action():
    env = FullHanabiEnv()
    obs = env.current_observation()
    sparta = SpartaSingleAgentWrapper(lambda: HeuristicBlueprint(), SpartaConfig(num_rollouts=1, epsilon=0.0))
    action = sparta.act_with_env(env, env.current_player(), obs)
    assert action in obs.legal_moves
