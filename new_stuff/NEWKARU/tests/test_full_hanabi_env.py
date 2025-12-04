from __future__ import annotations

# add ./new_stuff/NEWKARU to sys.path for imports
import os
import sys
path_string = "/Users/isaaczhu/MIT/25-26/HIKARU/new_stuff/NEWKARU"
sys.path.append(os.path.abspath(path_string))

import pytest

from hanabi_learning_environment import pyhanabi

from envs.full_hanabi_env import (
    FullHanabiEnv,
    HanabiObservation,
    _action_dict_to_move,
    _move_to_action_dict,
)


@pytest.fixture
def env() -> FullHanabiEnv:
    return FullHanabiEnv(auto_reset=False)


def test_reset_returns_valid_observation(env: FullHanabiEnv) -> None:
    obs = env.reset(seed=7)
    assert isinstance(obs, HanabiObservation)
    assert env.current_player() == obs.player_id
    assert env.current_player() != pyhanabi.CHANCE_PLAYER_ID
    assert env.legal_moves(as_dict=True) == obs.legal_moves_dict
    assert env.deck_size() == obs.deck_size
    assert env.information_tokens() == obs.information_tokens


def test_step_returns_new_observation(env: FullHanabiEnv) -> None:
    obs = env.reset(seed=3)
    move = obs.legal_moves[0]
    next_obs, reward, done, info = env.step(move)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "score" in info and "last_move" in info
    if not done:
        assert isinstance(next_obs, HanabiObservation)
        assert env.current_player() == next_obs.player_id
        assert env.current_player() != pyhanabi.CHANCE_PLAYER_ID
    else:
        assert next_obs is None


def test_clone_does_not_share_state(env: FullHanabiEnv) -> None:
    env.reset(seed=11)
    clone = env.clone()
    original_hands = env.player_hands()
    clone.step(clone.legal_moves()[0])
    assert env.player_hands() == original_hands


def test_move_serialization_round_trip(env: FullHanabiEnv) -> None:
    obs = env.reset(seed=5)
    for move in obs.legal_moves:
        as_dict = _move_to_action_dict(move)
        rebuilt = _action_dict_to_move(as_dict)
        assert _move_to_action_dict(rebuilt) == as_dict


def test_observation_for_each_player(env: FullHanabiEnv) -> None:
    env.reset(seed=9)
    for pid in range(env.num_players()):
        obs = env.observation_for_player(pid)
        assert obs.player_id == pid
        assert len(obs.observed_hands) == env.num_players()
        assert obs.deck_size == env.deck_size()
        assert obs.fireworks.keys() == env.fireworks().keys()


def test_reset_seed_reproducible() -> None:
    env_a = FullHanabiEnv(auto_reset=False)
    env_b = FullHanabiEnv(auto_reset=False)
    env_a.reset(seed=42)
    env_b.reset(seed=42)
    assert env_a.player_hands() == env_b.player_hands()
    assert env_a.fireworks() == env_b.fireworks()


def test_action_dict_step_matches_move_step() -> None:
    env_move = FullHanabiEnv(auto_reset=False)
    env_dict = FullHanabiEnv(auto_reset=False)
    obs_move = env_move.reset(seed=13)
    env_dict.reset(seed=13)

    move = obs_move.legal_moves[0]
    move_dict = _move_to_action_dict(move)

    env_move.step(move)
    env_dict.step(move_dict)

    assert env_move.player_hands() == env_dict.player_hands()
    assert env_move.information_tokens() == env_dict.information_tokens()
    assert env_move.current_player() == env_dict.current_player()
    assert env_move.fireworks() == env_dict.fireworks()


def test_hint_and_discard_adjust_information_tokens(env: FullHanabiEnv) -> None:
    obs = env.reset(seed=17)
    initial_tokens = env.information_tokens()
    hint_dict = next((m for m in obs.legal_moves_dict if m["action_type"] in {"REVEAL_COLOR", "REVEAL_RANK"}), None)
    assert hint_dict is not None, "expected at least one hint action"

    env.step(hint_dict)
    assert env.information_tokens() == initial_tokens - 1

    obs_after_hint = env.current_observation()
    discard_move = next((m for m in obs_after_hint.legal_moves_dict if m["action_type"] == "DISCARD"), None)
    assert discard_move is not None, "expected discard action to be legal"

    env.step(discard_move)
    assert env.information_tokens() == initial_tokens
