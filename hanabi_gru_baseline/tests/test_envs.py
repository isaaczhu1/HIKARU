"""Targeted tests for the HanabiEnv2P observation pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
import types
import numpy as np
import pytest

VEC_LEN = 11

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no branch
    sys.path.insert(0, str(ROOT))


class _FakeHanabiEnv:
    """Minimal stub that mimics rl_env.HanabiEnv for observation tests."""

    def __init__(self, config):
        self.config = config
        self._vec = np.arange(VEC_LEN, dtype=np.float32)
        self.current_player = 0
        self.last_action = None

    def reset(self):  # pragma: no cover - trivial
        self.current_player = 0
        return self._make_obs()

    def step(self, action):  # pragma: no cover - exercised in targeted tests
        self.last_action = action
        players = self.config.get("players", 2)
        self.current_player = (self.current_player + 1) % players
        return self._make_obs(), 0.0, False, {}

    def _make_obs(self):
        player_obs = []
        legal_moves = [
            {"action_type": "PLAY", "card_index": 0},
            {"action_type": "DISCARD", "card_index": 0},
            {"action_type": "REVEAL_COLOR", "color": "R", "target_offset": 1},
            {"action_type": "REVEAL_RANK", "rank": 1, "target_offset": 1},
        ]
        fireworks = {"R": 0}
        for _ in range(self.config.get("players", 2)):
            player_obs.append({
                "vectorized": self._vec.copy(),
                "legal_moves": legal_moves,
                "fireworks": fireworks,
            })
        return {
            "current_player": self.current_player,
            "player_observations": player_obs,
        }


# Install the stub modules before importing the code under test.
if "hanabi_learning_environment" not in sys.modules:  # pragma: no branch
    hle_mod = types.ModuleType("hanabi_learning_environment")
    rl_env_mod = types.ModuleType("hanabi_learning_environment.rl_env")
    rl_env_mod.HanabiEnv = _FakeHanabiEnv
    hle_mod.rl_env = rl_env_mod
    sys.modules["hanabi_learning_environment"] = hle_mod
    sys.modules["hanabi_learning_environment.rl_env"] = rl_env_mod
else:  # pragma: no cover
    sys.modules["hanabi_learning_environment.rl_env"].HanabiEnv = _FakeHanabiEnv

from hanabi_gru_baseline.hanabi_envs import HanabiEnv2P  # noqa: E402  (after stubbing)


def test_observation_vector_length_matches_probe():
    env = HanabiEnv2P(seed=0, obs_conf="minimal")
    obs = env.reset()
    assert obs["obs"].shape == (VEC_LEN,)


def test_reset_preserves_full_vectorized_observation():
    env = HanabiEnv2P(seed=0, obs_conf="minimal")
    obs = env.reset()
    np.testing.assert_array_equal(obs["obs"], np.arange(VEC_LEN, dtype=np.float32))


def test_legal_mask_marks_all_stub_moves(monkeypatch):
    target_mod = sys.modules["hanabi_learning_environment.rl_env"]

    class RecordingEnv(_FakeHanabiEnv):
        pass

    monkeypatch.setattr(target_mod, "HanabiEnv", RecordingEnv)
    env = HanabiEnv2P(seed=0, obs_conf="minimal", players=2, colors=2, ranks=2, hand_size=2)
    obs = env.reset()
    legal_ids = np.flatnonzero(obs["legal_mask"]).tolist()
    raw_obs = env._env._make_obs()
    expected = sorted({env._id_from_rl_action(m) for m in env._legal_moves_from_obs(raw_obs)})
    assert sorted(legal_ids) == expected


def test_hint_actions_include_target_offset(monkeypatch):
    target_mod = sys.modules["hanabi_learning_environment.rl_env"]

    class RecordingEnv(_FakeHanabiEnv):
        pass

    monkeypatch.setattr(target_mod, "HanabiEnv", RecordingEnv)
    env = HanabiEnv2P(seed=0, obs_conf="minimal", players=2, colors=2, ranks=2, hand_size=2)
    env.reset()
    hint_id = env._a_reveal_color0
    obs, reward, terminated, _ = env.step(hint_id)
    assert not terminated
    assert env._env.last_action["action_type"] == "REVEAL_COLOR"
    assert env._env.last_action["target_offset"] == 1


def test_prev_other_action_tracks_last_move(monkeypatch):
    target_mod = sys.modules["hanabi_learning_environment.rl_env"]

    class RecordingEnv(_FakeHanabiEnv):
        pass

    monkeypatch.setattr(target_mod, "HanabiEnv", RecordingEnv)
    env = HanabiEnv2P(seed=0, obs_conf="minimal", players=2, colors=2, ranks=2, hand_size=2)
    obs = env.reset()
    assert int(obs["prev_other_action"]) == env.sentinel_none
    play_id = env._a_play0
    obs, *_ = env.step(play_id)
    assert int(obs["prev_other_action"]) == play_id
