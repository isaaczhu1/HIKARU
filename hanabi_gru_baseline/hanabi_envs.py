# hanabi_envs.py
# -----------------------------------------------------------------------------
# HanabiGym2P over DeepMind rl_env.HanabiEnv (2 players), observation-driven.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from hanabi_learning_environment import rl_env

# Debug prints (export HANABI_DEBUG=1 to enable)
_DEBUG = bool(int(os.environ.get("HANABI_DEBUG", "0")))


class HanabiGym2P(gym.Env):
    """
    Gymnasium-compatible wrapper around DeepMind's rl_env.HanabiEnv for 2-player Hanabi.

    Observations:
      - "obs": flat float32 vector (DeepMind "vectorized" obs for the current player)
      - "legal_mask": float32[ num_moves ] with 1.0 for legal action ids, 0.0 otherwise
      - "seat": int {0,1}, the current player index
      - "prev_other_action": int in [0..num_moves] where num_moves is a sentinel "none"

    Actions:
      action_id in [0..num_moves-1], with layout:
        0..H-1           : PLAY card_index = i
        H..2H-1          : DISCARD card_index = i-H
        2H..2H+C-1       : REVEAL_COLOR color = i-(2H)
        2H+C..2H+C+R-1   : REVEAL_RANK  rank  = i-(2H+C)
    """

    metadata = {"render_modes": []}

    def __init__(self, seed, obs_conf,
                 players=2, colors=5, ranks=5, hand_size=5,
                 max_information_tokens=8, max_life_tokens=3,
                 random_start_player=False):
        super().__init__()
        self.players = players
        self.colors = colors
        self.ranks = ranks
        self.hand_size = hand_size

        # Underlying DeepMind rl_env
        self._env = rl_env.HanabiEnv(config={
            "players": players,
            "colors": colors,
            "ranks": ranks,
            "hand_size": hand_size,
            "max_information_tokens": max_information_tokens,
            "max_life_tokens": max_life_tokens,
            "random_start_player": random_start_player,
            "seed": seed,
        })

        # --- probe real vectorized obs length once, set stable obs space ---
        obs_dim = 1
        try:
            _probe = self._env.reset()
            _seat = int(_probe.get("current_player", 0))
            _pov = _probe["player_observations"][_seat]
            _vec = np.asarray(_pov.get("vectorized", []), dtype=np.float32)
            if _vec.ndim == 1 and _vec.size > 0:
                obs_dim = int(_vec.size)
        except Exception:
            pass

        # --- dynamic action layout ---
        self._a_play0         = 0
        self._a_discard0      = self._a_play0 + hand_size
        self._a_reveal_color0 = self._a_discard0 + hand_size
        self._a_reveal_rank0  = self._a_reveal_color0 + colors

        self.num_moves = 2 * hand_size + colors + ranks
        self.sentinel_none = self.num_moves      # sentinel for "no previous action"
        self._hint_target_offset = 1 if players > 1 else 0

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
            "legal_mask": spaces.Box(0.0, 1.0, shape=(self.num_moves,), dtype=np.float32),
            "seat": spaces.Discrete(players),
            "prev_other_action": spaces.Discrete(self.num_moves + 1),
        })
        self.action_space = spaces.Discrete(self.num_moves)

        # Runtime state
        self._last_obs = None
        self._score_accum = 0.0   # running sum of rewards (our notion of score)
        self.prev_action = [self.sentinel_none for _ in range(self.players)]  # per-seat last action id
        self.turn_count = 0

    # ----------------------------- Utilities -------------------------------- #
    @staticmethod
    def _seat_from_obs(obs: dict) -> int:
        return int(obs.get("current_player", 0))

    def _legal_moves_from_obs(self, obs: dict):
        p = self._seat_from_obs(obs)
        return list(obs["player_observations"][p].get("legal_moves", []))

    # ID layout helpers
    def _id_for_play(self, idx: int) -> int:
        return idx

    def _id_for_discard(self, idx: int) -> int:
        return self.hand_size + idx

    def _id_for_reveal_color(self, color: int) -> int:
        return 2 * self.hand_size + color

    def _id_for_reveal_rank(self, rank: int) -> int:
        return 2 * self.hand_size + self.colors + rank

    def _parse_color(self, c) -> int:
        """Map rl_env color payloads ('R','Y','G','B','W' or names/ints) -> 0..colors-1."""
        if isinstance(c, int):
            return max(0, min(c, self.colors - 1))
        s = str(c).strip().lower()
        letter_map = {"r": 0, "y": 1, "g": 2, "b": 3, "w": 4}
        if s in letter_map:
            return max(0, min(letter_map[s], self.colors - 1))
        name_map = {"red": 0, "yellow": 1, "green": 2, "blue": 3, "white": 4}
        if s in name_map:
            return max(0, min(name_map[s], self.colors - 1))
        return 0  # safe fallback

    def _parse_rank(self, r) -> int:
        """Map rl_env rank payloads ('1'..'5' or ints 0..4/1..5) -> 0..ranks-1."""
        if isinstance(r, int):
            if 0 <= r < self.ranks:  # already 0-based
                return r
            if 1 <= r <= self.ranks:  # 1-based
                return r - 1
            return max(0, min(r, self.ranks - 1))
        s = str(r).strip()
        if s.isdigit():
            v = int(s)
            return max(0, min(v - 1, self.ranks - 1))
        return 0

    def _id_from_rl_action(self, a):
        t = self._act_type(a)
        if t == "PLAY":
            return self._a_play0 + int(a["card_index"])
        if t == "DISCARD":
            return self._a_discard0 + int(a["card_index"])
        if t == "REVEAL_COLOR":
            # rl_env color is typically 0..colors-1 (or a letter); handle both
            return self._a_reveal_color0 + self._parse_color(a.get("color", 0))
        if t == "REVEAL_RANK":
            # rl_env rank is usually 1..ranks
            rank1 = int(a.get("rank", 1))
            return self._a_reveal_rank0 + (rank1 - 1)
        raise ValueError(f"Unknown action dict: {a}")

    def _rl_action_from_id(self, gid):
        if gid < self._a_discard0:
            return {"action_type": "PLAY", "card_index": gid - self._a_play0}
        if gid < self._a_reveal_color0:
            return {"action_type": "DISCARD", "card_index": gid - self._a_discard0}
        if gid < self._a_reveal_rank0:
            return {
                "action_type": "REVEAL_COLOR",
                "color": gid - self._a_reveal_color0,
                "target_offset": self._hint_target_offset,
            }
        return {
            "action_type": "REVEAL_RANK",
            "rank": (gid - self._a_reveal_rank0) + 1,
            "target_offset": self._hint_target_offset,
        }

    def _rl_action_matches_id(self, a: dict, gid: int) -> bool:
        H, C, R = self.hand_size, self.colors, self.ranks
        t = self._act_type(a)
        if t is None:
            return False

        # Quick range check per bucket
        if 0 <= gid < H:
            if t != "PLAY" or "card_index" not in a:
                return False
        elif H <= gid < 2 * H:
            if t != "DISCARD" or "card_index" not in a:
                return False
        elif 2 * H <= gid < 2 * H + C:
            if t != "REVEAL_COLOR":
                return False
        elif 2 * H + C <= gid < 2 * H + C + R:
            if t != "REVEAL_RANK":
                return False
        else:
            return False

        return self._id_from_rl_action(a) == gid

    # ------------------------------- Gym API -------------------------------- #
    def _reset_world(self, seed=None):
        try:
            obs = self._env.reset(seed=seed) if seed is not None else self._env.reset()
        except TypeError:
            obs = self._env.reset()

        self._last_obs = obs
        self._score_accum = 0.0
        self.prev_action = [self.sentinel_none for _ in range(self.players)]
        self.turn_count = 0
        return obs

    def _act_type(self, a: dict) -> str:
        """Return action type string from rl_env action dict."""
        return a.get("action_type", a.get("type", None))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs = self._reset_world(seed)
        return self._pack_obs(obs), {}

    def step(self, a_id: int):
        # Build legal mask from last obs; choose matching action dict
        legal = self._legal_moves_from_obs(self._last_obs)
        if not legal:
            # Defensive reset (should be rare)
            obs = self._reset_world()
            info = {"score": float(self._score_accum)}
            return self._pack_obs(obs), 0.0, False, False, info

        choice = None
        for a in legal:
            if self._rl_action_matches_id(a, a_id):
                choice = a
                break
        if choice is None:
            choice = legal[0]

        # Step underlying env
        next_obs, env_rew, done, info = self._env.step(choice)
        self._last_obs = next_obs

        # Reward shaping: ensure non-negative reward increments
        # DeepMind Hanabi's reward is typically the score delta, but we clamp to >= 0.
        shaped = float(max(0.0, float(env_rew)))

        # Accumulate shaped reward as our notion of "score"
        self._score_accum += shaped

        # Make sure info is a dict and always carries the current accumulated score
        if info is None:
            info = {}
        if not isinstance(info, dict):
            info = dict(info)
        info["score"] = float(self._score_accum)

        # prev action bookkeeping:
        seat_now = self._seat_from_obs(next_obs)
        just_acted = (seat_now - 1) % self.players
        if 0 <= just_acted < self.players:
            self.prev_action[just_acted] = int(a_id)

        self.turn_count += 1
        terminated = bool(done)
        truncated = False
        return self._pack_obs(next_obs), shaped, terminated, truncated, info

    # ------------------------------ Packing --------------------------------- #
    def _pack_obs(self, obs: dict) -> dict:
        seat = self._seat_from_obs(obs)

        # Use DeepMind's vectorized observation for the current player; pad/trim if needed
        try:
            pov = obs["player_observations"][seat]
            obs_vec = np.asarray(pov.get("vectorized", []), dtype=np.float32)
            if obs_vec.ndim != 1 or obs_vec.size == 0:
                obs_vec = np.zeros((self.observation_space["obs"].shape[0],), dtype=np.float32)
            target = self.observation_space["obs"].shape[0]
            if obs_vec.size != target:
                if obs_vec.size > target:
                    obs_vec = obs_vec[:target]
                else:
                    pad = np.zeros((target,), dtype=np.float32)
                    pad[:obs_vec.size] = obs_vec
                    obs_vec = pad
        except Exception:
            obs_vec = np.zeros((self.observation_space["obs"].shape[0],), dtype=np.float32)

        # Legal mask from rl_env legal moves
        legal_mask = np.zeros((self.num_moves,), dtype=np.float32)
        try:
            for a in self._legal_moves_from_obs(obs):
                gid = self._id_from_rl_action(a)
                if 0 <= gid < self.num_moves:
                    legal_mask[gid] = 1.0
        except Exception:
            pass

        # Previous action by the opponent (already sentinel-initialized on first move)
        prev_other = self.prev_action[(seat + 1) % self.players]

        return {
            "obs": obs_vec,
            "legal_mask": legal_mask,
            "seat": int(seat),
            "prev_other_action": int(prev_other),
        }
