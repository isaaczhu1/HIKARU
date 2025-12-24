# hanabi_envs.py
# -----------------------------------------------------------------------------
# Gym-free wrappers around DeepMind's rl_env.HanabiEnv plus a simple vector env.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from hanabi_learning_environment import rl_env

_DEBUG = bool(int(os.environ.get("HANABI_DEBUG", "0")))


class HanabiEnv2P:
    """
    Canonical single-environment wrapper for 2-player Hanabi (DeepMind HLE).

    API:
      reset(seed=None) -> obs_dict
      step(action_id) -> obs_dict, reward, done, info

    Observations (np arrays / ints):
      - "obs": float32[obs_dim]    (vectorized observation for current player)
      - "legal_mask": float32[num_moves] with 1.0 for legal ids, 0 otherwise
      - "seat": int {0,1}          (current player)
      - "prev_other_action": int in [0..num_moves] where num_moves is sentinel
    """

    def __init__(
        self,
        seed: int,
        *,
        players: int = 2,
        colors: int = 5,
        ranks: int = 5,
        hand_size: int = 5,
        max_information_tokens: int = 8,
        max_life_tokens: int = 3,
        random_start_player: bool = False,
    ):
        self.players = players
        self.colors = colors
        self.ranks = ranks
        self.hand_size = hand_size

        # Underlying DeepMind rl_env
        self._env = rl_env.HanabiEnv(
            config={
                "players": players,
                "colors": colors,
                "ranks": ranks,
                "hand_size": hand_size,
                "max_information_tokens": max_information_tokens,
                "max_life_tokens": max_life_tokens,
                "random_start_player": random_start_player,
                "seed": seed,
            }
        )

        # Probe vectorized obs length once to set a stable obs_dim
        self.obs_dim = 1
        try:
            _probe = self._env.reset()
            _seat = int(_probe.get("current_player", 0))
            _pov = _probe["player_observations"][_seat]
            _vec = np.asarray(_pov.get("vectorized", []), dtype=np.float32)
            if _vec.ndim == 1 and _vec.size > 0:
                self.obs_dim = int(_vec.size)
        except Exception:
            pass

        # Action layout
        self._a_play0 = 0
        self._a_discard0 = self._a_play0 + hand_size
        self._a_reveal_color0 = self._a_discard0 + hand_size
        self._a_reveal_rank0 = self._a_reveal_color0 + colors

        self.num_moves = 2 * hand_size + colors + ranks
        self.sentinel_none = self.num_moves  # sentinel for "no previous action"
        self._hint_target_offset = 1 if players > 1 else 0

        # Runtime state
        self._last_obs = None
        self._score_accum = 0.0
        self.prev_action = [self.sentinel_none for _ in range(self.players)]
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
            return self._a_reveal_color0 + self._parse_color(a.get("color", 0))
        if t == "REVEAL_RANK":
            rank1 = int(a.get("rank", 1))
            return self._a_reveal_rank0 + rank1
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
            "rank": (gid - self._a_reveal_rank0),
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

    # ------------------------------- API ------------------------------------ #
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

    def reset(self, *, seed: int | None = None):
        obs = self._reset_world(seed)
        return self._pack_obs(obs)

    def step(self, a_id: int):
        legal = self._legal_moves_from_obs(self._last_obs)
        if not legal:
            # Defensive reset (should be rare)
            obs = self._reset_world()
            info = {"score": float(self._score_accum)}
            return self._pack_obs(obs), 0.0, False, info

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

        shaped = float(max(0.0, float(env_rew)))
        self._score_accum += shaped

        if info is None:
            info = {}
        if not isinstance(info, dict):
            info = dict(info)
        info["score"] = float(self._score_accum)

        seat_now = self._seat_from_obs(next_obs)
        just_acted = (seat_now - 1) % self.players
        if 0 <= just_acted < self.players:
            self.prev_action[just_acted] = int(a_id)

        self.turn_count += 1
        terminated = bool(done)
        return self._pack_obs(next_obs), shaped, terminated, info

    # ------------------------------ Packing --------------------------------- #
    def _pack_obs(self, obs: dict) -> dict:
        seat = self._seat_from_obs(obs)

        # Use DeepMind's vectorized observation for the current player; pad/trim if needed
        try:
            pov = obs["player_observations"][seat]
            obs_vec = np.asarray(pov.get("vectorized", []), dtype=np.float32)
            if obs_vec.ndim != 1 or obs_vec.size == 0:
                obs_vec = np.zeros((self.obs_dim,), dtype=np.float32)
            target = self.obs_dim
            if obs_vec.size != target:
                if obs_vec.size > target:
                    obs_vec = obs_vec[:target]
                else:
                    pad = np.zeros((target,), dtype=np.float32)
                    pad[:obs_vec.size] = obs_vec
                    obs_vec = pad
        except Exception:
            obs_vec = np.zeros((self.obs_dim,), dtype=np.float32)

        legal_mask = np.zeros((self.num_moves,), dtype=np.float32)
        try:
            for a in self._legal_moves_from_obs(obs):
                gid = self._id_from_rl_action(a)
                if 0 <= gid < self.num_moves:
                    legal_mask[gid] = 1.0
        except Exception:
            pass

        prev_other = self.prev_action[(seat + 1) % self.players]

        return {
            "obs": obs_vec.astype(np.float32, copy=False),
            "legal_mask": legal_mask.astype(np.float32, copy=False),
            "seat": int(seat),
            "prev_other_action": int(prev_other),
        }


def stack_obs_list(obs_list: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Stack a list of obs dicts into batched numpy arrays."""
    if not obs_list:
        raise ValueError("obs_list is empty")
    keys = obs_list[0].keys()
    batched: Dict[str, np.ndarray] = {}
    for k in keys:
        vals = [obs[k] for obs in obs_list]
        if isinstance(vals[0], np.ndarray):
            batched[k] = np.stack(vals, axis=0)
        else:
            batched[k] = np.asarray(vals)
    return batched


class HanabiVecEnvSync:
    """Simple synchronous vectorized env wrapper that owns multiple HanabiEnv2P."""

    def __init__(self, n_envs: int, seed0: int, hanabi_cfg) -> None:
        self.envs: List[HanabiEnv2P] = [
            HanabiEnv2P(
                seed=seed0 + i,
                players=hanabi_cfg.players,
                colors=hanabi_cfg.colors,
                ranks=hanabi_cfg.ranks,
                hand_size=hanabi_cfg.hand_size,
                max_information_tokens=hanabi_cfg.max_information_tokens,
                max_life_tokens=hanabi_cfg.max_life_tokens,
                random_start_player=hanabi_cfg.random_start_player,
            )
            for i in range(n_envs)
        ]

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def reset_all(self, seed0: int | None = None) -> Dict[str, np.ndarray]:
        obs_list = []
        for i, env in enumerate(self.envs):
            seed = None if seed0 is None else seed0 + i
            obs_list.append(env.reset(seed=seed))
        return stack_obs_list(obs_list)

    def step_all(
        self, actions: Sequence[int]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        obs_list: List[Dict[str, Any]] = []
        rew_list: List[float] = []
        done_list: List[bool] = []
        info_list: List[Dict[str, Any]] = []

        for env, a in zip(self.envs, actions):
            o, r, d, info = env.step(int(a))
            obs_list.append(o)
            rew_list.append(float(r))
            done_list.append(bool(d))
            info_list.append(info)

        batched_obs = stack_obs_list(obs_list)
        rewards = np.asarray(rew_list, dtype=np.float32)
        dones = np.asarray(done_list, dtype=bool)
        return batched_obs, rewards, dones, info_list

    def reset_indices(self, idxs: Iterable[int]) -> List[Dict[str, Any]]:
        fresh: List[Dict[str, Any]] = []
        for i in idxs:
            fresh.append(self.envs[int(i)].reset())
        return fresh


# Backwards-compatible alias for older name
HanabiGym2P = HanabiEnv2P

__all__ = ["HanabiEnv2P", "HanabiGym2P", "HanabiVecEnvSync", "stack_obs_list"]
