# envs.py
# -----------------------------------------------------------------------------
# HanabiGym2P over DeepMind rl_env.HanabiEnv (2 players), observation-driven.
#
# This version:
# - Does NOT require env.state or env.game. It reads everything from the
#   rl_env observation dicts returned by reset()/step().
# - Legal moves come from obs["player_observations"][seat]["legal_moves"].
# - Fixed action id template: [PLAY 0..H-1][DISCARD 0..H-1][REVEAL_COLOR 0..C-1][REVEAL_RANK 0..R-1].
# - "obs" feature vector: if a pyhanabi state isn’t exposed, we emit zeros
#   (so training runs end-to-end). If you want rich features, we can later
#   swap in an encoder built from the rl_env observation dict.
#
# Observation dict (to the agent):
#   {
#     "obs": float32[obs_dim],          # zeros if pyhanabi state isn't available
#     "legal_mask": float32[num_moves], # 0/1 mask over global action ids
#     "seat": int {0,1},                # current acting seat
#     "prev_other_action": int in [0..num_moves]  # num_moves = sentinel "none"
#   }
#
# Reward: +1 only when score increases (successful play).
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from hanabi_learning_environment import rl_env

# Debug prints (export HANABI_DEBUG=1 to enable)
_DEBUG = bool(int(os.environ.get("HANABI_DEBUG", "0")))

# ---------------------------- Env wrapper ---------------------------------- #
class HanabiGym2P(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        colors: int = 5,
        ranks: int = 5,
        players: int = 2,
        hand_size: int = 5,
        seed: int = 0,
        obs_conf: str = "minimal",
        reward_mode: str = "play_plus_one",
        include_touched_flags: bool = True,
    ):
        assert players == 2, "This baseline supports exactly 2 players."

        self.colors = colors
        self.ranks = ranks
        self.players = players
        self.hand_size = hand_size
        self.seed = seed
        self.obs_conf = obs_conf
        self.reward_mode = reward_mode

        # Fixed global action count
        self.num_moves = 2 * hand_size + colors + ranks

        # Build rl_env (handles CHANCE internally)
        self.env = rl_env.HanabiEnv({
            "colors": colors,
            "ranks": ranks,
            "players": players,
            "hand_size": hand_size,
            "max_information_tokens": 8,
            "max_life_tokens": 3,
            # minimal vs card_knowledge doesn't change rl_env outputs we use here
            "observation_type": 0,  # rl_env uses its own obs dict anyway
            "seed": seed,
        })

        # Trackers
        self.prev_action = [-1 for _ in range(players)]  # last action by seat
        self.turn_count = 0
        self._last_obs = None      # last rl_env observation dict
        self._last_score = 0       # to shape +1 on successful play

        # Observation space: use a small placeholder vector for now (221 is typical obs_dim; set 1 to be conservative)
        obs_dim = 1
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
                "legal_mask": spaces.Box(0.0, 1.0, (self.num_moves,), dtype=np.float32),
                "seat": spaces.Discrete(players),
                "prev_other_action": spaces.Discrete(self.num_moves + 1),
            }
        )
        self.action_space = spaces.Discrete(self.num_moves)

        # Touch flags (kept for interface; not used in this minimal rl_env path)
        self._touched = [np.zeros((hand_size, 2), dtype=np.float32) for _ in range(players)] if include_touched_flags else None

        # Produce an initial obs to validate wiring
        _ = self._reset_world(seed)

    # ----------------------------- Utilities -------------------------------- #
    @staticmethod
    def _seat_from_obs(obs: dict) -> int:
        return int(obs.get("current_player", 0))

    def _score_from_obs(self, obs: dict) -> int:
        # rl_env obs does not always carry score directly; we can derive from fireworks stacks in player obs
        # Use the current player's view for simplicity:
        p = self._seat_from_obs(obs)
        pov = obs["player_observations"][p]
        # fireworks is dict color->highest rank played (0..4). Score is sum of (rank+1) across colors.
        fw = pov.get("fireworks", {})
        return int(sum(int(v) + 1 for v in fw.values()))

    def _legal_moves_from_obs(self, obs: dict):
        p = self._seat_from_obs(obs)
        return list(obs["player_observations"][p].get("legal_moves", []))

    # ID layout helpers
    def _id_for_play(self, idx: int) -> int: return idx
    def _id_for_discard(self, idx: int) -> int: return self.hand_size + idx
    def _id_for_reveal_color(self, color: int) -> int: return 2 * self.hand_size + color
    def _id_for_reveal_rank(self, rank: int) -> int: return 2 * self.hand_size + self.colors + rank

    def _parse_color(self, c) -> int:
        """Map rl_env color payloads ('R','Y','G','B','W' or names/ints) -> 0..colors-1."""
        if isinstance(c, int):
            return max(0, min(c, self.colors - 1))
        s = str(c).strip().lower()
        # single-letter codes
        letter_map = {"r": 0, "y": 1, "g": 2, "b": 3, "w": 4}
        if s in letter_map:
            return max(0, min(letter_map[s], self.colors - 1))
        # full names (just in case)
        name_map = {"red": 0, "yellow": 1, "green": 2, "blue": 3, "white": 4}
        if s in name_map:
            return max(0, min(name_map[s], self.colors - 1))
        return 0  # safe fallback

    def _parse_rank(self, r) -> int:
        """Map rl_env rank payloads ('1'..'5' or ints 0..4/1..5) -> 0..ranks-1."""
        if isinstance(r, int):
            # Some builds already use 0-based; some 1-based. Clamp either way.
            if 0 <= r < self.ranks:
                return r
            if 1 <= r <= self.ranks:
                return r - 1
            return max(0, min(r, self.ranks - 1))
        s = str(r).strip()
        if s.isdigit():
            v = int(s)
            # Assume 1-based in strings
            return max(0, min(v - 1, self.ranks - 1))
        return 0


    def _id_from_rl_action(self, a: dict) -> int:
        at = a.get("action_type", "")
        ats = str(at).upper()
        if "PLAY" in ats:
            return self._id_for_play(int(a.get("card_index", 0)))
        if "DISCARD" in ats:
            return self._id_for_discard(int(a.get("card_index", 0)))
        if "REVEAL" in ats and "COLOR" in ats:
            c = self._parse_color(a.get("color", 0))
            return self._id_for_reveal_color(c)
        if "REVEAL" in ats and "RANK" in ats:
            r = self._parse_rank(a.get("rank", 0))
            return self._id_for_reveal_rank(r)
        # fallback
        return 0


    def _rl_action_matches_id(self, a: dict, gid: int) -> bool:
        H, C, R = self.hand_size, self.colors, self.ranks
        # Validate bucket + index
        if 0 <= gid < H:  # PLAY
            return "card_index" in a and self._id_from_rl_action(a) == gid
        if H <= gid < 2 * H:  # DISCARD
            return "card_index" in a and self._id_from_rl_action(a) == gid
        if 2 * H <= gid < 2 * H + C:  # REVEAL_COLOR
            return "color" in a and self._id_from_rl_action(a) == gid
        if 2 * H + C <= gid < 2 * H + C + R:  # REVEAL_RANK
            return "rank" in a and self._id_from_rl_action(a) == gid
        return False

    # ------------------------------- Gym API -------------------------------- #
    def _reset_world(self, seed=None):
        # Some rl_env builds accept seed=..., others don't.
        try:
            obs = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        except TypeError:
            obs = self.env.reset()

        self._last_obs = obs
        self._last_score = self._score_from_obs(obs)
        self.prev_action = [-1 for _ in range(self.players)]
        self.turn_count = 0
        if self._touched is not None:
            for s in range(self.players):
                self._touched[s].fill(0.0)

        # --- NEW: infer obs_dim from rl_env vectorized features and update space once
        try:
            seat = self._seat_from_obs(obs)
            pov = obs["player_observations"][seat]
            vec = np.asarray(pov.get("vectorized", []), dtype=np.float32)
            if vec.ndim == 1 and vec.size > 0:
                current_dim = self.observation_space["obs"].shape[0]
                if current_dim != vec.size:
                    # Update obs space to the real feature size
                    self.observation_space["obs"] = gym.spaces.Box(
                        -np.inf, np.inf, (vec.size,), dtype=np.float32
                    )
        except Exception:
            pass

        return obs



    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs = self._reset_world(seed)
        return self._pack_obs(obs), {}

    def step(self, a_id: int):
        # Build legal mask from last obs; choose matching action dict
        legal = self._legal_moves_from_obs(self._last_obs)
        if not legal:
            # Unexpected with rl_env; do a defensive reset
            obs = self._reset_world()
            return self._pack_obs(obs), 0.0, False, False, {}

        choice = None
        for a in legal:
            if self._rl_action_matches_id(a, a_id):
                choice = a
                break
        if choice is None:
            choice = legal[0]

        # Shaped reward: +1 per score increase
        score_before = self._last_score

        next_obs, env_rew, done, info = self.env.step(choice)
        self._last_obs = next_obs
        score_after = self._score_from_obs(next_obs)
        self._last_score = score_after

        shaped = float(max(0, score_after - score_before)) if self.reward_mode == "play_plus_one" else float(env_rew)

        # prev action bookkeeping
        seat_now = self._seat_from_obs(next_obs)
        just_acted = (seat_now - 1) % self.players
        if 0 <= just_acted < self.players:
            self.prev_action[just_acted] = int(a_id)

        self.turn_count += 1
        terminated = bool(done)
        truncated = False
        return self._pack_obs(next_obs), shaped, terminated, truncated, (info or {})

    # ------------------------------ Packing --------------------------------- #
    def _pack_obs(self, obs: dict) -> dict:
        seat = self._seat_from_obs(obs)

        # --- Use DeepMind's full vectorized observation for the current player
        try:
            pov = obs["player_observations"][seat]
            obs_vec = np.asarray(pov.get("vectorized", []), dtype=np.float32)
            if obs_vec.ndim != 1 or obs_vec.size == 0:
                # fallback to zeros if not present
                obs_vec = np.zeros((self.observation_space["obs"].shape[0],), dtype=np.float32)
            # If rl_env suddenly changes size mid-run (shouldn't), pad/trim safely
            target = self.observation_space["obs"].shape[0]
            if obs_vec.size != target:
                if obs_vec.size > target:
                    obs_vec = obs_vec[:target]
                else:
                    pad = np.zeros((target,), dtype=np.float32)
                    pad[:obs_vec.size] = obs_vec
                    obs_vec = pad
        except Exception:
            # very defensive fallback
            obs_vec = np.zeros((self.observation_space["obs"].shape[0],), dtype=np.float32)

        # --- Legal mask from rl_env legal moves
        legal_mask = np.zeros((self.num_moves,), dtype=np.float32)
        try:
            for a in self._legal_moves_from_obs(obs):
                gid = self._id_from_rl_action(a)
                if 0 <= gid < self.num_moves:
                    legal_mask[gid] = 1.0
        except Exception:
            pass

        # --- Previous action by the opponent
        prev_other = self.prev_action[(seat + 1) % self.players]
        if prev_other < 0:
            prev_other = self.num_moves

        return {
            "obs": obs_vec,
            "legal_mask": legal_mask,
            "seat": int(seat),
            "prev_other_action": int(prev_other),
        }
