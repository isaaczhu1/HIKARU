
from __future__ import annotations

import sys
from pathlib import Path

import torch

from hanabi_learning_environment import pyhanabi

# Ensure the repo root is importable so sibling package hanabi_gru_baseline can be used.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hanabi_gru_baseline.config import CFG as GRU_CFG
from hanabi_gru_baseline.model import HanabiGRUPolicy
from hanabi_gru_baseline.utils import load_ckpt

class NaiveGRUBlueprint:
    """
    A naive implementation of a GRU-based blueprint.
    Rolls out full history each time.
    """

    def __init__(self, model_config, model_ckpt_path, device: str | torch.device | None = None):
        self.config = model_config
        self.device = torch.device(device or "cpu")

        state = load_ckpt(model_ckpt_path)
        model_state = state["model"] if isinstance(state, dict) and "model" in state else state

        # Infer architecture from the checkpoint to avoid mismatch issues.
        obs_w = model_state["obs_fe.0.weight"]
        pi_w = model_state["pi.weight"]
        hidden = obs_w.shape[0]
        obs_dim = obs_w.shape[1]
        num_moves = pi_w.shape[0]
        action_emb_dim = model_state["prev_other_emb.weight"].shape[1]
        seat_emb_dim = model_state["seat_emb.weight"].shape[1]
        include_prev_self = "prev_self_emb.weight" in model_state

        # Allow config overrides for hidden sizes if provided; otherwise stick to checkpoint-derived.
        cfg_model = getattr(model_config, "model", None)
        if cfg_model is not None:
            hidden = getattr(cfg_model, "hidden", hidden)
            action_emb_dim = getattr(cfg_model, "action_emb", action_emb_dim)
            seat_emb_dim = getattr(cfg_model, "seat_emb", seat_emb_dim)
            include_prev_self = getattr(cfg_model, "include_prev_self", include_prev_self)

        self.net = HanabiGRUPolicy(
            obs_dim=obs_dim,
            num_moves=num_moves,
            hidden=hidden,
            action_emb_dim=action_emb_dim,
            seat_emb_dim=seat_emb_dim,
            include_prev_self=include_prev_self,
        ).to(self.device)
        self.net.load_state_dict(model_state)
        self.net.eval()

        self.obs_dim = obs_dim
        self.num_moves = num_moves
        self.hidden = hidden

        # Cache game-size metadata from config (used for action id mapping).
        hanabi_cfg = getattr(self.config, "hanabi", None)
        self._hand_size = getattr(hanabi_cfg, "hand_size", 5)
        self._colors = getattr(hanabi_cfg, "colors", 5)
        self._ranks = getattr(hanabi_cfg, "ranks", 5)
        self._sentinel_none = self.num_moves

        players = getattr(hanabi_cfg, "players", 2)
        self._hint_target_offset = 1 if players > 1 else 0

        # Recurrent state and optional prev-self tracking
        self._h = self.net.initial_state(batch=1, device=self.device)
        self._prev_self_id = self._sentinel_none

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, observation, *, prev_self_action=None):
        """Select an action for the current player given HanabiObservation."""
        # Build an encoder tied to the same game as the observation to avoid pointer mismatches.
        class _GameShim:
            def __init__(self, c_game):
                self._game = c_game

            @property
            def c_game(self):
                return self._game

        encoder = pyhanabi.ObservationEncoder(
            _GameShim(observation.raw_observation._game),
            pyhanabi.ObservationEncoderType.CANONICAL,
        )

        # Vectorize observation
        obs_vec = torch.tensor(
            encoder.encode(observation.raw_observation),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).unsqueeze(0)  # [1,1,obs_dim?]
        if obs_vec.shape[-1] != self.obs_dim:
            # Pad or trim to match training dimension
            target = self.obs_dim
            cur = obs_vec.shape[-1]
            if cur < target:
                pad = torch.zeros(1, 1, target, device=self.device, dtype=torch.float32)
                pad[..., :cur] = obs_vec
                obs_vec = pad
            else:
                obs_vec = obs_vec[..., :target]

        seat = torch.tensor([[observation.current_player]], device=self.device, dtype=torch.long)

        # Build legal mask and map legal moves to ids
        legal_moves = list(observation.legal_moves)
        legal_ids = [self._id_from_move(m) for m in legal_moves]
        legal_mask = torch.zeros(self.num_moves, device=self.device, dtype=torch.float32)
        for gid in legal_ids:
            if 0 <= gid < self.num_moves:
                legal_mask[gid] = 1.0
        legal_mask = legal_mask.view(1, 1, -1)

        # Previous other action: use last move from opponent if available
        prev_other_id = self._extract_prev_other_id(observation, legal_ids)
        prev_other = torch.tensor([[prev_other_id]], device=self.device, dtype=torch.long)

        # Optional previous self action embedding
        prev_self = None
        if getattr(self.net, "include_prev_self", False):
            prev_self_id = self._prev_self_id if prev_self_action is None else int(prev_self_action)
            prev_self = torch.tensor([[prev_self_id]], device=self.device, dtype=torch.long)

        logits, _, h_new = self.net(
            obs_vec=obs_vec,
            seat=seat,
            prev_other=prev_other,
            h=self._h,
            prev_self=prev_self,
        )
        # Detach GRU state to prevent graph accumulation during rollout.
        self._h = h_new.detach()
        logits = logits.squeeze(0)  # [1, num_moves]

        # Mask illegal actions
        very_neg = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(legal_mask.squeeze(0) < 0.5, very_neg)
        dist = torch.distributions.Categorical(logits=masked)
        action_id = int(dist.sample().item())

        # Track prev self for next step if applicable
        if getattr(self.net, "include_prev_self", False):
            self._prev_self_id = action_id

        # Map chosen id back to actual pyhanabi move
        move = self._move_from_id(action_id, legal_moves)
        return move

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _id_from_move(self, move: pyhanabi.HanabiMove) -> int:
        t = move.type()
        if t == pyhanabi.HanabiMoveType.PLAY:
            return move.card_index()
        if t == pyhanabi.HanabiMoveType.DISCARD:
            return self._hand_size + move.card_index()
        if t == pyhanabi.HanabiMoveType.REVEAL_COLOR:
            return 2 * self._hand_size + move.color()
        if t == pyhanabi.HanabiMoveType.REVEAL_RANK:
            return 2 * self._hand_size + self._colors + move.rank()
        return -1

    def _move_from_id(self, gid: int, legal_moves):
        for m in legal_moves:
            if self._id_from_move(m) == gid:
                return m
        # Fallback to first legal move if mapping failed
        return legal_moves[0]

    def _extract_prev_other_id(self, observation, legal_ids):
        # last_moves are ordered most recent first; find an opponent move
        for item in observation.last_moves:
            pid = item.get("player")
            move_dict = item.get("move")
            if pid is None or move_dict is None:
                continue
            if pid == observation.player_id:
                continue
            move = self._dict_to_move(move_dict)
            if move is None:
                continue
            gid = self._id_from_move(move)
            if gid >= 0:
                return gid
        return self._sentinel_none

    def _dict_to_move(self, move_dict):
        # Minimal converter using pyhanabi helpers
        action_type = move_dict.get("action_type")
        if action_type == "PLAY":
            return pyhanabi.HanabiMove.get_play_move(int(move_dict["card_index"]))
        if action_type == "DISCARD":
            return pyhanabi.HanabiMove.get_discard_move(int(move_dict["card_index"]))
        if action_type == "REVEAL_COLOR":
            color = move_dict["color"]
            if isinstance(color, str):
                color = pyhanabi.color_char_to_idx(color)
            return pyhanabi.HanabiMove.get_reveal_color_move(
                int(move_dict["target_offset"]), int(color)
            )
        if action_type == "REVEAL_RANK":
            return pyhanabi.HanabiMove.get_reveal_rank_move(
                int(move_dict["target_offset"]), int(move_dict["rank"])
            )
        if action_type == "DEAL":
            return None
        raise ValueError(f"Unsupported move dict: {move_dict}")


# Alias for naming consistency
HanabiGRUBlueprint = NaiveGRUBlueprint
