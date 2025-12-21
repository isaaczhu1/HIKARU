
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

from sparta_wrapper.hanabi_utils import _debug
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG, DEVICE, DEBUG


# -----------------------------------------------------------------------------
# Shared model cache: load each checkpoint/device pair once per process.
# -----------------------------------------------------------------------------
class _SharedModel:
    def __init__(self, net, obs_dim, num_moves, hidden_dim, hand_size, colors, ranks, hint_target_offset, device):
        self.net = net
        self.obs_dim = obs_dim
        self.num_moves = num_moves
        self.hidden_dim = hidden_dim
        self.hand_size = hand_size
        self.colors = colors
        self.ranks = ranks
        self.hint_target_offset = hint_target_offset
        self.device = device
    
    @torch.no_grad()    
    def forward(self, obs_vec, seat, prev_other, h, prev_self):
        """
        return the result of a forward pass
        """
        logits, _, h_new = self.net(
            obs_vec=obs_vec,
            seat=seat,
            prev_other=prev_other,
            h=h,
            prev_self=prev_self,
        )
        return logits, _, h_new
    
    def initial_state(self):
        return self.net.initial_state(batch=1, device=self.device)


_MODEL_CACHE: dict[tuple[str, str], _SharedModel] = {}


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Choose configured device, with safe CPU fallback if CUDA is unavailable."""
    target = torch.device(device or DEVICE)
    if target.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return target


def _get_shared_model(model_ckpt_path, model_config, device: torch.device) -> _SharedModel:
    """
    Loads the "_SharedModel" weights from the given model checkpoint path and device.
    """
    device = _resolve_device(device)
    key = (str(Path(model_ckpt_path).resolve()), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    state = load_ckpt(model_ckpt_path)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state

    # Infer architecture from the checkpoint to avoid mismatch issues.
    obs_w = model_state["obs_fe.0.weight"]
    pi_w = model_state["pi.weight"]
    hidden_dim = obs_w.shape[0]
    obs_dim = obs_w.shape[1]
    num_moves = pi_w.shape[0]
    action_emb_dim = model_state["prev_other_emb.weight"].shape[1]
    seat_emb_dim = model_state["seat_emb.weight"].shape[1]
    include_prev_self = "prev_self_emb.weight" in model_state

    # Allow config overrides for hidden sizes if provided; otherwise stick to checkpoint-derived.
    cfg_model = getattr(model_config, "model", None)
    if cfg_model is not None:
        hidden_dim = getattr(cfg_model, "hidden", hidden_dim)
        action_emb_dim = getattr(cfg_model, "action_emb", action_emb_dim)
        seat_emb_dim = getattr(cfg_model, "seat_emb", seat_emb_dim)
        include_prev_self = getattr(cfg_model, "include_prev_self", include_prev_self)

    net = HanabiGRUPolicy(
        obs_dim=obs_dim,
        num_moves=num_moves,
        hidden=hidden_dim,
        action_emb_dim=action_emb_dim,
        seat_emb_dim=seat_emb_dim,
        include_prev_self=include_prev_self,
    ).to(device)
    net.load_state_dict(model_state)
    net.eval()

    # Cache game-size metadata from config (used for action id mapping).
    hanabi_cfg = getattr(model_config, "hanabi", None)
    hand_size = getattr(hanabi_cfg, "hand_size", 5)
    colors = getattr(hanabi_cfg, "colors", 5)
    ranks = getattr(hanabi_cfg, "ranks", 5)
    players = getattr(hanabi_cfg, "players", 2)
    hint_target_offset = 1 if players > 1 else 0

    shared = _SharedModel(
        net=net,
        obs_dim=obs_dim,
        num_moves=num_moves,
        hidden_dim=hidden_dim,
        hand_size=hand_size,
        colors=colors,
        ranks=ranks,
        hint_target_offset=hint_target_offset,
        device=device
    )
    _MODEL_CACHE[key] = shared

    if DEBUG:
        _debug("Loaded shared model for GRU Blueprints")
    return shared