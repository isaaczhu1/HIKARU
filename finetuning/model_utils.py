"""Utilities for loading the Hanabi GRU policy for fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch

from hanabi_gru_baseline.model import HanabiGRUPolicy
from hanabi_gru_baseline.utils import load_ckpt


@dataclass
class ModelInfo:
    obs_dim: int
    num_moves: int
    hidden: int
    action_emb_dim: int
    seat_emb_dim: int
    include_prev_self: bool
    ckpt_path: Path


def infer_model_dims(state_dict: Dict[str, torch.Tensor]) -> ModelInfo:
    obs_dim = state_dict["obs_fe.0.weight"].shape[1]
    num_moves = state_dict["prev_other_emb.weight"].shape[0] - 1
    hidden = state_dict["gru.weight_hh_l0"].shape[1]
    action_emb_dim = state_dict["prev_other_emb.weight"].shape[1]
    seat_emb_dim = state_dict["seat_emb.weight"].shape[1]
    include_prev_self = "prev_self_emb.weight" in state_dict

    return ModelInfo(
        obs_dim=int(obs_dim),
        num_moves=int(num_moves),
        hidden=int(hidden),
        action_emb_dim=int(action_emb_dim),
        seat_emb_dim=int(seat_emb_dim),
        include_prev_self=bool(include_prev_self),
        ckpt_path=Path(),
    )


def load_model_from_ckpt(ckpt_path: str | Path, device: str | torch.device):
    """
    Load HanabiGRUPolicy weights from a PPO checkpoint.

    Returns (model, info).
    """
    ckpt_path = Path(ckpt_path)
    raw = load_ckpt(str(ckpt_path))
    state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

    info = infer_model_dims(state_dict)
    info.ckpt_path = ckpt_path

    net = HanabiGRUPolicy(
        obs_dim=info.obs_dim,
        num_moves=info.num_moves,
        hidden=info.hidden,
        action_emb_dim=info.action_emb_dim,
        seat_emb_dim=info.seat_emb_dim,
        include_prev_self=info.include_prev_self,
    ).to(device)
    net.load_state_dict(state_dict)
    net.train()

    return net, info
