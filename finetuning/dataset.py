"""Torch Dataset and collate utilities for SPARTA fine-tuning."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class SpartaDataset(Dataset):
    """Hold per-step supervision samples produced by build_training_samples."""

    def __init__(self, samples: Sequence[Dict]):
        if not samples:
            raise ValueError("No samples provided for SpartaDataset")
        self.samples: List[Dict] = list(samples)

        # Validate shapes based on the first sample.
        first = self.samples[0]
        self.obs_dim = int(np.asarray(first["obs_vec"]).reshape(-1).shape[0])
        self.num_moves = int(np.asarray(first["legal_mask"]).reshape(-1).shape[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_sparta(batch: Sequence[Dict], include_prev_self: bool, sentinel: int):
    """Collate a batch of dict samples into torch tensors."""
    obs = torch.tensor([b["obs_vec"] for b in batch], dtype=torch.float32)
    seat = torch.tensor([b["seat"] for b in batch], dtype=torch.long)
    prev_other = torch.tensor([b["prev_other_action"] for b in batch], dtype=torch.long)
    legal = torch.tensor([b["legal_mask"] for b in batch], dtype=torch.float32)
    action_target = torch.tensor([b["action_target"] for b in batch], dtype=torch.long)
    value_target = torch.tensor([b["value_target"] for b in batch], dtype=torch.float32)

    is_override = [b.get("is_override") for b in batch]
    has_override = any(v is not None for v in is_override)
    if has_override:
        override_tensor = torch.tensor(
            [int(v) if v is not None else 0 for v in is_override], dtype=torch.float32
        )
    else:
        override_tensor = None

    prev_self = None
    if include_prev_self:
        prev_self = torch.full_like(prev_other, fill_value=int(sentinel), dtype=torch.long)

    return {
        "obs": obs,
        "seat": seat,
        "prev_other": prev_other,
        "prev_self": prev_self,
        "legal": legal,
        "action_target": action_target,
        "value_target": value_target,
        "is_override": override_tensor,
    }
