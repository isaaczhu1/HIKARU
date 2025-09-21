# utils.py
# Small helpers for seeding, checkpointing, and misc runtime niceties.

from __future__ import annotations
import os
import random
import json
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Set Python, NumPy, and PyTorch seeds. Leave cuDNN nondeterministic off by default
    (determinism can slow training a lot and isn't necessary for this baseline).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reasonable defaults for performance on CPU/GPU:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def save_ckpt(
    path: str,
    model_state: Dict[str, Any],
    optim_state: Dict[str, Any],
    update: int,
    cfg: Dict[str, Any] | None = None,
) -> None:
    """
    Save a checkpoint. We store model/optimizer states with torch.save,
    but also emit a small JSON sidecar with lightweight metadata.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model_state,
        "optim": optim_state,
        "update": int(update),
        "cfg": cfg if cfg is not None else {},
    }
    torch.save(payload, path)

    # Optional human-readable sidecar (without tensors)
    try:
        meta = {"update": int(update), "has_cfg": cfg is not None}
        with open(path + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass


def load_ckpt(path: str) -> Dict[str, Any]:
    """
    Load a checkpoint saved by save_ckpt.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def to_device(x: Any, device: torch.device | str) -> Any:
    """
    Convenience: move tensors or nested structures (lists/tuples/dicts) to device.
    Not used heavily in Phase 1, but handy if you expand the code.
    """
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [to_device(v, device) for v in x]
        return type(x)(t)
    return x


__all__ = ["seed_everything", "save_ckpt", "load_ckpt", "to_device"]
