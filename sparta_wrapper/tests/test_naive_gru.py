from __future__ import annotations

from pathlib import Path

import pytest

from sparta_wrapper.naive_gru_blueprint import NaiveGRUBlueprint, GRU_CFG


def test_init_naive_gru_from_checkpoint():
    repo_root = Path(__file__).resolve().parents[2]
    ckpt_path = repo_root / "gru_checkpoints" / "ckpt_020000.pt"
    if not ckpt_path.is_file():
        pytest.skip(f"checkpoint not found at {ckpt_path}")

    cfg = GRU_CFG()
    blueprint = NaiveGRUBlueprint(cfg, ckpt_path, device="cpu")

    assert blueprint.net is not None
    assert blueprint.num_moves > 0
    h0 = blueprint.net.initial_state(batch=1, device=blueprint.device)
    assert h0.shape[0] == 1 and h0.shape[1] == 1
