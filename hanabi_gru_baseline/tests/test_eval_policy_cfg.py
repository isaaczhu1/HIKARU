from __future__ import annotations

from hanabi_gru_baseline.config import CFG
from hanabi_gru_baseline.eval_policy import _merge_cfg


def test_merge_cfg_overrides_nested_fields():
    base = CFG()
    cfg_dict = {
        "seed": 42,
        "model": {"hidden": 99, "action_emb": 7},
        "hanabi": {"colors": 3, "ranks": 4, "hand_size": 2},
    }
    merged = _merge_cfg(base, cfg_dict)
    assert merged.seed == 42
    assert merged.model.hidden == 99
    assert merged.model.action_emb == 7
    # untouched defaults still present
    assert merged.model.seat_emb == CFG.model.seat_emb
    assert merged.hanabi.colors == 3
    assert merged.hanabi.ranks == 4
    assert merged.hanabi.hand_size == 2
