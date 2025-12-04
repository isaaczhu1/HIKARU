from __future__ import annotations

import torch

from hanabi_gru_baseline.model import HanabiGRUPolicy


def test_policy_forward_shapes_without_prev_self():
    batch, T, obs_dim, num_moves = 3, 4, 5, 7
    net = HanabiGRUPolicy(
        obs_dim=obs_dim,
        num_moves=num_moves,
        hidden=8,
        action_emb_dim=4,
        seat_emb_dim=2,
        include_prev_self=False,
    )
    obs = torch.randn(batch, T, obs_dim)
    seat = torch.zeros(batch, T, dtype=torch.long)
    prev_other = torch.zeros(batch, T, dtype=torch.long)
    h0 = net.initial_state(batch)
    logits, value, h_n = net(obs, seat, prev_other, h0)
    assert logits.shape == (batch, T, num_moves)
    assert value.shape == (batch, T)
    assert h_n.shape == (1, batch, net.hidden)


def test_policy_forward_with_prev_self():
    batch, T, obs_dim, num_moves = 2, 3, 6, 5
    net = HanabiGRUPolicy(
        obs_dim=obs_dim,
        num_moves=num_moves,
        hidden=10,
        action_emb_dim=3,
        seat_emb_dim=2,
        include_prev_self=True,
    )
    obs = torch.randn(batch, T, obs_dim)
    seat = torch.ones(batch, T, dtype=torch.long)
    prev_other = torch.zeros(batch, T, dtype=torch.long)
    prev_self = torch.full((batch, T), fill_value=num_moves, dtype=torch.long)
    h0 = net.initial_state(batch)
    logits, value, h_n = net(obs, seat, prev_other, h0, prev_self=prev_self)
    assert logits.shape == (batch, T, num_moves)
    assert value.shape == (batch, T)
    assert h_n.shape == (1, batch, net.hidden)
