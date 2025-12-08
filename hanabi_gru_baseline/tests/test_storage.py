from __future__ import annotations

import torch

from hanabi_gru_baseline.storage import RolloutStorage


def _make_storage(T: int, done_indices=None):
    done_indices = set(done_indices or [])
    N = 1
    obs_dim = 1
    num_actions = 1
    dev = torch.device("cpu")
    storage = RolloutStorage(T=T, N=N, obs_dim=obs_dim, num_actions=num_actions, device=dev)

    # Fill tensors with simple, distinguishable values
    storage.obs = torch.arange(T, device=dev, dtype=torch.float32).view(T, N, obs_dim)
    storage.legal = torch.ones(T, N, num_actions, device=dev, dtype=torch.float32)
    storage.seat = torch.zeros(T, N, device=dev, dtype=torch.long)
    storage.prev_other = torch.zeros(T, N, device=dev, dtype=torch.long)
    storage.act = torch.zeros(T, N, device=dev, dtype=torch.long)
    storage.logp = torch.zeros(T, N, device=dev, dtype=torch.float32)
    storage.val = torch.zeros(T, N, device=dev, dtype=torch.float32)
    storage.adv = torch.zeros(T, N, device=dev, dtype=torch.float32)
    storage.ret = torch.zeros(T, N, device=dev, dtype=torch.float32)
    storage.done = torch.zeros(T, N, device=dev, dtype=torch.float32)
    for idx in done_indices:
        storage.done[idx, 0] = 1.0
    return storage


def _collect_window_obs(storage: RolloutStorage, seq_len: int):
    windows = []
    for mb in storage.iter_sequence_minibatches(seq_len=seq_len, batch_size=1024):
        for seq in mb["obs"]:
            windows.append(seq.squeeze(-1).cpu().tolist())
    return windows


def test_sliding_windows_no_done():
    storage = _make_storage(T=6)
    windows = sorted(_collect_window_obs(storage, seq_len=4))
    assert windows == sorted([
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
    ])


def test_sliding_windows_respect_done_boundary():
    # done at index 3 splits the sequence into [0,1,2] and [4,5,6]
    storage = _make_storage(T=7, done_indices={3})
    windows = sorted(_collect_window_obs(storage, seq_len=3))
    assert windows == sorted([
        [0.0, 1.0, 2.0],
        [4.0, 5.0, 6.0],
    ])
