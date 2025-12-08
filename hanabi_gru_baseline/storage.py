# storage.py

import numpy as np
import torch


class RolloutStorage:
    """
    Fixed-size rollout buffer for PPO with vectorized envs.
    Shapes: [T, N, ...]
    Stores: obs, legal, seat, prev_other, action, logp, value, reward, done, adv, ret
    Optional: pre-forward GRU hidden per step as h[t, n, H] if provided.
    """
    def __init__(self, T: int, N: int, obs_dim: int, num_actions: int, device: torch.device):
        self.T, self.N = T, N
        self.dev = device
        self.ptr = 0

        self.obs   = torch.zeros(T, N, obs_dim, device=device, dtype=torch.float32)
        self.legal = torch.zeros(T, N, num_actions, device=device, dtype=torch.float32)
        self.seat  = torch.zeros(T, N, device=device, dtype=torch.long)
        self.prev_other = torch.zeros(T, N, device=device, dtype=torch.long)

        self.act  = torch.zeros(T, N, device=device, dtype=torch.long)
        self.logp = torch.zeros(T, N, device=device, dtype=torch.float32)
        self.val  = torch.zeros(T, N, device=device, dtype=torch.float32)
        self.rew  = torch.zeros(T, N, device=device, dtype=torch.float32)
        self.done = torch.zeros(T, N, device=device, dtype=torch.float32)

        self.adv  = torch.zeros(T, N, device=device, dtype=torch.float32)
        self.ret  = torch.zeros(T, N, device=device, dtype=torch.float32)

        self.h = None
        self._h_size = None

    def reset_episode_slice(self):
        self.ptr = 0

    @torch.no_grad()
    def add(self, d: dict):
        t = self.ptr
        self.ptr += 1
        if not (0 <= t < self.T):
            raise RuntimeError(f"RolloutStorage overflow: t={t}, T={self.T}")

        self.obs[t].copy_(d["obs"])
        self.legal[t].copy_(d["legal"])
        self.seat[t].copy_(d["seat"])
        self.prev_other[t].copy_(d["prev_other"])

        self.act[t].copy_(d["action"])
        self.logp[t].copy_(d["logp"])
        self.val[t].copy_(d["value"])
        self.rew[t].copy_(d["reward"])
        self.done[t].copy_(d["done"])

        if "h" in d and d["h"] is not None:
            h_in = d["h"]
            if h_in.dim() == 3 and h_in.shape[0] == 1:
                h_in = h_in.squeeze(0)
            if h_in.dim() != 2 or h_in.shape[0] != self.N:
                raise ValueError(f'Expected "h" shape [N,H] or [1,N,H], got {tuple(h_in.shape)}')
            H = h_in.shape[1]
            if self.h is None:
                self.h = torch.zeros(self.T, self.N, H, device=self.dev, dtype=torch.float32)
                self._h_size = H
            elif self._h_size != H:
                raise ValueError(f'Hidden size changed within rollout: {self._h_size} -> {H}')
            self.h[t].copy_(h_in)

    @torch.no_grad()
    def compute_gae(self, gamma: float, lam: float, v_boot: torch.Tensor):
        T, N = self.T, self.N
        adv_next = torch.zeros(N, device=self.dev, dtype=torch.float32)
        for t in reversed(range(T)):
            next_val = v_boot if t == T - 1 else self.val[t + 1]
            nonterm = 1.0 - self.done[t]
            delta = self.rew[t] + gamma * next_val * nonterm - self.val[t]
            adv_next = delta + gamma * lam * nonterm * adv_next
            self.adv[t] = adv_next
        self.ret = self.adv + self.val
        return self.adv, self.ret

    def iter_minibatches(self, batch_size: int):
        T, N = self.T, self.N
        B = T * N

        def flat(x):
            return x.reshape(B, *x.shape[2:])

        flat_obs   = flat(self.obs)
        flat_legal = flat(self.legal)
        flat_seat  = flat(self.seat)
        flat_prev  = flat(self.prev_other)
        flat_act   = flat(self.act)
        flat_logp  = flat(self.logp)
        flat_val   = flat(self.val)
        flat_adv   = flat(self.adv)
        flat_ret   = flat(self.ret)

        flat_h = None
        if self.h is not None:
            flat_h = flat(self.h)

        idx = torch.randperm(B, device=self.dev)
        for i in range(0, B, batch_size):
            j = idx[i:i + batch_size]
            mb = dict(
                obs=flat_obs[j],
                legal=flat_legal[j],
                seat=flat_seat[j],
                prev_other=flat_prev[j],
                act=flat_act[j],
                logp_old=flat_logp[j],
                val_old=flat_val[j],
                adv=flat_adv[j],
                ret=flat_ret[j],
            )
            if flat_h is not None:
                mb["h0"] = flat_h[j]
            yield mb

    def iter_sequence_minibatches(self, seq_len: int, batch_size: int):
        T, N = self.T, self.N
        L = int(seq_len)
        assert L >= 1, "seq_len must be >= 1"

        has_h = self.h is not None

        seq_t_indices = []
        seq_env_ids   = []

        def _append_segment(segment: list, env_id: int):
            S_seg = len(segment)
            if S_seg >= L:
                for start in range(0, S_seg - L + 1):
                    t_seq = torch.tensor(
                        segment[start:start + L],
                        device=self.dev,
                        dtype=torch.long,
                    )
                    seq_t_indices.append(t_seq)
                    seq_env_ids.append(env_id)

        for n in range(N):
            seat_vals = torch.unique(self.seat[:, n])
            for s_val in seat_vals.tolist():
                mask = (self.seat[:, n] == s_val)
                idx_all = mask.nonzero(as_tuple=False).squeeze(-1)
                if idx_all.numel() == 0:
                    continue

                segment = []
                last_t = None
                for t in idx_all.tolist():
                    if not segment:
                        segment = [t]
                        last_t = t
                        continue

                    if self.done[last_t:t + 1, n].any():
                        _append_segment(segment, n)
                        segment = [t]
                    else:
                        segment.append(t)
                    last_t = t

                _append_segment(segment, n)

        S = len(seq_t_indices)
        if S == 0:
            return

        t_seq_mat = torch.stack(seq_t_indices, dim=0).to(self.dev)
        env_ids   = torch.tensor(seq_env_ids, device=self.dev, dtype=torch.long)

        perm = torch.randperm(S, device=self.dev)
        t_seq_mat = t_seq_mat[perm]
        env_ids   = env_ids[perm]

        def gather_seq(x):
            T_, N_ = x.shape[:2]
            rest = x.shape[2:]
            x_flat = x.view(T_, N_, -1)
            g = x_flat[t_seq_mat, env_ids.view(S, 1)]
            return g.view(S, L, *rest)

        obs_seq   = gather_seq(self.obs)
        legal_seq = gather_seq(self.legal)
        seat_seq  = gather_seq(self.seat)
        prev_seq  = gather_seq(self.prev_other)
        act_seq   = gather_seq(self.act)
        logp_seq  = gather_seq(self.logp)
        val_seq   = gather_seq(self.val)
        adv_seq   = gather_seq(self.adv)
        ret_seq   = gather_seq(self.ret)

        if has_h:
            t0 = t_seq_mat[:, 0]
            h0 = self.h[t0, env_ids, :]
        else:
            h0 = torch.zeros(S, 1, device=self.dev, dtype=torch.float32)

        for i in range(0, S, batch_size):
            sl = slice(i, min(i + batch_size, S))
            yield dict(
                obs=obs_seq[sl],
                legal=legal_seq[sl],
                seat=seat_seq[sl],
                prev_other=prev_seq[sl],
                act=act_seq[sl],
                logp_old=logp_seq[sl],
                val_old=val_seq[sl],
                adv=adv_seq[sl],
                ret=ret_seq[sl],
                h0=h0[sl],
            )
