# storage.py

import numpy as np
import torch


class RolloutStorage:
    """
    Fixed-size rollout buffer for PPO with vectorized envs.
    Shapes: [T, N, ...]
    Stores: obs, legal, seat, prev_other, action, logp, value, reward, done, adv, ret
    Optional: pre-forward GRU hidden per step as h[t, n, H] if provided.

    Conventions:
      - obs[t, n] is the observation *before* taking action[t, n].
      - h[t, n] is the GRU hidden state for the acting seat in env n
        *before* forwarding obs[t, n] through the network (i.e., the h_in
        that produced logits/value/logp for that step).
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

        # Optional recurrent state buffer (allocated lazily on first add() that provides "h")
        self.h = None       # shape [T, N, H] when present
        self._h_size = None

    def reset_episode_slice(self):
        """Start writing from t=0 again."""
        self.ptr = 0

    @torch.no_grad()
    def add(self, d: dict):
        """
        Append one time step worth of data for all N envs.
        Expected keys: obs, legal, seat, prev_other, action, logp, value, reward, done.
        Optional key: "h" (pre-forward hidden for current step) with shape [N, H] or [1, N, H].
        """
        t = self.ptr
        self.ptr += 1
        if not (0 <= t < self.T):
            raise RuntimeError(f"RolloutStorage overflow: t={t}, T={self.T}")

        self.obs[t].copy_(d["obs"])                 # [N, obs_dim]
        self.legal[t].copy_(d["legal"])             # [N, A]
        self.seat[t].copy_(d["seat"])               # [N]
        self.prev_other[t].copy_(d["prev_other"])   # [N]

        self.act[t].copy_(d["action"])              # [N]
        self.logp[t].copy_(d["logp"])               # [N]
        self.val[t].copy_(d["value"])               # [N]
        self.rew[t].copy_(d["reward"])              # [N]
        self.done[t].copy_(d["done"])               # [N] in {0,1}

        # Optional recurrent hidden
        if "h" in d and d["h"] is not None:
            h_in = d["h"]
            if h_in.dim() == 3 and h_in.shape[0] == 1:
                # [1, N, H] -> [N, H]
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
        """
        Generalized Advantage Estimation (GAE-Lambda).
        v_boot: [N] value estimate at s_T (post-rollout state).
        """
        T, N = self.T, self.N
        adv_next = torch.zeros(N, device=self.dev, dtype=torch.float32)
        for t in reversed(range(T)):
            next_val = v_boot if t == T - 1 else self.val[t + 1]            # [N]
            nonterm = 1.0 - self.done[t]                                    # [N]
            delta = self.rew[t] + gamma * next_val * nonterm - self.val[t]  # [N]
            adv_next = delta + gamma * lam * nonterm * adv_next
            self.adv[t] = adv_next
        self.ret = self.adv + self.val
        return self.adv, self.ret

    # ---------------------- Feedforward (flattened) API ---------------------- #
    def iter_minibatches(self, batch_size: int):
        """
        Yields flattened minibatches of size ~batch_size (last may be smaller).
        Each field is shaped [B,...] where B = T*N.

        Returned dict keys:
          - obs:       [B, obs_dim]
          - legal:     [B, A]
          - seat:      [B]
          - prev_other:[B]
          - act:       [B]
          - logp_old:  [B]
          - val_old:   [B]
          - adv:       [B]
          - ret:       [B]
          - h0:        [B, H]  (if hidden was recorded; else omitted)
        """
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
            flat_h = flat(self.h)  # [B, H]

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
                mb["h0"] = flat_h[j]  # [B, H]
            yield mb

    # ---------------------- Recurrent (sequence) API ------------------------- #
    def iter_sequence_minibatches(self, seq_len: int, batch_size: int):
        """
        Yield sequence minibatches for recurrent PPO / BPTT, constructed per (env, seat).

        Each item is a dict with tensors shaped:
          - obs:        [B, L, obs_dim]
          - legal:      [B, L, A]
          - seat:       [B, L]
          - prev_other: [B, L]
          - act:        [B, L]
          - logp_old:   [B, L]
          - val_old:    [B, L]
          - adv:        [B, L]
          - ret:        [B, L]
          - h0:         [B, H]    (initial GRU hidden *before* the first step in each sequence)

        Construction:
          * For each env n and each seat s appearing in self.seat[:, n], collect the
            indices t where seat[t, n] == s. These indices define that seat's timeline.
          * Chunk that timeline into non-overlapping segments of length L = seq_len:
               idx_s = [t0, t1, ..., t_{K-1}]  → segments [t0..t_{L-1}], [t_L..t_{2L-1}], ...
          * Each such segment becomes one sequence. We gather obs/legal/etc. at those
            (t, n) pairs, and take h0 from self.h[t0, n] if available (else zeros).
        """
        T, N = self.T, self.N
        L = int(seq_len)
        assert L >= 1, "seq_len must be >= 1"

        # No hidden recorded: we can still build sequences, but h0 will be zeros
        has_h = self.h is not None
        H = self._h_size if has_h else 0

        # Collect all sequences across (env, seat)
        seq_t_indices = []  # list of [L] LongTensors (time indices)
        seq_env_ids   = []  # list of env ids (int)
        # seats will be gathered from self.seat later

        for n in range(N):
            # Unique seat values in this env's trajectory (usually {0,1})
            seat_vals = torch.unique(self.seat[:, n])
            for s_val in seat_vals.tolist():
                # Indices where this seat was acting in env n
                mask = (self.seat[:, n] == s_val)
                idx = mask.nonzero(as_tuple=False).squeeze(-1)  # [K] or []
                if idx.numel() < L:
                    continue
                # Non-overlapping chunks of length L along this seat's own timeline
                num_full = idx.numel() // L
                if num_full <= 0:
                    continue
                for j in range(num_full):
                    start = j * L
                    end = start + L
                    t_seq = idx[start:end]  # [L]
                    seq_t_indices.append(t_seq)
                    seq_env_ids.append(n)

        S = len(seq_t_indices)
        if S == 0:
            return  # nothing to yield

        # Stack into tensors on device
        t_seq_mat = torch.stack(seq_t_indices, dim=0).to(self.dev)  # [S, L]
        env_ids   = torch.tensor(seq_env_ids, device=self.dev, dtype=torch.long)  # [S]

        # Shuffle sequence order
        perm = torch.randperm(S, device=self.dev)
        t_seq_mat = t_seq_mat[perm]
        env_ids   = env_ids[perm]

        # Helper: gather [S, L, ...] from [T, N, ...] using (t_seq_mat, env_ids)
        def gather_seq(x):
            # x: [T, N, ...]
            # For each sequence s and position l, take x[t_seq_mat[s,l], env_ids[s]]
            S_, L_ = t_seq_mat.shape
            assert L_ == L
            t_idx = t_seq_mat.unsqueeze(-1)                        # [S, L, 1]
            n_idx = env_ids.view(S_, 1, 1).expand(S_, L_, 1)       # [S, L, 1]
            out = x[t_idx, n_idx]                                  # [S, L, ...]
            return out.squeeze(-2)                                 # [S, L, ...]

        obs_seq   = gather_seq(self.obs)
        legal_seq = gather_seq(self.legal)
        seat_seq  = gather_seq(self.seat)
        prev_seq  = gather_seq(self.prev_other)
        act_seq   = gather_seq(self.act)
        logp_seq  = gather_seq(self.logp)
        val_seq   = gather_seq(self.val)
        adv_seq   = gather_seq(self.adv)
        ret_seq   = gather_seq(self.ret)

        # Initial hidden for each sequence = hidden stored at (t0, n),
        # where t0 = first time index in that sequence for that seat.
        if has_h:
            t0 = t_seq_mat[:, 0]                     # [S]
            h0 = self.h[t0, env_ids, :]             # [S, H]
        else:
            # No hidden recorded: provide zeros; ppo_update will adjust if needed.
            h0 = torch.zeros(len(env_ids), 1, device=self.dev, dtype=torch.float32)

        # Now yield minibatches over sequences
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

    # ------------------------------ Diagnostics ------------------------------ #
    def last_ep_return_mean(self):
        """
        Quick-and-dirty estimate: sum rewards to done within this rollout window.
        If no episode finished, return the mean partial return so far.
        """
        ep_ret = []
        cur = torch.zeros(self.N, device=self.dev, dtype=torch.float32)
        for t in range(self.T):
            cur += self.rew[t]
            ended = self.done[t] > 0
            if ended.any():
                ep_ret.extend(cur[ended].tolist())
                cur[ended] = 0.0
        return float(np.mean(ep_ret)) if ep_ret else float(cur.mean().item())
