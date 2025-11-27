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
        Expected keys (same as before): obs, legal, seat, prev_other, action, logp, value, reward, done
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
        """
        B = self.T * self.N

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

        idx = torch.randperm(B, device=self.dev)
        for i in range(0, B, batch_size):
            j = idx[i:i + batch_size]
            yield dict(
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

    # ---------------------- Recurrent (sequence) API ------------------------- #
    def iter_sequence_minibatches(self, seq_len: int, batch_size: int):
        """
        Yield sequence minibatches for recurrent PPO / BPTT.
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
        Notes:
          * We drop the final partial chunk if T % L != 0 (keeps shapes simple).
          * If no hidden was stored, h0 is returned as zeros.
        """
        T, N, L = self.T, self.N, int(seq_len)
        assert L >= 1, "seq_len must be >= 1"

        # number of full sequences per env
        S_per_env = T // L
        if S_per_env == 0:
            return  # nothing to yield

        # Build index list of (t0, n) for each sequence
        starts_t = torch.arange(0, S_per_env * L, L, device=self.dev)  # [S_per_env]
        env_ids  = torch.arange(N, device=self.dev)                    # [N]
        # Mesh to all (t0, n) pairs
        t0_grid  = starts_t.view(-1, 1).repeat(1, N)                   # [S_per_env, N]
        n_grid   = env_ids.view(1, -1).repeat(S_per_env, 1)            # [S_per_env, N]

        # Flatten to [S] where S = S_per_env * N
        t0_flat = t0_grid.reshape(-1)
        n_flat  = n_grid.reshape(-1)
        S = t0_flat.numel()

        # Shuffle sequence order
        perm = torch.randperm(S, device=self.dev)
        t0_flat = t0_flat[perm]
        n_flat  = n_flat[perm]

        # Helper to gather a [S, L, ...] block from [T, N, ...]
        def gather_seq(x):
            # x: [T, N, ...]
            # We index per sequence with (t0 + k, n)
            k = torch.arange(L, device=self.dev).view(1, L, 1)                     # [1, L, 1]
            t_idx = t0_flat.view(S, 1, 1) + k                                      # [S, L, 1]
            n_idx = n_flat.view(S, 1, 1).expand_as(t_idx)                          # [S, L, 1]
            # Advanced indexing
            return x[t_idx, n_idx].squeeze(-2)                                     # [S, L, ...]

        obs_seq   = gather_seq(self.obs)
        legal_seq = gather_seq(self.legal)
        seat_seq  = gather_seq(self.seat)
        prev_seq  = gather_seq(self.prev_other)
        act_seq   = gather_seq(self.act)
        logp_seq  = gather_seq(self.logp)
        val_seq   = gather_seq(self.val)
        adv_seq   = gather_seq(self.adv)
        ret_seq   = gather_seq(self.ret)

        # Initial hidden for each sequence = hidden stored at (t0, n)
        if self.h is not None:
            h0 = self.h[t0_flat, n_flat]                                           # [S, H]
        else:
            # No hidden recorded: provide zeros
            H = 1
            # Try to infer H from obs vs model later; for now set 1 and let caller expand/ignore.
            # Better: let ppo_update detect missing h and create zeros of model.hidden size.
            h0 = torch.zeros(S, 1, device=self.dev, dtype=torch.float32)

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
