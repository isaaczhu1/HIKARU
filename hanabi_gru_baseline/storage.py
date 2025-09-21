# storage.py
import numpy as np
import torch

class RolloutStorage:
    """
    Fixed-size rollout buffer for PPO with vectorized envs.
    Shapes: [T, N, ...]
    Stores: obs, legal, seat, prev_other, action, logp, value, reward, done, adv, ret
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

    def reset_episode_slice(self):
        """Start writing from t=0 again."""
        self.ptr = 0

    @torch.no_grad()
    def add(self, d: dict):
        """Append one time step worth of data for all N envs."""
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

    def iter_minibatches(self, batch_size: int):
        """
        Yields flattened minibatches of size ~batch_size (last may be smaller).
        Each field is shaped [B,...] where B = T*N.
        """
        B = self.T * self.N

        def flat(x):
            # Keep trailing dims (e.g., legal logits)
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
                val_old=flat_val[j],   # not strictly required but handy for diagnostics
                adv=flat_adv[j],
                ret=flat_ret[j],
            )

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