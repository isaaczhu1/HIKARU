# train.py
import torch, torch.nn.functional as F
from gymnasium.vector import AsyncVectorEnv
from envs import HanabiGym2P
from model import HanabiGRUPolicy
from utils import masked_categorical

def make_vec_env(n):
    def thunk(i):
        return lambda: HanabiGym2P(seed=1000+i, obs_conf="minimal")
    return AsyncVectorEnv([thunk(i) for i in range(n)])

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, T = 64, 128
    gamma, lam = 0.99, 0.95
    ent_coef, vf_coef, clip = 0.02, 0.5, 0.2
    lr, epochs, mb = 3e-4, 4, 2048

    env = make_vec_env(N)
    first, _ = env.reset()
    obs_dim = first["obs"].shape[-1]
    num_moves = first["legal_mask"].shape[-1]

    net = HanabiGRUPolicy(obs_dim, num_moves, hidden=256, include_prev_self=False).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # Hidden per seat
    h0 = net.initial_state(N, device=device)  # seat 0
    h1 = net.initial_state(N, device=device)  # seat 1

    # Rollout buffers (you’d wrap these in a class in storage.py)
    for update in range(10_000):
        bufs = []  # collect step dicts, then stack
        # cache the per-step hidden we used (for truncated BPTT you might detach every rollout)
        for t in range(T):
            o = env.call("noop")  # not real; below we use env.step/get; this is just structure
        # ---- Real loop: pull obs and step ----
            obs_np = first["obs"]; legal_np = first["legal_mask"]
            seat_np = first["seat"]; prev_other_np = first["prev_other_action"]

            obs = torch.from_numpy(obs_np).to(device).float()             # [N, obs_dim]
            legal = torch.from_numpy(legal_np).to(device).float()         # [N, A]
            seat = torch.from_numpy(seat_np).to(device).long()            # [N]
            prev_other = torch.from_numpy(prev_other_np).to(device).long()# [N]

            # pick the right hidden bank
            h = torch.where(seat.view(1, -1, 1)==0, h0, h1)
            # BUT GRU expects contiguous [1,B,H] per seat; easier: gather by seat mask
            idx0 = (seat==0).nonzero(as_tuple=False).squeeze(-1)
            idx1 = (seat==1).nonzero(as_tuple=False).squeeze(-1)

            # Prepare a [B,T=1,...] batch in original order
            # Forward all at once by building a h batch consistent with env order:
            h_batch = torch.zeros_like(h0)
            h_batch[:, :, :] = h0  # temp; for simplicity do two forwards below:

            # Simpler and clear: forward in one shot with full batch h selected seatwise
            # Step as length-1 time sequence
            logits, value, h_new = net(
                obs_vec=obs.unsqueeze(1),
                seat=seat.unsqueeze(1),
                prev_other=prev_other.unsqueeze(1),
                h=torch.where(seat.view(1,-1,1)==0, h0, h1)
            )
            logits = logits.squeeze(1); value = value.squeeze(1)

            act, logp, dist = masked_categorical(logits, legal)
            # Step env
            first, rew, done, trunc, _ = env.step(act.detach().cpu().numpy())

            # Write back h to the correct bank
            h0[:, idx0, :] = h_new[:, idx0, :]
            h1[:, idx1, :] = h_new[:, idx1, :]

            # Reset hidden on done
            if done.any():
                done_mask = torch.from_numpy(done.astype(np.float32)).to(device)
                reset_idx = (done_mask > 0).nonzero(as_tuple=False).squeeze(-1)
                h0[:, reset_idx, :] = 0.0
                h1[:, reset_idx, :] = 0.0

            bufs.append(dict(obs=obs, legal=legal, seat=seat, prev_other=prev_other,
                             act=act, logp=logp.detach(), val=value.detach(),
                             rew=torch.from_numpy(rew).to(device).float(),
                             done=torch.from_numpy(done.astype(np.float32)).to(device)))

        # === compute GAE & returns (stack time first) ===
        # ... (standard GAE over T×N using bufs[t]['val'], bufs[t]['rew'], bufs[t]['done'])
        # === PPO update ===
        # Flatten, shuffle, minibatch; compute policy loss with clip=0.2, value loss, entropy=0.02
        # opt.step(), log stats, etc.
