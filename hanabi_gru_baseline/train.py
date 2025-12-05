# train.py
# -----------------------------------------------------------------------------
# End-to-end PPO + GRU baseline trainer for 2-player Hanabi (DeepMind HLE).
#
# Wires together:
#   - envs.HanabiGym2P   (single-agent self-play over HLE)
#   - model.HanabiGRUPolicy (GRU(256) policy/value)
#   - storage.RolloutStorage (T x N GAE buffer)
#   - ppo.ppo_update     (masked PPO)
#   - utils              (seed + ckpt)
#
# Robustness:
#   * SyncVectorEnv by default on macOS (HLE + async/fork can hang).
#   * Auto-reset any env slots that are done OR have zero-legal before sampling.
#   * After env.step, reset done slots immediately so next loop sees valid legals.
#   * Never assert that legal>0; enforce with resets + a last-resort fallback.
#   * Coerce dtypes when splicing partial resets to avoid NumPy surprises.
# -----------------------------------------------------------------------------

import os
import sys
import time
import argparse
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

from config import CFG
from hanabi_envs import HanabiGym2P
from model import HanabiGRUPolicy
from storage import RolloutStorage
from ppo import ppo_update  # masked_categorical not needed (we use ε-greedy mix in-place)
from utils import seed_everything, save_ckpt, load_ckpt

from torch.utils.tensorboard import SummaryWriter

IS_DARWIN = sys.platform == "darwin"


# ----------------------------- Argument parsing ------------------------------ #
def parse_args():
    ap = argparse.ArgumentParser(description="PPO+GRU baseline for 2p Hanabi (HLE)")
    ap.add_argument("--device", type=str, default=None, help="cuda | cpu (auto if None)")
    ap.add_argument("--total-updates", type=int, default=None, help="override total updates")
    ap.add_argument("--lr", type=float, default=None, help="override learning rate")
    ap.add_argument("--ckpt", type=str, default=None, help="resume checkpoint path")
    ap.add_argument("--save-dir", type=str, default=None, help="override output dir")
    ap.add_argument("--debug", action="store_true", help="extra asserts/prints")
    ap.add_argument("--async-env", action="store_true",
                    help="use AsyncVectorEnv (default false on macOS for stability)")
    ap.add_argument("--variant", type=str, default="twoxtwo",
                    choices=["twoxtwo", "standard"],
                    help="Hanabi size preset.")
    return ap.parse_args()


# ----------------------------- Vec env factory ------------------------------- #
def make_vec_env(n_envs, seed0, obs_conf="minimal", use_async=None, hanabi_cfg=None):
    def thunk(i):
        return lambda: HanabiGym2P(
            seed=seed0 + i,
            obs_conf=obs_conf,
            players=hanabi_cfg.players,
            colors=hanabi_cfg.colors,
            ranks=hanabi_cfg.ranks,
            hand_size=hanabi_cfg.hand_size,
            max_information_tokens=hanabi_cfg.max_information_tokens,
            max_life_tokens=hanabi_cfg.max_life_tokens,
            random_start_player=hanabi_cfg.random_start_player,
        )

    if use_async is None:
        use_async = not IS_DARWIN
    Vec = AsyncVectorEnv if use_async else SyncVectorEnv
    return Vec([thunk(i) for i in range(n_envs)])


# ----------------------------- Env reset helper ------------------------------ #
def _reset_indices(env, idxs):
    """
    Reset a subset of vectorized envs and return fresh obs_dicts in the same slot order.
    """
    if not idxs:
        return []

    fresh = []
    if hasattr(env, "env_method"):  # AsyncVectorEnv
        results = env.env_method("reset", indices=idxs)
        for r in results:
            r0 = r[0] if isinstance(r, tuple) else r
            assert isinstance(r0, dict) and "obs" in r0, "reset() did not return an obs-dict"
            fresh.append(r0)
        return fresh

    if hasattr(env, "envs"):  # SyncVectorEnv
        for i in idxs:
            res = env.envs[i].reset()
            r0 = res[0] if isinstance(res, tuple) else res
            assert isinstance(r0, dict) and "obs" in r0, "reset() did not return an obs-dict"
            fresh.append(r0)
        return fresh

    # Fallback: reset all and pick the ones we need (rare)
    all_obs, _ = env.reset()
    return [all_obs[j] for j in idxs]


# ---------------------------- One environment step --------------------------- #
@torch.no_grad()
def do_step(env, net, device, h0, h1, obs_dict, debug=False, eps=0.0):
    """
    Execute a single time step for all N envs in the vector env.
    Auto-resets any slots with zero-legal before sampling, and post-step for done slots.
    """
    N = obs_dict["obs"].shape[0]

    # ----- Guard: zero-legal -> reset those slots BEFORE sampling -----
    legal_np = obs_dict["legal_mask"]
    sum_legal = legal_np.sum(axis=1)
    if np.any(sum_legal == 0):
        bad_idxs = np.nonzero(sum_legal == 0)[0].tolist()
        fresh = _reset_indices(env, bad_idxs)
        for slot, j in enumerate(bad_idxs):
            new_o = fresh[slot]
            obs_dict["obs"][j] = new_o["obs"].astype(np.float32, copy=False)
            obs_dict["legal_mask"][j] = new_o["legal_mask"].astype(np.float32, copy=False)
            obs_dict["seat"][j] = np.int64(new_o["seat"])
            obs_dict["prev_other_action"][j] = np.int64(new_o["prev_other_action"])
        legal_np = obs_dict["legal_mask"]

    # ----- Torchify current batch -----
    obs   = torch.from_numpy(obs_dict["obs"]).to(device).float()                # [N, obs_dim]
    legal = torch.from_numpy(legal_np).to(device).float()                        # [N, A]
    seat  = torch.from_numpy(obs_dict["seat"]).to(device).long()                 # [N]
    prev  = torch.from_numpy(obs_dict["prev_other_action"]).to(device).long()    # [N]

    # Last-resort safety: if any row is still zero-legal, make action 0 legal
    zero_rows = (legal.sum(dim=1) == 0)
    if zero_rows.any():
        legal[zero_rows, 0] = 1.0
        if debug:
            zr = zero_rows.nonzero(as_tuple=False).squeeze(-1).tolist()
            print(f"[warn] Forcing a legal action on rows {zr}", flush=True)

    # ----- Select per-seat hidden -----
    h_in = torch.where(seat.view(1, -1, 1) == 0, h0, h1)                         # [1,N,H]

    # ----- Forward (seq len 1) -----
    logits, value, h_new = net(
        obs_vec=obs.unsqueeze(1),        # [N,1,obs_dim]
        seat=seat.unsqueeze(1),          # [N,1]
        prev_other=prev.unsqueeze(1),    # [N,1]
        h=h_in                           # [1,N,H]
    )
    logits = logits.squeeze(1)           # [N, A]
    value  = value.squeeze(1)            # [N]

    # ----- Sample masked actions with ε-greedy mixture over legal -----
    probs = torch.softmax(logits, dim=-1)                  # [N, A]
    probs = probs * legal                                  # mask illegal
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    uniform = legal / legal.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mix = (1.0 - eps) * probs + eps * uniform              # [N, A]
    action = torch.multinomial(mix, num_samples=1).squeeze(1)  # [N]

    # Log-prob under the POLICY (not the mixture) for PPO ratios
    logp = torch.log(probs.gather(1, action.view(-1, 1)).squeeze(1).clamp_min(1e-8))

    # ----- Step env -----
    next_obs_dict, rew_np, done_np, trunc_np, info = env.step(action.detach().cpu().numpy())

    # Normalize env outputs defensively
    if not isinstance(rew_np, np.ndarray):
        rew_np = np.asarray(rew_np, dtype=np.float32)
    if isinstance(trunc_np, np.ndarray):
        done_or_trunc = np.logical_or(done_np, trunc_np)
    else:
        done_or_trunc = done_np
    if not isinstance(done_or_trunc, np.ndarray):
        done_or_trunc = np.asarray(done_or_trunc, dtype=bool)

    # ----- Write back updated hidden to acting seat bank -----
    idx0 = (seat == 0).nonzero(as_tuple=False).squeeze(-1)
    idx1 = (seat == 1).nonzero(as_tuple=False).squeeze(-1)
    if idx0.numel() > 0:
        h0[:, idx0, :] = h_new[:, idx0, :]
    if idx1.numel() > 0:
        h1[:, idx1, :] = h_new[:, idx1, :]

    # ----- Reset both seat hiddens for done slots -----
    if np.any(done_or_trunc):
        reset_ids = torch.from_numpy(done_or_trunc.astype(np.uint8)).to(device).nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            h0[:, reset_ids, :] = 0.0
            h1[:, reset_ids, :] = 0.0

    # ----- Immediately auto-reset done envs so next loop has valid legals -----
    if np.any(done_or_trunc):
        idxs = np.nonzero(done_or_trunc)[0].tolist()
        fresh = _reset_indices(env, idxs)
        for slot, j in enumerate(idxs):
            o = fresh[slot]
            next_obs_dict["obs"][j] = o["obs"].astype(np.float32, copy=False)
            next_obs_dict["legal_mask"][j] = o["legal_mask"].astype(np.float32, copy=False)
            next_obs_dict["seat"][j] = np.int64(o["seat"])
            next_obs_dict["prev_other_action"][j] = np.int64(o["prev_other_action"])

    # ----- Build storage record -----
    step_record = {
        "obs": obs,                              # [N, obs_dim] float
        "legal": legal,                          # [N, A] float (0/1)
        "seat": seat,                            # [N] long
        "prev_other": prev,                      # [N] long
        "action": action,                        # [N] long
        "logp": logp.detach(),                   # [N] float
        "value": value.detach(),                 # [N] float
        "reward": torch.from_numpy(rew_np).to(device).float(),               # [N]
        "done": torch.from_numpy(done_or_trunc.astype(np.float32)).to(device), # [N] {0,1}
        # New: store pre-forward hidden so storage can serve BPTT sequences later.
        "h": h_in.squeeze(0).detach(),           # [N, H]
    }

    epi = {"done": done_or_trunc, "reward": rew_np}
    return next_obs_dict, step_record, h0, h1, epi


# --------------------------------- Trainer ---------------------------------- #
def main(cfg: CFG, args):
    # ---------- Device & seeding ---------- #
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] using device: {device}")
    seed_everything(cfg.seed)

    out_dir = args.save_dir or cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Environments ---------- #
    env = make_vec_env(cfg.num_envs, cfg.seed, obs_conf=cfg.obs_mode,
                       use_async=args.async_env, hanabi_cfg=cfg.hanabi)
    obs_dict, _ = env.reset()  # dict of numpy arrays
    print("[debug] initial legal sums:", obs_dict["legal_mask"].sum(axis=1)[:8])

    # ---------- tensorboard logging ---------- #

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"tb/{out_dir}/{timestamp}")

    # Infer dimensions from first observation
    obs_dim   = obs_dict["obs"].shape[-1]
    num_moves = obs_dict["legal_mask"].shape[-1]
    N         = cfg.num_envs
    T         = cfg.unroll_T

    # ---------- Model & Optimizer ---------- #
    net = HanabiGRUPolicy(
        obs_dim=obs_dim,
        num_moves=num_moves,
        hidden=cfg.model.hidden,
        action_emb_dim=cfg.model.action_emb,
        seat_emb_dim=cfg.model.seat_emb,
        include_prev_self=cfg.model.include_prev_self
    ).to(device)

    lr = args.lr if args.lr is not None else cfg.ppo.lr
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # Per-seat GRU hidden banks: [1, N, H]
    h0 = net.initial_state(N, device=device)  # seat 0
    h1 = net.initial_state(N, device=device)  # seat 1

    # ---------- Rollout Storage ---------- #
    storage = RolloutStorage(T=T, N=N, obs_dim=obs_dim, num_actions=num_moves, device=device)

    # ---------- Resume (optional) ---------- #
    start_update = 0
    if args.ckpt:
        state = load_ckpt(args.ckpt)
        net.load_state_dict(state["model"])
        try:
            opt.load_state_dict(state["optim"])
        except Exception:
            print("[warn] optimizer state from ckpt not loaded (optimizer mismatch).")
        start_update = int(state.get("update", 0))
        print(f"[resume] loaded {args.ckpt} @ update {start_update}")

    # ---------- RL bookkeeping ---------- #
    total_updates = args.total_updates or cfg.total_updates
    global_env_steps = start_update * (N * T)
    ep_ret = np.zeros(N, dtype=np.float32)
    ep_len = np.zeros(N, dtype=np.int32)
    ret_hist = []  # rolling window of completed episode returns

    # --------------------------- Training loop --------------------------- #
    wall_t0 = time.time()
    for update in range(start_update, total_updates):
        # ---- LR decay ----
        progress = (update + 1) / total_updates
        lr_now = cfg.ppo.lr + (cfg.sched.lr_final - cfg.ppo.lr) * progress
        for g in opt.param_groups:
            g["lr"] = lr_now

        # ---- ε-greedy decay ----
        eps_decay = min(1.0, (update + 1) / max(1, cfg.sched.eps_decay_until))
        eps_now = cfg.sched.eps0 * (1.0 - eps_decay)  # goes to 0 by eps_decay_until

        # ---- Reset rollout buffer cursor ----
        storage.reset_episode_slice()

        # ---- Rollout collection: T steps ----
        with torch.no_grad():
            for t in range(T):
                obs_dict, step_rec, h0, h1, epi = do_step(
                    env=env, net=net, device=device,
                    h0=h0, h1=h1, obs_dict=obs_dict,
                    debug=args.debug, eps=eps_now
                )
                storage.add(step_rec)

                # Episodic logging bookkeeping
                ep_ret += epi["reward"]
                ep_len += 1
                if np.any(epi["done"]):
                    done_ids = np.where(epi["done"])[0]
                    ret_hist.extend(ep_ret[done_ids].tolist())
                    ep_ret[done_ids] = 0.0
                    ep_len[done_ids] = 0

                global_env_steps += N

            # ---- Bootstrap value at last state (for GAE tail) ----
            obs_t   = torch.from_numpy(obs_dict["obs"]).to(device).float()       # [N, obs_dim]
            seat_t  = torch.from_numpy(obs_dict["seat"]).to(device).long()       # [N]
            prev_t  = torch.from_numpy(obs_dict["prev_other_action"]).to(device).long()  # [N]
            h_in_T  = torch.where(seat_t.view(1, -1, 1) == 0, h0, h1)            # [1,N,H]
            _, v_boot, _ = net(
                obs_vec=obs_t.unsqueeze(1),
                seat=seat_t.unsqueeze(1),
                prev_other=prev_t.unsqueeze(1),
                h=h_in_T
            )
            v_boot = v_boot.squeeze(1)  # [N]

        # ---- Compute GAE/Returns on buffer ----
        storage.compute_gae(
            gamma=cfg.ppo.gamma,
            lam=cfg.ppo.gae_lambda,
            v_boot=v_boot
        )

        # ---- PPO update (masked) ----
        decay = min(1.0, (update + 1) / max(1, cfg.sched.ent_decay_until))
        ent_coef_now = cfg.ppo.ent_coef + (cfg.sched.ent_final - cfg.ppo.ent_coef) * decay
        logs = ppo_update(policy=net, optimizer=opt, storage=storage, cfg=cfg, ent_coef_override=ent_coef_now)

        vf_coef = cfg.ppo.vf_coef
        # Approximate mean total loss across minibatches, matching ppo.py:
        total_loss = logs["loss_pi"] + vf_coef * logs["loss_v"] - ent_coef_now * logs["entropy"]

        # Detach hidden banks so graph doesn’t grow across rollouts
        h0 = h0.detach()
        h1 = h1.detach()

        # ---- Logging ----
        if (update + 1) % cfg.log_interval == 0:
            fps = int((N * T) / max(1e-6, (time.time() - wall_t0)))
            wall_t0 = time.time()

            ep_return_mean = float(np.mean(ret_hist[-100:])) if len(ret_hist) > 0 else float(ep_ret.mean())
            ep_return_med  = float(np.median(ret_hist[-100:])) if len(ret_hist) > 0 else float(np.median(ep_ret))

            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            print(
                f"[{time_str}] [upd {update+1:05d}/{total_updates:05d}] "
                f"env_steps={global_env_steps:,} "
                f"R/ep(mean,last100)={ep_return_mean:.2f} "
                f"R/ep(med,last100)={ep_return_med:.2f} "
                f"loss_pi={logs['loss_pi']:.3f} "
                f"loss_v={logs['loss_v']:.3f} "
                f"total_loss={total_loss:.3f} "
                f"entropy={logs['entropy']:.3f} "
                f"clipfrac={logs['clip_frac']:.2f} "
                f"fps~{fps}"
            )


            # Tensorboard logging
            tb_writer.add_scalar('charts/ep_return_mean', ep_return_mean, global_env_steps)
            tb_writer.add_scalar('charts/ep_return_med', ep_return_med, global_env_steps)
            tb_writer.add_scalar('losses/loss_pi', logs['loss_pi'], global_env_steps)
            tb_writer.add_scalar('losses/loss_v', logs['loss_v'], global_env_steps)
            tb_writer.add_scalar('losses/entropy', logs['entropy'], global_env_steps)
            tb_writer.add_scalar('losses/clip_frac', logs['clip_frac'], global_env_steps)
            tb_writer.add_scalar('losses/total_loss', total_loss, global_env_steps)
            tb_writer.flush()

        # ---- Checkpointing ----
        if (update + 1) % cfg.save_interval == 0:
            ckpt_path = os.path.join(out_dir, f"ckpt_{update+1:06d}.pt")
            if hasattr(cfg, "__dict__"):
                cfg_state = dict(cfg.__dict__)
            else:
                cfg_state = dict(cfg)
            save_ckpt(
                path=ckpt_path,
                model_state=net.state_dict(),
                optim_state=opt.state_dict(),
                update=update + 1,
                cfg=cfg_state
            )
            print(f"[save] {ckpt_path}")

    print("Training finished.")


# --------------------------------- Entrypoint -------------------------------- #
if __name__ == "__main__":
    args = parse_args()
    cfg = CFG()

    # CLI overrides
    if args.lr is not None:
        cfg.ppo.lr = args.lr
    if args.total_updates is not None:
        cfg.total_updates = args.total_updates
    if args.save_dir is not None:
        cfg.out_dir = args.save_dir

    # Variant → sizes (requires cfg.hanabi to exist in config.py)
    if args.variant == "twoxtwo":
        cfg.hanabi.colors = 2
        cfg.hanabi.ranks = 2
        cfg.hanabi.hand_size = 2  # two-by-two means 2-card hands here
        if args.save_dir is None:
            cfg.out_dir = os.path.join(cfg.out_dir, "twoxtwo")
    else:  # standard 5x5
        cfg.hanabi.colors = 5
        cfg.hanabi.ranks = 5
        cfg.hanabi.hand_size = 5

    # On macOS, default to SyncVectorEnv unless user explicitly asks for async
    if IS_DARWIN and not args.async_env:
        print("[info] macOS detected: using SyncVectorEnv (set --async-env to override)")

    main(cfg, args)
