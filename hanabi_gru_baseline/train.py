# train.py
# -----------------------------------------------------------------------------
# End-to-end PPO + GRU baseline trainer for 2-player Hanabi (DeepMind HLE).
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
from ppo import ppo_update
from utils import seed_everything, save_ckpt, load_ckpt

from torch.utils.tensorboard import SummaryWriter

IS_DARWIN = sys.platform == "darwin"


# ----------------------------- Argument parsing ------------------------------ #
def parse_args():
    ap = argparse.ArgumentParser(description="PPO+GRU baseline for 2p Hanabi (HLE)")
    ap.add_argument("--device", type=str, default=None, help="cuda | cpu (auto if None)")
    ap.add_argument("--total-updates", type=int, default=None, help="override total updates")
    ap.add_argument("--lr", type=float, default=None, help="override learning rate")
    ap.add_argument("--lr-final", type=float, default=None, help="override final learning rate for schedule")
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
    Handles both AsyncVectorEnv and SyncVectorEnv, as well as fallback vector resets.
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
    if isinstance(all_obs, dict):
        for j in idxs:
            slot = {}
            for k, v in all_obs.items():
                try:
                    slot[k] = v[j]
                except Exception:
                    pass
            fresh.append(slot)
        return fresh

    return [all_obs[j] for j in idxs]


# ---------------------------- One environment step --------------------------- #
@torch.no_grad()
def do_step(env, net, device, h0, h1, obs_dict, debug=False, eps=0.0):
    """
    Execute a single time step for all N envs in the vector env.
    Auto-resets any slots with zero-legal before sampling, and post-step for done slots.
    """
    N = obs_dict["obs"].shape[0]
    forced_reset = np.zeros(N, dtype=np.bool_)

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
            forced_reset[j] = True
        legal_np = obs_dict["legal_mask"]

        # Keep GRU in sync with env resets: zero hidden for forced-reset envs
        fr_ids = np.nonzero(forced_reset)[0]
        if fr_ids.size > 0:
            fr_t = torch.from_numpy(fr_ids).to(device)
            h0[:, fr_t, :] = 0.0
            h1[:, fr_t, :] = 0.0

    # ----- Torchify current batch -----
    obs   = torch.from_numpy(obs_dict["obs"]).to(device).float()         # [N, obs_dim]
    legal = torch.from_numpy(legal_np).to(device).float()                # [N, A]
    seat  = torch.from_numpy(obs_dict["seat"]).to(device).long()         # [N]
    prev  = torch.from_numpy(obs_dict["prev_other_action"]).to(device).long()  # [N]

    # Last-resort safety: if any row is still zero-legal, make action 0 legal
    zero_rows = (legal.sum(dim=1) == 0)
    if zero_rows.any():
        legal[zero_rows, 0] = 1.0
        if debug:
            zr = zero_rows.nonzero(as_tuple=False).squeeze(-1).tolist()
            print(f"[warn] Forcing a legal action on rows {zr}", flush=True)

    # ----- Select per-seat hidden -----
    h_in = torch.where(seat.view(1, -1, 1) == 0, h0, h1)                 # [1, N, H]

    # ----- Forward (seq len 1) -----
    logits, value, h_new = net(
        obs_vec=obs.unsqueeze(1),        # [N, 1, obs_dim]
        seat=seat.unsqueeze(1),          # [N, 1]
        prev_other=prev.unsqueeze(1),    # [N, 1]
        h=h_in                           # [1, N, H]
    )
    logits = logits.squeeze(1)           # [N, A]
    value  = value.squeeze(1)            # [N]

    # ----- Sample masked actions with ε-greedy mixture over legal -----
    probs = torch.softmax(logits, dim=-1)              # [N, A]
    probs = probs * legal                              # mask illegal
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    uniform = legal / legal.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mix = (1.0 - eps) * probs + eps * uniform          # [N, A]
    action = torch.multinomial(mix, num_samples=1).squeeze(1)  # [N]

    # Log-prob under the POLICY (not the mixture) for PPO ratios
    logp = torch.log(probs.gather(1, action.view(-1, 1)).squeeze(1).clamp_min(1e-8))

    # ----- Step env -----
    next_obs_dict, rew_np, done_np, trunc_np, info = env.step(action.detach().cpu().numpy())

    # Extract per-env score from info if present
    scores_np = None
    if info is not None:
        if isinstance(info, dict) and "score" in info:
            scores_np = np.asarray(info["score"], dtype=np.float32)
        elif isinstance(info, (list, tuple)):
            maybe_scores = []
            for item in info:
                if isinstance(item, dict) and "score" in item:
                    maybe_scores.append(item["score"])
                else:
                    maybe_scores.append(np.nan)
            if maybe_scores:
                scores_np = np.asarray(maybe_scores, dtype=np.float32)

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
        "obs": obs,
        "legal": legal,
        "seat": seat,
        "prev_other": prev,
        "action": action,
        "logp": logp.detach(),
        "value": value.detach(),
        "reward": torch.from_numpy(rew_np).to(device).float(),
        "done": torch.from_numpy(done_or_trunc.astype(np.float32)).to(device),
        "h": h_in.squeeze(0).detach(),
    }

    # include score for logging in main
    epi = {"done": done_or_trunc, "reward": rew_np, "score": scores_np}
    return next_obs_dict, step_record, h0, h1, epi, forced_reset


def run_eval(net, cfg, device, n_episodes=16, greedy=True):
    """
    Run a small number of single-env evaluation games and return
    (mean, median, max) episode scores.

    Uses the same HanabiGym2P wrapper as training, but decoupled from
    AsyncVectorEnv / logging weirdness.
    """
    from hanabi_envs import HanabiGym2P
    import numpy as np
    import torch

    # Single eval env, same game config as training
    eval_env = HanabiGym2P(
        seed=cfg.seed + 12345,
        obs_conf=cfg.obs_mode,
        players=cfg.hanabi.players,
        colors=cfg.hanabi.colors,
        ranks=cfg.hanabi.ranks,
        hand_size=cfg.hanabi.hand_size,
        max_information_tokens=cfg.hanabi.max_information_tokens,
        max_life_tokens=cfg.hanabi.max_life_tokens,
        random_start_player=cfg.hanabi.random_start_player,
    )

    scores = []
    net_was_training = net.training
    net.eval()

    with torch.no_grad():
        for ep in range(n_episodes):
            # Different seed per eval episode for variety
            obs_dict, _ = eval_env.reset(seed=cfg.seed + 1000 + ep)

            # Per-seat GRU hidden banks, shape [1, 1, H]
            h0 = net.initial_state(1, device=device)
            h1 = net.initial_state(1, device=device)

            done = False
            truncated = False
            score = 0.0

            while not (done or truncated):
                # Torchify
                obs   = torch.from_numpy(obs_dict["obs"]).to(device).float().unsqueeze(0)     # [1, obs_dim]
                legal = torch.from_numpy(obs_dict["legal_mask"]).to(device).float().unsqueeze(0)  # [1, A]
                seat  = torch.tensor([obs_dict["seat"]], device=device).long()               # [1]
                prev  = torch.tensor([obs_dict["prev_other_action"]], device=device).long()  # [1]

                # Choose the right hidden bank for current seat
                h_in = torch.where(seat.view(1, -1, 1) == 0, h0, h1)   # [1, 1, H]

                # Forward through GRU policy (seq_len=1)
                logits, _, h_new = net(
                    obs_vec=obs.unsqueeze(1),        # [1, 1, obs_dim]
                    seat=seat.unsqueeze(1),          # [1, 1]
                    prev_other=prev.unsqueeze(1),    # [1, 1]
                    h=h_in                           # [1, 1, H]
                )
                logits = logits.squeeze(1)           # [1, A]

                # Mask illegal moves
                probs = torch.softmax(logits, dim=-1) * legal
                probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

                if greedy:
                    action = probs.argmax(dim=-1)
                else:
                    action = torch.multinomial(probs, num_samples=1).squeeze(1)

                a_id = int(action.item())

                # Step env
                obs_dict, rew, done, truncated, info = eval_env.step(a_id)

                # Update hidden banks for the acting seat
                if seat.item() == 0:
                    h0[:, :, :] = h_new
                else:
                    h1[:, :, :] = h_new

                # Our wrapper keeps cumulative score in info["score"]
                score = float(info.get("score", score + float(rew)))

            scores.append(score)

    if net_was_training:
        net.train()

    scores = np.asarray(scores, dtype=np.float32)
    return float(scores.mean()), float(np.median(scores)), float(scores.max())


# --------------------------------- Trainer ---------------------------------- #
def main(cfg: CFG, args):
    # ---------- Device & seeding ---------- #
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] using device: {device}")
    seed_everything(cfg.seed)

    out_dir = args.save_dir or cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    max_score = int(cfg.hanabi.colors * cfg.hanabi.ranks)
    print(f"[info] Hanabi {cfg.hanabi.colors}x{cfg.hanabi.ranks}, max score = {max_score}")

    # ---------- Environments ---------- #
    env = make_vec_env(cfg.num_envs, cfg.seed, obs_conf=cfg.obs_mode,
                       use_async=args.async_env, hanabi_cfg=cfg.hanabi)
    obs_dict, _ = env.reset()
    print("[debug] initial legal sums:", obs_dict["legal_mask"].sum(axis=1)[:8])

    # ---------- TensorBoard logging ---------- #
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=os.path.join("tb", out_dir, timestamp))

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
        include_prev_self=cfg.model.include_prev_self,
    ).to(device)

    lr = args.lr if args.lr is not None else cfg.ppo.lr
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # Per-seat GRU hidden banks: [1, N, H]
    h0 = net.initial_state(N, device=device)
    h1 = net.initial_state(N, device=device)

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
        eps_now = cfg.sched.eps0 * (1.0 - eps_decay)

        # ---- Reset rollout buffer cursor ----
        storage.reset_episode_slice()

        # ---- Rollout collection: T steps ----
        with torch.no_grad():
            for t in range(T):
                obs_dict, step_rec, h0, h1, epi, forced_reset = do_step(
                    env=env, net=net, device=device,
                    h0=h0, h1=h1, obs_dict=obs_dict,
                    debug=args.debug, eps=eps_now,
                )
                storage.add(step_rec)

                # Treat forced resets as episode boundaries for logging only
                if forced_reset.any():
                    ep_ret[forced_reset] = 0.0
                    ep_len[forced_reset] = 0

                # Track shaped returns (for debugging / fallback)
                ep_ret += epi["reward"]
                ep_len += 1

                if np.any(epi["done"]):
                    done_ids = np.where(epi["done"])[0]

                    # Score = sum of per-step shaped rewards over the episode
                    ep_returns = ep_ret[done_ids]

                    if args.debug:
                        print("[DEBUG TRAIN] done_ids:", done_ids,
                            "ep_returns_at_done:", ep_returns)
                        if (ep_returns < -1e-6).any() or (ep_returns > max_score + 1e-6).any():
                            raise AssertionError(f"Episode return out of range: {ep_returns} (max_score={max_score})")

                    ret_hist.extend(ep_returns.tolist())
                    ep_ret[done_ids] = 0.0
                    ep_len[done_ids] = 0



                global_env_steps += N

            # ---- Bootstrap value at last state (for GAE tail) ----
            obs_t   = torch.from_numpy(obs_dict["obs"]).to(device).float()       # [N, obs_dim]
            seat_t  = torch.from_numpy(obs_dict["seat"]).to(device).long()       # [N]
            prev_t  = torch.from_numpy(obs_dict["prev_other_action"]).to(device).long()  # [N]
            h_in_T  = torch.where(seat_t.view(1, -1, 1) == 0, h0, h1)            # [1, N, H]
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

        # ---- PPO update ----
        decay = min(1.0, (update + 1) / max(1, cfg.sched.ent_decay_until))
        ent_coef_now = cfg.ppo.ent_coef + (cfg.sched.ent_final - cfg.ppo.ent_coef) * decay
        logs = ppo_update(policy=net, optimizer=opt, storage=storage, cfg=cfg,
                          ent_coef_override=ent_coef_now)

        vf_coef = cfg.ppo.vf_coef
        total_loss = logs["loss_pi"] + vf_coef * logs["loss_v"] - ent_coef_now * logs["entropy"]

        # Detach hidden banks so graph doesn’t grow across rollouts
        h0 = h0.detach()
        h1 = h1.detach()

        # ---- Logging ----
        if (update + 1) % cfg.log_interval == 0:
            fps = int((N * T) / max(1e-6, (time.time() - wall_t0)))
            wall_t0 = time.time()

            # Training-return stats from the vector env (BROKEN for async, fine for sync)
            if len(ret_hist) > 0:
                window = ret_hist[-100:]
                train_mean = float(np.mean(window))
                train_med  = float(np.median(window))
                train_count = len(window)
                train_max  = float(np.max(window))
            else:
                train_mean = float(ep_ret.mean())
                train_med  = float(np.median(ep_ret))
                train_count = 0
                train_max  = 0.0

            # --- Eval pass: trustworthy scores using single env ---
            eval_mean = eval_med = eval_max = 0.0
            if args.async_env:
                try:
                    eval_mean, eval_med, eval_max = run_eval(
                        net, cfg, device, n_episodes=16, greedy=True
                    )
                except Exception as e:
                    print(f"[warn] eval run failed: {e}", flush=True)

            # For printing:
            # - sync: R/ep(...) = training stats (same as before)
            # - async: R/ep(...) = eval stats, and we also show train_R/... separately
            if args.async_env:
                r_mean = eval_mean
                r_med  = eval_med
                r_count = train_count  # still show how many train eps we saw
                r_max  = eval_max
            else:
                r_mean = train_mean
                r_med  = train_med
                r_count = train_count
                r_max  = train_max

            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            if args.async_env:
                # async: R/ep = eval; training stats printed separately
                print(
                    f"[{time_str}] [upd {update+1:05d}/{total_updates:05d}] "
                    f"env_steps={global_env_steps:,} "
                    f"R/ep(mean,eval)={r_mean:.2f} "
                    f"R/ep(med,eval)={r_med:.2f} "
                    f"R/ep(max,eval)={r_max:.2f} "
                    f"train_R/ep(mean,last100)={train_mean:.2f} "
                    f"train_R/ep(med,last100)={train_med:.2f} "
                    f"train_R/ep(count,last100)={train_count} "
                    f"train_R/ep(max,last100)={train_max:.2f} "
                    f"loss_pi={logs['loss_pi']:.3f} "
                    f"loss_v={logs['loss_v']:.3f} "
                    f"total_loss={total_loss:.3f} "
                    f"entropy={logs['entropy']:.3f} "
                    f"clipfrac={logs['clip_frac']:.2f} "
                    f"fps~{fps}"
                )
            else:
                # sync: keep old-format R/ep from training
                print(
                    f"[{time_str}] [upd {update+1:05d}/{total_updates:05d}] "
                    f"env_steps={global_env_steps:,} "
                    f"R/ep(mean,last100)={r_mean:.2f} "
                    f"R/ep(med,last100)={r_med:.2f} "
                    f"R/ep(count,last100)={r_count} "
                    f"R/ep(max,last100)={r_max:.2f} "
                    f"loss_pi={logs['loss_pi']:.3f} "
                    f"loss_v={logs['loss_v']:.3f} "
                    f"total_loss={total_loss:.3f} "
                    f"entropy={logs['entropy']:.3f} "
                    f"clipfrac={logs['clip_frac']:.2f} "
                    f"fps~{fps}"
                )


        # ---- Checkpointing ----
        if (update + 1) % cfg.save_interval == 0:
            ckpt_path = os.path.join(out_dir, f"ckpt_{update+1:06d}.pt")
            cfg_state = dict(cfg.__dict__) if hasattr(cfg, "__dict__") else dict(cfg)
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
    if args.lr_final is not None:
        cfg.sched.lr_final = args.lr_final
    if args.total_updates is not None:
        cfg.total_updates = args.total_updates
    if args.save_dir is not None:
        cfg.out_dir = args.save_dir

    # Variant → sizes
    if args.variant == "twoxtwo":
        cfg.hanabi.colors = 2
        cfg.hanabi.ranks = 2
        cfg.hanabi.hand_size = 2
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
