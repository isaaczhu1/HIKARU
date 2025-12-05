# ppo.py

import torch
import torch.nn.functional as F
import numpy as np


def masked_categorical(logits, legal_mask):
    """
    Returns: action, log_prob(action), Categorical(dist over *masked* actions)
    """
    very_neg = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(legal_mask < 0.5, very_neg)
    dist = torch.distributions.Categorical(logits=masked)
    a = dist.sample()
    return a, dist.log_prob(a), dist


@torch.no_grad()
def _approx_kl(old_logp, new_logp):
    # E[ old_logp - new_logp ], clamped to be non-negative
    return (old_logp - new_logp).mean().clamp_min(0.0)


def _value_loss_clipped(v_pred, v_old, ret, clip):
    """
    PPO-v1 style value clipping.
    v_pred, v_old, ret: tensors with same shape
    """
    v_clip = v_old + (v_pred - v_old).clamp(-clip, clip)
    loss_unclipped = (ret - v_pred).pow(2)
    loss_clipped   = (ret - v_clip).pow(2)
    return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()


def ppo_update(policy, optimizer, storage, cfg, ent_coef_override=None, target_kl=None):
    """
    Flat PPO (seq_len<=1) OR recurrent PPO with BPTT (seq_len>1 if cfg.ppo.seq_len set).

    - Flat mode uses storage.iter_minibatches(...)
      and forwards [B,1,...] through the GRU with h0 taken from storage if available.
    - Recurrent mode uses storage.iter_sequence_minibatches(seq_len, batch_size_in_seqs)
      and forwards [B,L,...] through the GRU with h0 from storage, where sequences
      are constructed per (env, seat) so that training recurrence matches rollout
      and each player's hidden state remains private to that player.
    """
    # --- Config / knobs ---
    clip = cfg.ppo.clip
    vf_coef = cfg.ppo.vf_coef
    max_grad_norm = cfg.ppo.max_grad_norm
    epochs = cfg.ppo.epochs
    ent_coef = getattr(cfg.ppo, "ent_coef", 0.0) if ent_coef_override is None else ent_coef_override
    seq_len = int(getattr(cfg.ppo, "seq_len", 1))
    v_clip = float(getattr(cfg.ppo, "value_clip", 0.2))
    # target_kl: prefer explicit arg, else cfg.sched.target_kl if present
    if target_kl is None:
        target_kl = getattr(getattr(cfg, "sched", object()), "target_kl", None)

    ent_all, clip_all, pi_all, v_all = [], [], [], []
    approx_kl_tracking = 0.0

    # ---------------------------- FLAT MODE ---------------------------------- #
    if seq_len <= 1:
        for _ in range(epochs):
            for mb in storage.iter_minibatches(cfg.ppo.minibatch):
                obs   = mb["obs"]        # [B, obs_dim]
                legal = mb["legal"]      # [B, A]
                seat  = mb["seat"]       # [B]
                prev  = mb["prev_other"] # [B]

                B = obs.shape[0]
                device = obs.device

                # Determine model hidden size
                H_model = getattr(policy, "hidden", None)
                if H_model is None and hasattr(policy, "gru"):
                    H_model = policy.gru.hidden_size

                # Initial hidden: use stored h0 if available; else zeros.
                if "h0" in mb:
                    h0 = mb["h0"]
                    if h0.dim() != 2:
                        h0 = h0.reshape(B, -1)
                    if H_model is not None and h0.shape[1] != H_model:
                        h0 = torch.zeros(B, H_model, device=device, dtype=obs.dtype)
                else:
                    if H_model is not None:
                        h0 = torch.zeros(B, H_model, device=device, dtype=obs.dtype)
                    else:
                        # Fallback: use policy.initial_state if size unknown
                        h0 = policy.initial_state(B, device=device).squeeze(0)
                h0 = h0.unsqueeze(0)  # [1, B, H]

                # Forward with seq_len=1
                logits, value, _ = policy(
                    obs_vec=obs.unsqueeze(1),            # [B,1,obs_dim]
                    seat=seat.unsqueeze(1),              # [B,1]
                    prev_other=prev.unsqueeze(1),        # [B,1]
                    h=h0
                )
                logits = logits.squeeze(1)               # [B, A]
                value  = value.squeeze(1)                # [B]

                # Masked categorical
                very_neg = torch.finfo(logits.dtype).min
                masked = logits.masked_fill(legal < 0.5, very_neg)
                dist = torch.distributions.Categorical(logits=masked)
                logp = dist.log_prob(mb["act"])          # [B]
                entropy = dist.entropy().mean()

                ratio = (logp - mb["logp_old"]).exp()
                # Normalize advantages in-minibatch
                adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std(unbiased=False) + 1e-8)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
                loss_pi = -(torch.min(surr1, surr2)).mean()

                # Value loss (clipped)
                loss_v = _value_loss_clipped(
                    v_pred=value,
                    v_old=mb["val_old"],
                    ret=mb["ret"],
                    clip=v_clip,
                )

                loss = loss_pi + vf_coef * loss_v - ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = _approx_kl(mb["logp_old"], logp)
                    clipfrac = (torch.abs(ratio - 1.0) > clip).float().mean()
                    approx_kl_tracking = float(approx_kl)

                ent_all.append(float(entropy))
                clip_all.append(float(clipfrac))
                pi_all.append(float(loss_pi))
                v_all.append(float(loss_v))

            # Early stop on target KL (per-epoch)
            if target_kl is not None and approx_kl_tracking > target_kl:
                break

        return {
            "entropy": np.mean(ent_all) if ent_all else 0.0,
            "clip_frac": np.mean(clip_all) if clip_all else 0.0,
            "loss_pi": np.mean(pi_all) if pi_all else 0.0,
            "loss_v": np.mean(v_all) if v_all else 0.0,
        }

    # ------------------------- RECURRENT (BPTT) MODE ------------------------- #
    # Convert sample-sized minibatch to sequences-per-minibatch
    # e.g. if minibatch=2048 and seq_len=16 -> 128 sequences per minibatch
    seqs_per_mb = max(1, int(cfg.ppo.minibatch // seq_len))

    for _ in range(epochs):
        for mb in storage.iter_sequence_minibatches(seq_len=seq_len, batch_size=seqs_per_mb):
            # mb fields shaped:
            #   obs [B,L,obs_dim], legal [B,L,A], seat [B,L], prev_other [B,L],
            #   act [B,L], logp_old [B,L], adv [B,L], ret [B,L], h0 [B,H], val_old [B,L]
            obs   = mb["obs"]
            legal = mb["legal"]
            seat  = mb["seat"]
            prev  = mb["prev_other"]
            act   = mb["act"]
            logp_old = mb["logp_old"]
            adv   = mb["adv"]
            ret   = mb["ret"]
            h0    = mb["h0"]
            val_old = mb["val_old"]

            B, L = obs.shape[0], obs.shape[1]
            A = legal.shape[-1]
            device = obs.device

            # Normalize advantages per minibatch over all timesteps
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            # Ensure h0 matches model hidden size; fall back to zeros if storage didn't record size
            H_model = getattr(policy, "hidden", None)
            if H_model is None and hasattr(policy, "gru"):
                H_model = policy.gru.hidden_size
            if h0.dim() != 2:
                h0 = h0.reshape(B, -1)
            if H_model is not None and h0.shape[1] != H_model:
                h0 = torch.zeros(B, H_model, device=device, dtype=obs.dtype)
            h0 = h0.unsqueeze(0)  # [1, B, H]

            # Forward full sequence
            logits, values, _ = policy(
                obs_vec=obs,                 # [B,L,obs_dim]
                seat=seat,                   # [B,L]
                prev_other=prev,             # [B,L]
                h=h0                         # [1,B,H]
            )  # logits: [B,L,A], values: [B,L]
            if values.dim() == 3 and values.shape[-1] == 1:
                values = values.squeeze(-1)  # [B,L]

            # Flatten time for masked dist & losses
            logits_f = logits.reshape(B * L, A)
            legal_f  = legal.reshape(B * L, A)
            very_neg = torch.finfo(logits_f.dtype).min
            masked_f = logits_f.masked_fill(legal_f < 0.5, very_neg)
            dist_f = torch.distributions.Categorical(logits=masked_f)

            act_f      = act.reshape(B * L)
            logp_new_f = dist_f.log_prob(act_f)               # [B*L]
            entropy_f  = dist_f.entropy().mean()

            logp_old_f = logp_old.reshape(B * L)
            ratio_f    = (logp_new_f - logp_old_f).exp()

            adv_f = adv.reshape(B * L)
            ret_f = ret.reshape(B * L)
            val_f = values.reshape(B * L)
            val_old_f = val_old.reshape(B * L)

            surr1 = ratio_f * adv_f
            surr2 = torch.clamp(ratio_f, 1.0 - clip, 1.0 + clip) * adv_f
            loss_pi = -(torch.min(surr1, surr2)).mean()

            # Value loss (clipped)
            loss_v  = _value_loss_clipped(
                v_pred=val_f,
                v_old=val_old_f,
                ret=ret_f,
                clip=v_clip,
            )

            loss = loss_pi + vf_coef * loss_v - ent_coef * entropy_f

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = _approx_kl(logp_old_f, logp_new_f)
                clipfrac  = (torch.abs(ratio_f - 1.0) > clip).float().mean()
                approx_kl_tracking = float(approx_kl)

            ent_all.append(float(entropy_f))
            clip_all.append(float(clipfrac))
            pi_all.append(float(loss_pi))
            v_all.append(float(loss_v))

        # Early stop on target KL (per-epoch)
        if target_kl is not None and approx_kl_tracking > target_kl:
            break

    return {
        "entropy": np.mean(ent_all) if ent_all else 0.0,
        "clip_frac": np.mean(clip_all) if clip_all else 0.0,
        "loss_pi": np.mean(pi_all) if pi_all else 0.0,
        "loss_v": np.mean(v_all) if v_all else 0.0,
    }
