# ppo.py
import torch
import numpy as np

def masked_categorical(logits, legal_mask):
    very_neg = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(legal_mask < 0.5, very_neg)
    dist = torch.distributions.Categorical(logits=masked)
    a = dist.sample()
    return a, dist.log_prob(a), dist


def ppo_update(policy, optimizer, storage, cfg, ent_coef_override=None, target_kl=None):
    # use override if provided, else fall back to config
    ent_coef = getattr(cfg.ppo, "ent_coef", 0.0) if ent_coef_override is None else ent_coef_override
    logs = {}
    ent_all, clip_all, pi_all, v_all = [], [], [], []
    for _ in range(cfg.ppo.epochs):
        for mb in storage.iter_minibatches(cfg.ppo.minibatch):
            logits, value, _ = policy(
                obs_vec=mb["obs"].unsqueeze(1),
                seat=mb["seat"].unsqueeze(1),
                prev_other=mb["prev_other"].unsqueeze(1),
                h=policy.initial_state(mb["obs"].shape[0], device=mb["obs"].device)
            )
            logits = logits.squeeze(1); value = value.squeeze(1)

            very_neg = torch.finfo(logits.dtype).min
            masked = logits.masked_fill(mb["legal"] < 0.5, very_neg)
            dist = torch.distributions.Categorical(logits=masked)
            logp = dist.log_prob(mb["act"])
            entropy = dist.entropy().mean()

            ratio = (logp - mb["logp_old"]).exp()
            adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std(unbiased=False) + 1e-8)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-cfg.ppo.clip, 1+cfg.ppo.clip) * adv
            loss_pi = -(torch.min(surr1, surr2)).mean()
            loss_v = 0.5 * (mb["ret"] - value).pow(2).mean()
            loss_ent = - cfg.ppo.ent_coef * entropy
            # loss = loss_pi + cfg.ppo.vf_coef * loss_v + loss_ent
            loss = loss_pi + cfg.ppo.vf_coef * loss_v - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.ppo.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                clipfrac = (torch.abs(ratio - 1.0) > cfg.ppo.clip).float().mean()
            ent_all.append(float(entropy)); clip_all.append(float(clipfrac))
            pi_all.append(float(loss_pi)); v_all.append(float(loss_v))
    logs["entropy"] = np.mean(ent_all); logs["clip_frac"] = np.mean(clip_all)
    logs["loss_pi"] = np.mean(pi_all); logs["loss_v"] = np.mean(v_all)
    return logs
