#!/usr/bin/env python
"""
Load a trained GRU policy checkpoint and evaluate it over N games.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

from hanabi_gru_baseline.config import CFG
from hanabi_gru_baseline.hanabi_envs import HanabiEnv2P
from hanabi_gru_baseline.model import HanabiGRUPolicy
from hanabi_gru_baseline.utils import load_ckpt


def _merge_cfg(base_cfg: CFG, cfg_dict: dict) -> CFG:
    """Shallow merge a saved cfg dict into a CFG instance."""

    def update(obj, d: dict):
        for k, v in d.items():
            if not hasattr(obj, k):
                continue
            cur = getattr(obj, k)
            if isinstance(v, dict) and hasattr(cur, "__dict__"):
                update(cur, v)
            else:
                setattr(obj, k, v)

    if cfg_dict:
        update(base_cfg, cfg_dict)
    return base_cfg


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained Hanabi GRU policy")
    p.add_argument("--ckpt", required=True, help="path to checkpoint (.pt)")
    p.add_argument("--episodes", type=int, default=50, help="number of games to run")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda (auto if None)")
    p.add_argument("--seed", type=int, default=0, help="base seed for eval games")
    p.add_argument("--greedy", action="store_true", help="use argmax actions instead of sampling")
    return p.parse_args()


def build_env(cfg: CFG, seed: int):
    return HanabiEnv2P(
        seed=seed,
        obs_conf=cfg.obs_mode,
        players=cfg.hanabi.players,
        colors=cfg.hanabi.colors,
        ranks=cfg.hanabi.ranks,
        hand_size=cfg.hanabi.hand_size,
        max_information_tokens=cfg.hanabi.max_information_tokens,
        max_life_tokens=cfg.hanabi.max_life_tokens,
        random_start_player=cfg.hanabi.random_start_player,
    )


def build_model(cfg: CFG, obs_dim: int, num_moves: int, device: torch.device):
    mcfg = cfg.model
    net = HanabiGRUPolicy(
        obs_dim=obs_dim,
        num_moves=num_moves,
        hidden=mcfg.hidden,
        action_emb_dim=mcfg.action_emb,
        seat_emb_dim=mcfg.seat_emb,
        include_prev_self=mcfg.include_prev_self,
    ).to(device)
    return net


def evaluate(net, env: HanabiEnv2P, device: torch.device, episodes: int, seed0: int, greedy: bool):
    scores = []
    net.eval()
    with torch.no_grad():
        for ep in range(episodes):
            obs_dict = env.reset(seed=seed0 + ep)
            h0 = net.initial_state(1, device=device)
            h1 = net.initial_state(1, device=device)
            done = False
            score = 0.0

            while not done:
                obs = torch.from_numpy(obs_dict["obs"]).to(device).float().unsqueeze(0)
                legal = torch.from_numpy(obs_dict["legal_mask"]).to(device).float().unsqueeze(0)
                seat = torch.tensor([obs_dict["seat"]], device=device).long()
                prev = torch.tensor([obs_dict["prev_other_action"]], device=device).long()

                h_in = torch.where(seat.view(1, -1, 1) == 0, h0, h1)

                logits, _, h_new = net(
                    obs_vec=obs.unsqueeze(1),
                    seat=seat.unsqueeze(1),
                    prev_other=prev.unsqueeze(1),
                    h=h_in,
                )
                logits = logits.squeeze(1)

                probs = torch.softmax(logits, dim=-1) * legal
                probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

                if greedy:
                    action = probs.argmax(dim=-1)
                else:
                    action = torch.multinomial(probs, num_samples=1).squeeze(1)

                a_id = int(action.item())
                obs_dict, rew, done, info = env.step(a_id)

                if seat.item() == 0:
                    h0[:, :, :] = h_new
                else:
                    h1[:, :, :] = h_new

                score = float(info.get("score", score + float(rew)))

            scores.append(score)
    return scores


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        device = "cpu"

    ckpt = load_ckpt(args.ckpt)
    cfg_dict = ckpt.get("cfg", {}) or {}
    cfg = _merge_cfg(CFG(), cfg_dict if isinstance(cfg_dict, dict) else {})

    state_dict = ckpt["model"]

    # Infer model dims from checkpoint
    num_moves_infer = state_dict["prev_other_emb.weight"].shape[0] - 1
    obs_dim_infer = state_dict["obs_fe.0.weight"].shape[1]
    hidden_infer = state_dict["gru.weight_hh_l0"].shape[1]
    action_emb_infer = state_dict["prev_other_emb.weight"].shape[1]
    seat_emb_infer = state_dict["seat_emb.weight"].shape[1]
    include_prev_self = "prev_self_emb.weight" in state_dict

    # Align config to inferred dimensions / likely Hanabi sizes
    if num_moves_infer == 20:
        cfg.hanabi.colors = 5
        cfg.hanabi.ranks = 5
        cfg.hanabi.hand_size = 5
    elif num_moves_infer == 8:
        cfg.hanabi.colors = 2
        cfg.hanabi.ranks = 2
        cfg.hanabi.hand_size = 2
    else:
        print(f"[warn] Unrecognized num_moves {num_moves_infer}; using current cfg hanabi sizes.")

    cfg.model.hidden = hidden_infer
    cfg.model.action_emb = action_emb_infer
    cfg.model.seat_emb = seat_emb_infer
    cfg.model.include_prev_self = include_prev_self

    env = build_env(cfg, seed=args.seed + 9999)
    probe = env.reset(seed=args.seed + 12345)
    obs_dim_env = probe["obs"].shape[-1]
    num_moves_env = probe["legal_mask"].shape[-1]

    if obs_dim_env != obs_dim_infer or num_moves_env != num_moves_infer:
        raise RuntimeError(
            f"Env shapes mismatch checkpoint: env (obs_dim={obs_dim_env}, num_moves={num_moves_env}) "
            f"vs ckpt (obs_dim={obs_dim_infer}, num_moves={num_moves_infer}). Adjust hanabi sizes."
        )

    net = build_model(cfg, obs_dim=obs_dim_infer, num_moves=num_moves_infer, device=device)
    net.load_state_dict(state_dict)

    scores = evaluate(net, env, device, episodes=args.episodes, seed0=args.seed, greedy=args.greedy)
    scores_np = np.asarray(scores, dtype=np.float32)
    print(f"Episodes: {len(scores)}")
    print(f"Mean score: {scores_np.mean():.2f}")
    print(f"Median score: {np.median(scores_np):.2f}")
    print(f"Max score: {scores_np.max():.2f}")
    print(f"Scores: {scores}")


if __name__ == "__main__":
    main()
