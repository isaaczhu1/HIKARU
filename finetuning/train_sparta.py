#!/usr/bin/env python
"""
Supervised fine-tuning of the Hanabi GRU policy on SPARTA rollouts.

Usage example:
  python -m finetuning.train_sparta \
    --data sparta_games.jsonl \
    --ckpt runs/hanabi/standard_train/ckpt_020000.pt \
    --out runs/hanabi/finetune_sparta/ckpt_ft.pt \
    --epochs 2 --batch-size 1024 --lr 5e-5
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import DataLoader

from finetuning.data import build_training_samples, load_sparta_logs
from finetuning.dataset import SpartaDataset, collate_sparta
from finetuning.model_utils import load_model_from_ckpt
from hanabi_gru_baseline.utils import save_ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised fine-tuning on SPARTA rollouts.")
    p.add_argument("--data", nargs="+", required=True, help="Path(s) to SPARTA log files (JSONL or JSON).")
    p.add_argument("--ckpt", required=True, help="PPO checkpoint to initialize the model.")
    p.add_argument("--out", required=True, help="Output path for the fine-tuned checkpoint.")
    p.add_argument("--epochs", type=int, default=1, help="Number of epochs over the SPARTA dataset.")
    p.add_argument("--batch-size", type=int, default=1024, help="Batch size for fine-tuning.")
    p.add_argument("--lr", type=float, default=3e-5, help="Learning rate (Adam).")
    p.add_argument("--lambda-value", type=float, default=0.5, help="Weight for value MSE loss.")
    p.add_argument("--lambda-entropy", type=float, default=1e-3, help="Weight for entropy bonus (small).")
    p.add_argument("--override-weight", type=float, default=1.0, help="Optional weight for override states (action_sparta != action_blueprint).")
    p.add_argument("--discount", type=float, default=1.0, help="Discount factor used when computing returns.")
    p.add_argument("--device", type=str, default=None, help="Device (cpu|cuda). Defaults to cuda if available.")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    p.add_argument("--pin-memory", action="store_true", help="Enable pin_memory for DataLoader.")
    p.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping value.")
    return p.parse_args()


def make_dataloader(
    samples: Sequence[dict],
    include_prev_self: bool,
    sentinel: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    ds = SpartaDataset(samples)
    collate = partial(collate_sparta, include_prev_self=include_prev_self, sentinel=sentinel)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def compute_loss(
    net,
    batch,
    lambda_value: float,
    lambda_entropy: float,
    override_weight: float,
) -> torch.Tensor:
    obs = batch["obs"]
    seat = batch["seat"]
    prev_other = batch["prev_other"]
    prev_self = batch["prev_self"]
    legal = batch["legal"]
    action_target = batch["action_target"]
    value_target = batch["value_target"]
    is_override = batch["is_override"]

    B = obs.shape[0]
    device = next(net.parameters()).device
    obs = obs.to(device)
    seat = seat.to(device)
    prev_other = prev_other.to(device)
    legal = legal.to(device)
    action_target = action_target.to(device)
    value_target = value_target.to(device)
    if is_override is not None:
        is_override = is_override.to(device)
    if prev_self is not None:
        prev_self = prev_self.to(device)

    h0 = net.initial_state(B, device=device)
    logits, value, _ = net(
        obs_vec=obs.unsqueeze(1),
        seat=seat.unsqueeze(1),
        prev_other=prev_other.unsqueeze(1),
        h=h0,
        prev_self=None if prev_self is None else prev_self.unsqueeze(1),
    )

    logits = logits.squeeze(1)  # [B, A]
    value = value.squeeze(1)  # [B]

    very_neg = torch.finfo(logits.dtype).min
    masked_logits = torch.where(legal > 0, logits, very_neg)

    # Policy imitation loss
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    chosen_logp = log_probs.gather(1, action_target.view(-1, 1)).squeeze(1)

    if is_override is not None and override_weight != 1.0:
        weights = torch.ones_like(chosen_logp)
        weights = torch.where(is_override > 0, weights * override_weight, weights)
        pi_loss = -(weights * chosen_logp).sum() / weights.sum().clamp_min(1e-6)
    else:
        pi_loss = -chosen_logp.mean()

    # Value regression loss
    value_loss = (value - value_target).pow(2).mean()

    # Entropy (on masked logits)
    dist = torch.distributions.Categorical(logits=masked_logits)
    entropy = dist.entropy().mean()

    total = pi_loss + lambda_value * value_loss - lambda_entropy * entropy
    return total, {"pi": float(pi_loss), "value": float(value_loss), "entropy": float(entropy)}


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    net, info = load_model_from_ckpt(args.ckpt, device=device)
    net.train()

    records = load_sparta_logs(args.data)
    samples = build_training_samples(
        records,
        num_moves=info.num_moves,
        obs_dim=info.obs_dim,
        discount=args.discount,
    )

    dataloader = make_dataloader(
        samples=samples,
        include_prev_self=info.include_prev_self,
        sentinel=info.num_moves,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    total_steps = len(dataloader)
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            optim.zero_grad(set_to_none=True)
            loss, metrics = compute_loss(
                net,
                batch,
                lambda_value=args.lambda_value,
                lambda_entropy=args.lambda_entropy,
                override_weight=args.override_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optim.step()

            epoch_loss += float(loss)
            if step % max(1, total_steps // 10) == 0 or step == total_steps:
                print(
                    f"[epoch {epoch+1}/{args.epochs}] step {step}/{total_steps} "
                    f"loss={loss:.4f} pi={metrics['pi']:.4f} "
                    f"val={metrics['value']:.4f} ent={metrics['entropy']:.4f}",
                    flush=True,
                )

        mean_loss = epoch_loss / max(1, total_steps)
        print(f"[epoch {epoch+1}] mean_loss={mean_loss:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_ckpt(
        path=str(out_path),
        model_state=net.state_dict(),
        optim_state=optim.state_dict(),
        update=0,
        cfg={"fine_tune": True, "from": str(args.ckpt)},
    )
    print(f"Saved fine-tuned checkpoint to {out_path}")


if __name__ == "__main__":
    main()
