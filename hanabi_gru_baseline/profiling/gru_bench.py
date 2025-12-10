#!/usr/bin/env python
"""
Simple forward-pass benchmark for Hanabi GRU policy.

Measures wall-clock time for N forward passes on random inputs that match the
shapes used in training/eval (batch x seq_len, obs_dim, num_moves).
"""

from __future__ import annotations

import argparse
import time

import torch

from hanabi_gru_baseline.model import HanabiGRUPolicy


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Hanabi GRU forward pass")
    p.add_argument("--device", type=str, default=None, help="cuda | cpu (auto if None)")
    p.add_argument("--gpu", action="store_true", help="force CUDA if available (overrides --device)")
    p.add_argument("--cpu", action="store_true", help="force CPU (overrides --device / --gpu)")
    p.add_argument("--batch", type=int, default=64, help="batch size (number of env slots)")
    p.add_argument("--seq-len", type=int, default=1, help="sequence length (timesteps)")
    p.add_argument("--obs-dim", type=int, default=658, help="observation vector size")
    p.add_argument("--num-moves", type=int, default=20, help="action count")
    p.add_argument("--hidden", type=int, default=256, help="GRU hidden size")
    p.add_argument("--action-emb", type=int, default=32, help="action embedding dim")
    p.add_argument("--seat-emb", type=int, default=8, help="seat embedding dim")
    p.add_argument("--include-prev-self", action="store_true", help="include prev self embedding path")
    p.add_argument("--n-queries", type=int, default=200, help="number of forward passes to time")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations (excluded from timing)")
    p.add_argument("--out", type=str, default="", help="optional path to write results (text)")
    return p.parse_args()


def main():
    args = parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.cpu:
        dev = "cpu"
    elif args.gpu:
        dev = "cuda"
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        dev = "cpu"

    model = HanabiGRUPolicy(
        obs_dim=args.obs_dim,
        num_moves=args.num_moves,
        hidden=args.hidden,
        action_emb_dim=args.action_emb,
        seat_emb_dim=args.seat_emb,
        include_prev_self=args.include_prev_self,
    ).to(dev)
    model.eval()

    B = args.batch
    L = args.seq_len
    obs = torch.randn(B, L, args.obs_dim, device=dev)
    seat = torch.randint(low=0, high=2, size=(B, L), device=dev)
    prev_other = torch.randint(low=0, high=args.num_moves + 1, size=(B, L), device=dev)
    prev_self = None
    if args.include_prev_self:
        prev_self = torch.randint(low=0, high=args.num_moves + 1, size=(B, L), device=dev)

    with torch.no_grad():
        h = model.initial_state(B, device=dev)

        # Warmup
        for _ in range(args.warmup):
            _ = model(obs, seat, prev_other, h, prev_self=prev_self)

        if dev.startswith("cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.n_queries):
            _ = model(obs, seat, prev_other, h, prev_self=prev_self)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    total = t1 - t0
    per = total / max(1, args.n_queries)
    qps = args.n_queries / total if total > 0 else float("inf")

    lines = [
        f"device={dev}",
        f"batch={B}, seq_len={L}, obs_dim={args.obs_dim}, num_moves={args.num_moves}",
        f"hidden={args.hidden}, action_emb={args.action_emb}, seat_emb={args.seat_emb}, include_prev_self={args.include_prev_self}",
        f"n_queries={args.n_queries}, warmup={args.warmup}",
        f"total_time={total:.6f}s, per_query={per*1e3:.3f} ms, queries_per_sec={qps:.1f}",
    ]
    out_text = "\n".join(lines)
    print(out_text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(out_text + "\n")


if __name__ == "__main__":
    main()
