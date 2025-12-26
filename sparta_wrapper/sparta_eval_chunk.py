# sparta_eval_chunk.py
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from hanabi_learning_environment import pyhanabi

# Ensure repo root importable
ROOT = Path(__file__).resolve().parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hanabi_gru_baseline.config import CFG as GRU_CFG
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG, SPARTA_CONFIG
from sparta_wrapper.gru_blueprint import GRUBlueprint
from sparta_wrapper.sparta import SpartaAgent
from sparta_wrapper.sparta_utils import SimulatedGame


def run_episode(seed: int, ckpt_path: Path, deviator: Optional[int], hanabi_cfg: dict) -> float:
    random.seed(seed)
    torch.manual_seed(seed)

    players = int(hanabi_cfg["players"])

    actor_blueprints = [
        (lambda: GRUBlueprint(ckpt_path, GRU_CFG, hanabi_cfg))
        for _ in range(players)
    ]

    if deviator is not None:
        actor_blueprints[deviator] = lambda: SpartaAgent(
            GRUBlueprint(ckpt_path, GRU_CFG, hanabi_cfg),
            hanabi_cfg,
            SPARTA_CONFIG,
        )

    game = SimulatedGame(
        history=[],
        set_deck=None,
        actor_blueprints=actor_blueprints,
        hanabi_game_config=hanabi_cfg,
    )

    while not game.terminal():
        game.step()

    return float(game.peak_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, required=True, help="Number of episodes in this chunk")
    ap.add_argument("--start-ep", type=int, default=0, help="Episode index offset for this chunk")
    ap.add_argument("--seed", type=int, default=0, help="Base seed; actual seed = seed + start_ep + i")
    ap.add_argument("--ckpt", type=str, required=True, help="GRU checkpoint path")
    ap.add_argument("--mode", type=str, default="both", choices=["baseline", "sparta", "both"])
    ap.add_argument("--deviator", type=int, default=0, help="Which player is SPARTA (when mode includes sparta)")
    ap.add_argument("--obs-type", type=int, default=None, help="Override pyhanabi observation_type int (0/1/2/3)")
    ap.add_argument("--out-json", type=str, default=None, help="Write results JSON here")
    ap.add_argument("--quiet", action="store_true", help="Less printing (recommended for Slurm)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build a per-run hanabi config (include seed!)
    hanabi_cfg = dict(HANABI_GAME_CONFIG)
    hanabi_cfg["seed"] = int(args.seed)  # base seed; per-episode we overwrite via new game creation below
    hanabi_cfg["random_start_player"] = False
    if args.obs_type is not None:
        hanabi_cfg["observation_type"] = int(args.obs_type)

    # Note: pyhanabi uses the game-config "seed" at construction. Since SimulatedGame constructs
    # a fresh HanabiGame each episode, we must pass the per-episode seed in hanabi_cfg each time.
    def make_cfg_for_ep(ep_seed: int) -> dict:
        d = dict(hanabi_cfg)
        d["seed"] = int(ep_seed)
        return d

    baseline_scores: List[float] = []
    sparta_scores: List[float] = []

    it = range(args.episodes)
    if not args.quiet:
        it = tqdm(it, desc="episodes")

    for i in it:
        ep_index = args.start_ep + i
        ep_seed = args.seed + ep_index

        if args.mode in ("baseline", "both"):
            new_score = run_episode(
                seed=ep_seed,
                ckpt_path=ckpt_path,
                deviator=None,
                hanabi_cfg=make_cfg_for_ep(ep_seed),
            )
            print(f"[ep {ep_index}] baseline score: {new_score}", flush=True)
            baseline_scores.append(new_score)

        if args.mode in ("sparta", "both"):
            new_score = run_episode(
                seed=ep_seed,
                ckpt_path=ckpt_path,
                deviator=args.deviator,
                hanabi_cfg=make_cfg_for_ep(ep_seed),
            )
            print(f"[ep {ep_index}] sparta score: {new_score}", flush=True)
            sparta_scores.append(new_score)

    out = {
        "ckpt": str(ckpt_path),
        "seed_base": int(args.seed),
        "start_ep": int(args.start_ep),
        "episodes": int(args.episodes),
        "mode": args.mode,
        "deviator": int(args.deviator),
        "hanabi_cfg": hanabi_cfg,
        "baseline_scores": baseline_scores,
        "sparta_scores": sparta_scores,
        "diffs": ([s - b for s, b in zip(sparta_scores, baseline_scores)]
                  if (args.mode == "both") else None),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }

    if args.out_json:
        p = Path(args.out_json).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        if not args.quiet:
            print(f"[write] {p}")

    # Always print a short summary
    def summarize(xs: List[float]) -> dict:
        if not xs:
            return {}
        xs = list(map(float, xs))
        return {"mean": sum(xs)/len(xs), "min": min(xs), "max": max(xs), "n": len(xs)}

    print("[summary] baseline:", summarize(baseline_scores))
    print("[summary] sparta:", summarize(sparta_scores))
    if out["diffs"] is not None:
        print("[summary] diffs:", summarize(out["diffs"]))


if __name__ == "__main__":
    main()
