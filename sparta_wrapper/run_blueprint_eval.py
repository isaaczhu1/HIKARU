"""Evaluate a blueprint policy alone (no SPARTA) and report mean score."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch
    sys.path.insert(0, str(ROOT))

from sparta_wrapper.eval_sparta import SpartaConfig, evaluate
from sparta_wrapper.heuristic_blueprint import HeuristicBlueprint


def _get_blueprint_factory(name: str) -> Callable[[], object]:
    name = name.lower().strip()
    if name == "heuristic":
        return lambda: HeuristicBlueprint()
    raise ValueError(f"Unknown blueprint '{name}'. Currently supported: heuristic")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Hanabi blueprint (no SPARTA).")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to average.")
    parser.add_argument("--blueprint", type=str, default="heuristic", help="Blueprint name (default: heuristic).")
    parser.add_argument("--logdir", type=str, default="runs/blueprint_eval", help="TensorBoard log directory.")
    args = parser.parse_args()

    blueprint_factory = _get_blueprint_factory(args.blueprint)
    cfg = SpartaConfig(num_rollouts=1)  # unused in pure blueprint run, kept for API compatibility

    writer: SummaryWriter | None = None
    if args.logdir:
        writer = SummaryWriter(args.logdir)

    mean_score = evaluate(
        args.episodes,
        use_sparta=False,
        sparta_cfg=cfg,
        writer=writer,
        prefix=f"{args.blueprint}_baseline",
        blueprint_factory=blueprint_factory,
    )

    if writer is not None:
        writer.add_scalar("summary/mean", mean_score, 0)
        writer.flush()
        writer.close()

    print(f"{args.blueprint} mean over {args.episodes} episodes: {mean_score:.3f}")


if __name__ == "__main__":
    main()
