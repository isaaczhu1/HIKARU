"""Small driver to compare heuristic blueprint vs SPARTA wrapper."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Callable, List

from hanabi_learning_environment import rl_env
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch
    sys.path.insert(0, str(ROOT))

from sparta_wrapper.hanabi_utils import _move_to_action_dict, build_observation
from sparta_wrapper.heuristic_blueprint import HeuristicBlueprint
from sparta_wrapper.sparta_search import SpartaConfig, SpartaSingleAgent


def _run_episode(env: rl_env.HanabiEnv, agents: List[Callable]) -> float:
    env.reset()
    state = env.state

    while not state.is_terminal():
        pid = state.cur_player()
        player_obs = build_observation(state, pid)
        action_move = agents[pid](player_obs, state)
        action_dict = _move_to_action_dict(action_move)
        _, _, done, info = env.step(action_dict)
        state = env.state
        if done:
            return float(info.get("score", state.score()))
    return float(state.score())


def evaluate(
    episodes: int,
    use_sparta: bool,
    sparta_cfg: SpartaConfig,
    writer: SummaryWriter | None,
    prefix: str,
    blueprint_factory: Callable[[], object] | None = None,
) -> float:
    env = rl_env.HanabiEnv({"players": 2})
    scores = []
    blueprint_factory = blueprint_factory or (lambda: HeuristicBlueprint())

    for _ in range(episodes):
        if use_sparta:
            search = SpartaSingleAgent(blueprint_factory, config=sparta_cfg)

            def _sparta_actor(player_obs, _state):
                return search.act(_state, player_obs.player_id)

            agents = [_sparta_actor, _sparta_actor]
        else:
            blueprints = [blueprint_factory(), blueprint_factory()]

            def _make_actor(bp):
                return lambda obs, _state: bp.act(obs)

            agents = [_make_actor(blueprints[0]), _make_actor(blueprints[1])]

        score = _run_episode(env, agents)
        scores.append(score)
        if writer is not None:
            step = len(scores)
            writer.add_scalar(f"{prefix}/score", score, step)
            writer.add_scalar(f"{prefix}/running_mean", sum(scores) / len(scores), step)
    return sum(scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate heuristic blueprint vs SPARTA wrapper.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes for SPARTA (and blueprint if enabled).")
    parser.add_argument("--rollouts", type=int, default=24, help="Rollouts per action.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Improvement threshold.")
    parser.add_argument("--logdir", type=str, default="runs/sparta_eval", help="TensorBoard log directory.")
    parser.add_argument("--blueprint", type=str, default="heuristic", help="Blueprint to wrap (currently supports 'heuristic').")
    parser.add_argument("--skip-blueprint", action="store_true", help="Skip standalone blueprint eval to save time.")
    args = parser.parse_args()

    sparta_cfg = SpartaConfig(num_rollouts=args.rollouts, epsilon=args.epsilon)

    logdir = None
    writer: SummaryWriter | None = None
    if args.logdir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logdir = Path(args.logdir) / timestamp
        logdir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(logdir))

    # Blueprint selection placeholder; extendable later.
    blueprint_factory = lambda: HeuristicBlueprint()

    base_mean = None
    if not args.skip_blueprint:
        base_mean = evaluate(
            args.episodes,
            use_sparta=False,
            sparta_cfg=sparta_cfg,
            writer=writer,
            prefix="blueprint",
            blueprint_factory=blueprint_factory,
        )
    sparta_mean = evaluate(
        args.episodes,
        use_sparta=True,
        sparta_cfg=sparta_cfg,
        writer=writer,
        prefix="sparta",
        blueprint_factory=blueprint_factory,
    )

    if writer is not None:
        if base_mean is not None:
            writer.add_scalar("summary/blueprint_mean", base_mean, 0)
        writer.add_scalar("summary/sparta_mean", sparta_mean, 0)
        writer.flush()
        writer.close()

    if base_mean is not None:
        print(f"Baseline heuristic mean over {args.episodes} episodes: {base_mean:.3f}")
    print(f"Heuristic+SPARTA mean over {args.episodes} episodes: {sparta_mean:.3f}")


if __name__ == "__main__":
    main()
