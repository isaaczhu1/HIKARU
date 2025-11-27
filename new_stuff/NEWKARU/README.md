# Hanabi SPARTA-Lite

This repo hosts a lightweight implementation of SPARTA-style decision-time search on top of Hanabi blueprint policies. It follows the two-track plan in `PLAN.md`: a tiny public-belief toy game for theory experiments and a full 2-player Hanabi stack for empirical testing.

## Python Environment

- Conda env: `hanabi`
- Core deps: `hanabi_learning_environment`, `numpy`, `torch`, `matplotlib` (install the rest as needed per component)

## Repo Layout (initial draft)

```
envs/           # Wrappers around DeepMind's Hanabi env + tiny-game env
blueprints/    # Hand-coded heuristic and future GRU/LSTM policies
search/        # Belief models and SPARTA wrappers
experiments/   # Entry points for running baselines and SPARTA evaluations
utils/         # Shared utilities (logging, plotting, helpers)
tests/         # Smoke tests (e.g., environment import test)
```

## Immediate Build Plan

1. Implement `envs/full_hanabi_env.py` as a thin wrapper around `hanabi_learning_environment` (2p config, cloning helpers).
2. Add `blueprints/heuristic_blueprint.py` with a simple deterministic policy (play-safe → hint → discard).
3. Introduce `search/belief_models.py` (world state representation + approximate sampler).
4. Implement `search/sparta_single.py` that wraps a blueprint with 1-ply search.
5. Create experiment scripts (`experiments/baseline_full.py`, `experiments/search_full.py`) and mirrored tiny-game versions.
6. Extend `tests/` with deterministic smoke tests for env + blueprint + SPARTA rollouts.

This structure keeps the heuristic baseline and SPARTA wrapper isolated from later learned blueprints while allowing reuse of env + belief code across both project tracks.
