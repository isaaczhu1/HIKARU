HANABI_GAME_CONFIG = {
    "players": 2,
    "colors": 1,
    "ranks": 5,
    "hand_size": 3,
    "max_information_tokens": 8,
    "max_life_tokens": 3,
}

DEVICE = "cuda"

CKPT_PATH = "runs/hanabi/standard_train/20251210_000611/ckpt_001000.pt"

SPARTA_CONFIG = {
    "num_rollouts": 16,
    "epsilon": 0.05,
    "rng_seed": None,

    # for rejection sampling
    "upstream_factor": 5,
    "max_attempts": 100,
}
