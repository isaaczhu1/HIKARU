HANABI_GAME_CONFIG = {
    "players": 2,
    "colors": 5,
    "ranks": 5,
    "hand_size": 5,
    "max_information_tokens": 8,
    "max_life_tokens": 3,
    "seed": 67,
}

DEVICE = "cuda"

CKPT_PATH = "gru_checkpoints/ckpt_020000.pt"

SPARTA_CONFIG = {
    "num_rollouts": 20,
    "t_threshold": 2,

    # number of moves to look at 
    "search_width": 3,

    # for rejection sampling
    "max_attempts": 100,
}

# Enable noisy debug prints when True.
DEBUG = False