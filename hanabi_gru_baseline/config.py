# config.py
class CFG:
    # general
    seed = 0
    out_dir = "runs/hanabi"
    num_envs = 64
    unroll_T = 128
    obs_mode = "minimal"
    log_interval = 10
    save_interval = 200
    total_updates = 2000

    class model:
        hidden = 256
        action_emb = 32
        seat_emb = 8
        include_prev_self = False

    class ppo:
        lr = 3e-4
        clip = 0.2
        ent_coef = 0.02
        vf_coef = 0.5
        epochs = 4
        minibatch = 2048
        gamma = 0.99
        gae_lambda = 0.95
        max_grad_norm = 0.5
