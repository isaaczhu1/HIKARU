# config.py
class CFG:
    # general
    seed = 0
    out_dir = "runs/hanabi"
    num_envs = 64          # (optional) 96 if your CPU can handle it
    unroll_T = 128         # longer rollouts improve value/GAE stability
    obs_mode = "minimal"
    log_interval = 10
    save_interval = 500    # fewer, larger checkpoints for long runs
    total_updates = 10000

    class model:
        hidden = 256
        action_emb = 32
        seat_emb = 8
        include_prev_self = False

    class ppo:
        lr = 3e-4              # will decay in code (see below)
        clip = 0.2
        ent_coef = 0.05        # start high; we’ll anneal
        vf_coef = 0.5
        epochs = 4             # fine with bigger batch; can go 6–8 later if stable
        minibatch = 2048       # with N*T=16384 → 8 mini-batches per epoch
        gamma = 0.99
        gae_lambda = 0.95
        max_grad_norm = 0.5

    # New: simple schedules (used by train.py)
    class sched:
        lr_final = 3e-5        # linear decay end
        ent_final = 0.01       # entropy decay end
        ent_decay_until = 2000 # updates to reach ent_final
        eps0 = 0.02            # ε-greedy start
        eps_decay_until = 2000 # updates to reach 0
        target_kl = 0.01       # PPO early-stop threshold
