import time
import numpy as np

from mini_env import TinyHanabi2x2, NUM_MOVES, SENTINEL_NONE


def main():
    env = TinyHanabi2x2(seed=123)
    obs, _ = env.reset()
    obs_dim = obs["obs"].shape[0]
    print(f"obs_dim={obs_dim}, num_moves={NUM_MOVES}")

    ep_ret = 0.0
    steps = 0
    t0 = time.time()

    # Roll a few random-legal steps to verify masking, prev_other_action, and basic dynamics
    for _ in range(50):
        legal_idxs = np.where(obs["legal_mask"] > 0)[0]
        a = int(np.random.choice(legal_idxs))

        prev = int(obs["prev_other_action"])
        seat = int(obs["seat"])

        obs, r, term, trunc, info = env.step(a)
        steps += 1
        ep_ret += r

        prev_str = prev if prev != SENTINEL_NONE else "NONE"
        print(
            f"step={steps:02d} seat={seat} prev_other={prev_str} "
            f"act={a} legal={len(legal_idxs)} r={r:.1f} "
            f"score={info['score']} done={term or trunc}"
        )

        if term or trunc:
            print(f"episode_end: return={ep_ret}")
            ep_ret = 0.0
            obs, _ = env.reset()

    dt = time.time() - t0
    print(f"Ran {steps} steps in {dt:.3f}s")


if __name__ == "__main__":
    main()
