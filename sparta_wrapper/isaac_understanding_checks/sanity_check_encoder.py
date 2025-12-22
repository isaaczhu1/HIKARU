# sanity_check_encoder.py
import numpy as np
from hanabi_learning_environment import rl_env, pyhanabi

class _GameShim:
    def __init__(self, c_game): self._game = c_game
    @property
    def c_game(self): return self._game

def main(steps=2000, seed=0):
    env = rl_env.HanabiEnv(config={
        "players": 2,
        "colors": 5,
        "ranks": 5,
        "hand_size": 5,
        "max_information_tokens": 8,
        "max_life_tokens": 3,
        "random_start_player": False,
        "seed": seed,
        "observation_type": pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value,
    })

    enc_cache = {}
    obs = env.reset()
    mismatches = 0

    for t in range(steps):
        seat = obs["current_player"]
        pov = obs["player_observations"][seat]

        vec_rl = np.asarray(pov["vectorized"], dtype=np.int32)

        han_obs = pov["pyhanabi"]               # <-- pyhanabi.HanabiObservation
        game_ptr = han_obs._game                # underlying c_game pointer
        enc = enc_cache.get(game_ptr)
        if enc is None:
            enc = pyhanabi.ObservationEncoder(_GameShim(game_ptr),
                                              pyhanabi.ObservationEncoderType.CANONICAL)
            enc_cache[game_ptr] = enc

        vec_direct = np.asarray(enc.encode(han_obs), dtype=np.int32)

        if vec_rl.shape != vec_direct.shape or not np.array_equal(vec_rl, vec_direct):
            mismatches += 1
            # print one example then bail loudly
            print("Mismatch at step", t, "seat", seat,
                  "shapes", vec_rl.shape, vec_direct.shape,
                  "num_diff", int(np.sum(vec_rl != vec_direct)) if vec_rl.shape == vec_direct.shape else "N/A")
            raise RuntimeError("Canonical re-encode != rl_env vectorized")

        # take a random legal action so we advance
        legal = pov["legal_moves_as_int"]  # these are native move-uids
        obs, rew, done, info = env.step(int(legal[0]))
        if done:
            obs = env.reset()

    print("All good. Steps:", steps, "mismatches:", mismatches)

if __name__ == "__main__":
    main()
