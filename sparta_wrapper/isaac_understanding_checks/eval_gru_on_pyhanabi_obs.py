# eval_gru_on_pyhanabi_obs.py

import numpy as np
from hanabi_learning_environment import rl_env, pyhanabi

def eval_blueprint(blueprint, n_games=200, seed0=0):
    scores = []
    for g in range(n_games):
        env = rl_env.HanabiEnv(config={
            "players": 2,
            "colors": 5,
            "ranks": 5,
            "hand_size": 5,
            "max_information_tokens": 8,
            "max_life_tokens": 3,
            "random_start_player": False,
            "seed": seed0 + g, # increment seed by 1 every game
            "observation_type": pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value,
        })
        obs = env.reset()
        blueprint.reset_episode()

        total = 0.0
        done = False
        while not done:
            seat = obs["current_player"]
            pov = obs["player_observations"][seat]
            han_obs = pov["pyhanabi"]

            move = blueprint.act(han_obs) # this is a pyhanabi.HanabiMove
            uid = env.game.get_move_uid(move) # this is a native move-uid int
            obs, rew, done, info = env.step(int(uid))
            total += float(rew) # rl_env reward telescopes to final score

        scores.append(total)

    scores = np.asarray(scores, dtype=np.float32)
    print(f"n={n_games}  mean={scores.mean():.3f}  std={scores.std():.3f}  min={scores.min():.1f}  max={scores.max():.1f}")
    return scores
