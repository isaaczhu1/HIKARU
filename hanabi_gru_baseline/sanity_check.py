# eval_random_2x2.py
import numpy as np
from hanabi_envs import HanabiEnv2P


def main():
    env = HanabiEnv2P(
        seed=123,
        players=2,
        colors=2,
        ranks=2,
        hand_size=2,
        max_information_tokens=8,
        max_life_tokens=3,
        random_start_player=False,
    )

    num_games = 100
    scores = []
    for g in range(num_games):
        obs = env.reset()
        done = False
        total_rew = 0.0
        while not done:
            legal = np.where(obs["legal_mask"] > 0)[0]
            a = int(np.random.choice(legal))
            obs, r, done, info = env.step(a)
            total_rew += r
        score = info.get("score", None)
        print(f"Game {g}: score={score}, total_rew={total_rew}")
        scores.append(score if score is not None else 0.0)

    print("Scores:", scores)
    print("Mean score:", np.mean(scores))


if __name__ == "__main__":
    main()
