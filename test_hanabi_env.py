from hanabi_learning_environment import rl_env


def main() -> None:
    env = rl_env.HanabiEnv({"players": 2})
    obs = env.reset()
    done = False

    print("=== New game ===")
    step = 0
    while not done:
        cur_player = obs["current_player"]
        player_obs = obs["player_observations"][cur_player]

        print(f"\nStep {step}, Player {cur_player}'s turn")
        print("Fireworks:", player_obs.get("fireworks"))
        print("Info tokens:", player_obs.get("information_tokens"))
        print("Life tokens:", player_obs.get("life_tokens"))

        state = env.state
        print("All hands (true state):")
        for p, hand in enumerate(state.player_hands()):
            print(f"  Player {p}: {hand}")

        print("Legal moves:")
        for m in player_obs["legal_moves"]:
            print(" ", m)

        action = player_obs["legal_moves"][0]
        print("Taking action:", action)

        obs, reward, done, _ = env.step(action)
        print("Reward:", reward)
        step += 1


if __name__ == "__main__":
    main()
