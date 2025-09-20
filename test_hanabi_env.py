from hanabi_learning_environment import rl_env

# Make a 2-player Hanabi game
env = rl_env.HanabiEnv({"players": 2})

obs = env.reset()
done = False

print("=== New game ===")
step = 0
while not done and step < 5:  # only show a few steps
    cur_player = obs["current_player"]
    player_obs = obs["player_observations"][cur_player]

    print(f"\nStep {step}, Player {cur_player}'s turn")
    print("Fireworks:", player_obs["fireworks"])
    print("Information tokens:", player_obs["information_tokens"])
    print("Life tokens:", player_obs["life_tokens"])
    print("Observed hands:", player_obs["observed_hands"])

    # Show all legal moves
    print("Legal moves (dict form):")
    for m in player_obs["legal_moves"]:
        print(" ", m)
    print("Legal moves (as ints):", player_obs["legal_moves_as_int"])

    # For demo: just take the first legal move
    action = player_obs["legal_moves"][0]
    print("Taking action:", action)

    obs, reward, done, _ = env.step(action)
    print("Reward from action:", reward)
    step += 1
