from hanabi_learning_environment import pyhanabi

game_config = {
    'colors': 5,
    'ranks': 5,
    'players': 2,
    'hand_size': 5,            # 5 cards for 2–3 players in standard Hanabi
    'max_information_tokens': 8,
    'max_life_tokens': 3,
    # if your version supports it:
    # 'observation_type': pyhanabi.ObservationType.CARD_KNOWLEDGE,
}

game = pyhanabi.HanabiGame(game_config)
state = game.new_initial_state()

print("hi2")
while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
    print("hi3")
    state.deal_random_card()
