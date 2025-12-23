import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hanabi_learning_environment import pyhanabi
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG
from sparta_wrapper.sparta_utils import *

def fabricate_test1():
    seeded_cfg = HANABI_GAME_CONFIG.copy()
    seeded_cfg["seed"] = 67
    
    game = pyhanabi.HanabiGame(seeded_cfg)
    state = game.new_initial_state()
    
    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        state.deal_random_card()
        
    print(state)
    
    choose = -2
    print(state.legal_moves()[choose])
    state.apply_move(state.legal_moves()[choose])
    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        state.deal_random_card()    
    print(state)

    choose = 4
    print(state.legal_moves()[choose])
    state.apply_move(state.legal_moves()[choose])
    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        state.deal_random_card()    
    print(state)

    choose = 6
    print(state.legal_moves()[choose])
    state.apply_move(state.legal_moves()[choose])
    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        state.deal_random_card()    
    print(state)
    
    print(state.card_count_from_perspective(2, 4, 0)) # should be 1
    print(state.card_count_from_perspective(2, 4, 1)) # should be 0
    
    guesser_id = 1
    guessed_hand_names = ["G3", "B3", "R3", "B1", "R5"]
    guessed_hand = [pyhanabi.HanabiCard(pyhanabi.color_char_to_idx(card[0]), int(card[1])-1) for card in guessed_hand_names]

    fabricated_history = fabricate_history(state, guesser_id, guessed_hand)
    for move in fabricated_history:
        print(move.to_dict())
        
    print(state)
    obs = state.observation(1)
    sample = consistent_hand_sampler(state, obs)
    for _ in range(100):
        print(sample())


if __name__ == "__main__":
    fabricate_test1()
    