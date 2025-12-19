"""
Implements:

 - fabricate_history: construct a "falsified history" HanabiState object given by (actual current staate, guessing person, what they guessed)
 - get_consistent_hand_with_obs: given a HanabiObservation 
 
"""



from hanabi_learning_environment import pyhanabi
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG


def fabricate_history(state, guesser_id, guessed_hand):
    deck = []
    hands_tracker = []
    
    game = pyhanabi.HanabiGame
    for move in state.move_history():
        