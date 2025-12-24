import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hanabi_learning_environment import pyhanabi
from hanabi_gru_baseline.config import CFG as GRU_CFG
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG
from sparta_wrapper.sparta_utils import *

from sparta_wrapper.gru_blueprint import GRUBlueprint

import torch

class SignalBlueprint:
    """Signal by looking at the next player's first card."""

    def act(self, observation: pyhanabi.HanabiObservation) -> pyhanabi.HanabiMove:
        partner_hand = observation.observed_hands()[1]
        if not partner_hand:
            return observation.legal_moves()[0]
        partner_rank = partner_hand[0].rank()
        if partner_rank == 0:
            return self._pick_move(observation, pyhanabi.HanabiMoveType.PLAY)
        elif partner_rank > 0:
            return self._pick_move(observation, pyhanabi.HanabiMoveType.REVEAL_COLOR)
        return observation.legal_moves()[0]

    def _pick_move(self, observation: pyhanabi.HanabiObservation, move_type: pyhanabi.HanabiMoveType) -> pyhanabi.HanabiMove:
        for move in observation.legal_moves():
            if move.type() == move_type:
                return move
        return observation.legal_moves()[0]

def signal_blueprint_factory() -> SignalBlueprint:
    return SignalBlueprint()

def squid_game(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    cfg = HANABI_GAME_CONFIG
    cfg['seed'] = seed
    
    game = pyhanabi.HanabiGame(cfg)
    state = game.new_initial_state()
    
    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        state.deal_random_card()
        
    print(state)
    
    act = signal_blueprint_factory().act(state.observation(state.cur_player()))
    print(act)
    
    state.apply_move(act)
    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        state.deal_random_card()
        
    print(state)
    
    print("checking consistency with fabricated history...")
    
    sampler = consistent_hand_sampler(state, state.observation(1))
    samples = [sampler() for _ in range(10)]
    
    for hand_guess in samples:
        is_consistent = check_consistent_with_partner_move(
            fabricate_history(state, 1, hand_guess),
            signal_blueprint_factory,
            act
        )
        print(f"Guessed hand: {[str(card) for card in hand_guess]}, Consistent with partner's move: {is_consistent}")
        
    for hand in sample(state, state.observation(1), signal_blueprint_factory, act, num_samples=5, max_attempts=50):
        print(hand) 

if __name__ == "__main__":
    squid_game(67)
    