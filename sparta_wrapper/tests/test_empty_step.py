import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hanabi_learning_environment import pyhanabi
from hanabi_gru_baseline.config import CFG as GRU_CFG
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG
from sparta_wrapper.sparta_utils import *

from sparta_wrapper.gru_blueprint import GRUBlueprint

print("im gonna take five years to import torch cause im a bot")
import torch
print("done pooping")

def evaluate_gru_blueprint(ckpt_path, episodes):
    def blueprint_factory():
        return GRUBlueprint(ckpt_path=ckpt_path, model_cfg=GRU_CFG, hanabi_cfg=HANABI_GAME_CONFIG)

    scores = []

    from tqdm import tqdm
    for _ in tqdm(range(episodes)):
        game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
        state = game.new_initial_state()
        
        # skip initial deal phase
        while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            state.deal_random_card()
        
        p0_guess = consistent_hand_sampler(state=state, obs=state.observation(0))()
        
        print(p0_guess)
        
        fabricated_history = fabricate_history(state, 0, p0_guess)
        
        print(fabricated_history)
        
        set_deck = remaining_deck(state)
        random.shuffle(set_deck)
        
        simulator = SimulatedGame(fabricated_history, set_deck, blueprint_factory)
        
        while not simulator.terminal():
            simulator.step()
            
        scores.append(simulator.peak_score)

        
    print(sum(scores) / len(scores))
    print(scores)
    

    
if __name__ == "__main__":
    random.seed(67)
    torch.manual_seed(67)
    evaluate_gru_blueprint("gru_checkpoints/ckpt_020000.pt", 20)