from hanabi_learning_environment import pyhanabi
from sparta_wrapper.sparta_config import *
from sparta_wrapper.sparta_utils import *

class SpartaAgent:
    def __init__(self, blueprint, hanabi_cfg=HANABI_GAME_CONFIG, sparta_config=SPARTA_CONFIG):
        self.blueprint = blueprint
        self.hanabi_cfg = hanabi_cfg
        self.sparta_config = sparta_config

    def act(self, obs: pyhanabi.HanabiObservation, state: pyhanabi.HanabiState) -> pyhanabi.HanabiMove:
        blueprint_move = self.blueprint.act(obs)
        legal_moves = obs.legal_moves()
        legal_moves.remove(blueprint_move)

        alternate_choices = random.sample(legal_moves, min(self.sparta_config["search_width"], len(legal_moves)))
        choices = alternate_choices + [blueprint_move]
        
        samples = sample(
            state=state,
            obs=obs,
            actor_blueprint=lambda: self.blueprint,
            num_samples=self.sparta_config["num_rollouts"],
            max_attempts=self.sparta_config["max_attempts"],
        )
        
        set_decks = [remaining_deck(state, obs.get_player()) for _ in samples]
        for deck, hand in zip(set_decks, samples):
            for card in hand:
                deck.remove(card)
            deck = random.shuffle(deck)
            
        evaluations = {move : _evaluate_move(state, obs, samples, set_decks, move) for move in choices}
        
        to_act = blueprint_move
        for move in alternate_choices:
            if t(evaluations[move], evaluations[blueprint_move]) > self.sparta_config["t_threshold"]:
                to_act = move
        
        return to_act

    def _evaluate_move(self, state, obs, samples, set_decks, move):
        rewards = []
        
        for sample_hand, deck in zip(samples, set_decks):
            game = SimulatedGame(
                history=fabricate_history(
                    state=state,
                    guesser_id=obs.get_player(),
                    guessed_hand=sample_hand
                ),
                set_deck=deck,
                actor_blueprint=lambda: self.blueprint,
                hanabi_game_config=self.hanabi_cfg
            )
            
            # advance fabricated steps
            while not game.exhausted_history():
                game.step()
                
            # apply the given move
            game.step(move_overwrite=move)
            
            # continue to end
            while not game.terminal():
                game.step()
                
            rewards.append(game.peak_score)

        return rewards