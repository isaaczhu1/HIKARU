"""
Implements:

 - fabricate_history: construct a "falsified history" HanabiState object given by (actual current staate, guessing person, what they guessed)
 - get_consistent_hand_with_obs: given a HanabiObservation 
 
"""

import random
import numpy as np
from hanabi_learning_environment import pyhanabi
from sparta_wrapper.hanabi_utils import unserialize, unmask_card
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG

def fabricate_history(state, guesser_id, guessed_hand):
    """
    precondition: 
        - guessed hand must be consistent with guesser_id's information
        - random starting player is FALSE
    """
    player_range = list(range(HANABI_GAME_CONFIG["players"]))
    hand_sz = HANABI_GAME_CONFIG["hand_size"]
    
    deck = []
    dealt_to = []
    deck_index = 0
    
    hands_tracker = [[] for _ in player_range]
    
    game = pyhanabi.HanabiGame
    for move in state.move_history():
        
        affected_player = move.player()
        if affected_player == pyhanabi.CHANCE_PLAYER_ID:
            
            # detect the card (hidden attribute, so requires jank to extract)
            card = unmask_card(move)
            
            # can be shown that the player to deal to is always the first player whose hand is incomplete
            # we're working under assumption of no random starting player.
            for deal_to in player_range:
                if len(hands_tracker[deal_to]) < hand_sz:
                    hands_tracker[deal_to].append(deck_index)
                    deck.append(card)
                    dealt_to.append(deal_to)
                    deck_index += 1
                    break
                
        elif move.move().type() in [pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD]:
            card_pos = move.move().card_index()
            hands_tracker[affected_player].pop(card_pos)
    
    # overwrite cards
    for index, card in zip(hands_tracker[guesser_id], guessed_hand):
        deck[index] = card
        
    deck_index = 0
    
    # overwrite history
    fabricated_history = []
    for move in state.move_history():
        
        affected_player = move.player()
        move_only = move.move()
        
        # serialize and modify
        serialized = move_only.to_dict()
        if affected_player == pyhanabi.CHANCE_PLAYER_ID:
            serialized["player"] = dealt_to[deck_index]
            serialized["color"] = deck[deck_index].color()
            serialized["rank"] = deck[deck_index].rank()
            deck_index += 1
            
        # hopefully lib-safe construction
        fabricated_history.append(unserialize(serialized))
        
    return fabricated_history

def remaining_deck(state, player, colors=HANABI_GAME_CONFIG["colors"], ranks=HANABI_GAME_CONFIG["ranks"]):
    cards = []
    for color in range(colors):
        for rank in range(ranks):
            counts = state.card_count_from_perspective(color=color, rank=rank, player=player)
            for _ in range(counts):
                cards.append(pyhanabi.HanabiCard(color=color, rank=rank))
    return cards

def consistent_hand_sampler(state, obs, colors=HANABI_GAME_CONFIG["colors"], ranks=HANABI_GAME_CONFIG["ranks"], hand_size=HANABI_GAME_CONFIG["hand_size"]):
    """
    get the sampler function
    """
    this_player = obs.get_player()
    # god knows why this thing is relative-indexed
    card_knowledge = obs.card_knowledge()[0]
    
    cards = remaining_deck(state=state, player=this_player, colors=colors, ranks=ranks)    
    slots = [[] for _ in range(hand_size)]
                
    for i, card in enumerate(cards):
        for j in range(hand_size):
            if card_knowledge[j].color_plausible(card.color()) and card_knowledge[j].rank_plausible(card.rank()):
                slots[j].append(i)
        
    def sample(hand_size=HANABI_GAME_CONFIG["hand_size"]):
        while True:
            sampled = [random.choice(slot) for slot in slots]
            if len(set(sampled)) == len(sampled):
                return [cards[i] for i in sampled]
    
    return sample

class SimulatedGame:
    def __init__(self, history, set_deck, actor_blueprints, hanabi_game_config=HANABI_GAME_CONFIG):
        self.history = history
        self.game = pyhanabi.HanabiGame(hanabi_game_config)
        self.state = self.game.new_initial_state()
        self.history_ptr = 0
        self.set_deck = set_deck
        self.deck_ptr = 0
        
        self.deal_to = None
        self.actors = [actor_blueprint() for actor_blueprint in actor_blueprints]
        self.peak_score = 0
        
        self.last_player_move_serialized = None
        
    def apply_move(self, move):
        if move.type() in [pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD]:
            self.deal_to = self.state.cur_player()   # acting player
        self.state.apply_move(move)
        self.peak_score = max(self.peak_score, self.state.score())

    
    # begin haram
    def exhausted_history(self):
        return self.history_ptr >= len(self.history)
    
    def read_next_history_item(self):
        assert not self.exhausted_history(), "bruh there is no more fabricated history"
        ret = self.history[self.history_ptr]
        self.history_ptr += 1
        return ret
    
    def exhausted_deck(self):
        return self.deck_ptr >= len(self.set_deck)
    
    def read_next_card(self):
        assert self.exhausted_history(), "you must consume all the fabricated history first"
        assert not self.exhausted_deck(), "please stop trolling"
        ret = self.set_deck[self.deck_ptr]
        self.deck_ptr += 1
        return ret
    # end haram
        
    def step(self, move_overwrite=None):
        """
        step forward. If move_overwrite is none, force that move instead.
        """
        cur_player = self.state.cur_player()
        
        # still in replay
        if not self.exhausted_history():
            # advance the hidden state if needed
            if cur_player != pyhanabi.CHANCE_PLAYER_ID:
                obs = self.state.observation(self.state.cur_player())
                act = self.actors[cur_player].act(obs, self.state)
                self.last_player_move_serialized = act.to_dict()
            self.apply_move(self.read_next_history_item() if move_overwrite is None else move_overwrite)
            
        # out of replay
        else:
            if cur_player != pyhanabi.CHANCE_PLAYER_ID:
                obs = self.state.observation(self.state.cur_player())
                move = self.actors[cur_player].act(obs, self.state)
                self.apply_move(move if move_overwrite is None else move_overwrite)
            else:
                if self.set_deck is None:
                    self.state.deal_random_card()
                elif not self.exhausted_deck():
                    card = self.read_next_card()
                    self.state.deal_specific_card(player_id=self.deal_to, color=card.color(), rank=card.rank())
                    
    def terminal(self):
        return self.state.is_terminal()
    
def check_consistent_with_partner_move(fabricated_history, actor_blueprints, partner_last_move):
    """
    Check if the fabricated history is consistent with the partner's last move.
    """
    game = SimulatedGame(history=fabricated_history, set_deck=[], actor_blueprints=actor_blueprints)
    while not game.exhausted_history():
        game.step()
    if partner_last_move is None:
        return game.last_player_move_serialized is None
    return game.last_player_move_serialized == partner_last_move.to_dict()

def sample(state, obs, actor_blueprint, num_samples, max_attempts, hanabi_cfg=HANABI_GAME_CONFIG):
    """
    Sample a hand for the guessing player that is consistent with their observation
    """
    guesser_id = obs.get_player()
    sampler = consistent_hand_sampler(state=state, obs=obs)
    
    # get last move
    partner_last_move = None
    for item in obs.last_moves():
        pid = item.player()
        move_dict = item.move().to_dict()
        if pid is None or move_dict is None:
            continue
        if pid == 0 or pid == pyhanabi.CHANCE_PLAYER_ID:
            continue
        partner_last_move = item.move()
        break
    
    # rejection sampling
    accepted_hands = []
    
    for _ in range(max_attempts):
        guessed_hand = sampler()
        if check_consistent_with_partner_move(
            fabricated_history=fabricate_history(state=state, guesser_id=guesser_id, guessed_hand=guessed_hand),
            actor_blueprints=[actor_blueprint for _ in range(hanabi_cfg["players"])],
            partner_last_move=partner_last_move
        ):
            accepted_hands.append(guessed_hand)
            if len(accepted_hands) >= num_samples:
                break
    
    while len(accepted_hands) < num_samples:
        accepted_hands.append(sampler())
    
    return accepted_hands

def t(a, b, eps=0.01):
    diffs = np.array(a) - np.array(b)
    return np.mean(diffs) / (np.std(diffs) + eps) * np.sqrt(diffs.shape[0]) 