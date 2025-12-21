"""
Implements:

 - fabricate_history: construct a "falsified history" HanabiState object given by (actual current staate, guessing person, what they guessed)
 - get_consistent_hand_with_obs: given a HanabiObservation 
 
"""



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