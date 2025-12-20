"""
Implements:

 - fabricate_history: construct a "falsified history" HanabiState object given by (actual current staate, guessing person, what they guessed)
 - get_consistent_hand_with_obs: given a HanabiObservation 
 
"""



from hanabi_learning_environment import pyhanabi
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG

def unmask_card(move):
    """
    move has a representation of the form <(Deal XY)> where X is color and Y is suit.
    Recover the color and suit, and map them back to the numerical values they are assigned in pyhanabi.
    """
    if isinstance(move, pyhanabi.HanabiMove):
        move = str(move)
    s = str(move).strip()
    # Expect formats like "<(Deal W4)>" or "(Deal W4)"
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1]
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]

    parts = s.split()
    if len(parts) != 2 or parts[0].lower() != "deal":
        raise ValueError(f"Unsupported move format for unmask_card: {move}")

    card = parts[1]
    if len(card) != 2:
        raise ValueError(f"Unsupported card token in move: {move}")

    color_char = card[0]
    rank_char = card[1]

    color = pyhanabi.color_char_to_idx(color_char)
    rank = int(rank_char) - 1
    return pyhanabi.HanabiCard(color=color, rank=rank)

def unserialize_move(serialized):
    """
    unserialize a pyhanabi move
    """
    method_map = {
        pyhanabi.HanabiMoveType.INVALID.name : (lambda: raise ValueError("cannot unserialize invalid move")),
        pyhanabi.HanabiMoveType.PLAY.name : pyhanabi.HanabiMove.get_play_move
        pyhanabi.HanabiMoveType.DISCARD.name : pyhanabi.HanabiMove.get_discard_move
        pyhanabi.HanabiMoveType.REVEAL_COLOR.name : pyhanabi.HanabiMove.get_reveal_color_move
        pyhanabi.HanabiMoveType.REVEAL_RANK.name : pyhanabi.HanabiMove.get_reveal_rank_move
        pyhanabi.HanabiMoveType.DEAL.name : pyhanabi.HanabiMove.get_deal_move
        pyhanabi.HanabiMoveType.RETURN.name : pyhanabi.HanabiMove.get_return_move
        pyhanabi.HanabiMoveType.DEAL_SPECIFIC.name : pyhanabi.HanabiMove.get_deal_specific_move
    }
    
    return method_map[serialized["action_type"]](**serialized)

def fabricate_history(state, guesser_id, guessed_hand):
    """
    precondition: 
        - guessed hand must be consistent with guesser_id's information
        - random starting player is FALSE
    """
    player_range = list(range(HANABI_GAME_CONFIG["players"]))
    hand_sz = HANABI_GAME_CONFIG["hand_size"]
    
    deck = []
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
                    deck_index += 1
                    break
                
        elif move.move().type() in [pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD]:
            card_pos = move.move().card_index()
            deck_tracking[affected_player].pop(card_pos)
    
    # overwrite cards
    for index, card in zip(hands_tracker[guesser_id], guessed_hand):
        deck[index] = card
        
    deck_index = 0
    
    # overwrite history
    fabricated_history = []
    for move in state.move_history():
        affected_player = move.player()
        
        # serialize and modify
        serialized = move.to_dict()
        if affected_player == pyhanabi.CHANCE_PLAYER_ID:
            serialized["action_type"] = "DEAL_SPECIFIC"
            serialized["color"] = deck[deck_index].color()
            serialized["rank"] = deck[deck_index].rank()
            
        # hopefully lib-safe construction
        fabricated_history.append(unserialize(serialized))
        
    return fabricated_history