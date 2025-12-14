"""Shared Hanabi helpers for SPARTA and heuristic blueprints."""

from __future__ import annotations

import random
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from hanabi_learning_environment import pyhanabi
from sparta_wrapper.sparta_config import DEBUG, HANABI_GAME_CONFIG

def _debug(msg: str) -> None:
    if DEBUG:
        print(msg, flush=True)

def _reprcard(card):
    if isinstance(card, pyhanabi.HanabiCard):
        return pyhanabi.COLOR_CHAR[card.color()] + str(card.rank() + 1)
    else:
        return pyhanabi.COLOR_CHAR[card[0]] + str(card[1] + 1)

def _reprhand(hand):
    return [_reprcard(card) for card in hand]


def _card_to_dict(card: pyhanabi.HanabiCard) -> Dict[str, Any]:
    return {
        "color": pyhanabi.COLOR_CHAR[card.color()] if card.color() >= 0 else None,
        "rank": card.rank(),
    }


def _hand_to_dict(hand: Sequence[pyhanabi.HanabiCard]) -> List[Dict[str, Any]]:
    return [_card_to_dict(card) for card in hand]


def _fireworks_to_dict(fireworks: Sequence[int]) -> Dict[str, int]:
    colors = pyhanabi.COLOR_CHAR[: len(fireworks)]
    return {color: level for color, level in zip(colors, fireworks)}


def _history_item_to_dict(item: pyhanabi.HanabiHistoryItem) -> Dict[str, Any]:
    payload = {
        "player": item.player(),
        "move": move_to_dict(item.move()),
        "scored": item.scored(),
        "information_token": item.information_token(),
        "color": pyhanabi.COLOR_CHAR[item.color()] if item.color() >= 0 else None,
        "rank": item.rank(),
        "card_info_revealed": item.card_info_revealed(),
        "card_info_newly_revealed": item.card_info_newly_revealed(),
    }
    return payload


def _knowledge_to_dict(
    knowledge: pyhanabi.HanabiCardKnowledge, num_colors: int = 5, num_ranks: int = 5
) -> Dict[str, Any]:
    color_idx = knowledge.color()
    rank_idx = knowledge.rank()
    color = pyhanabi.COLOR_CHAR[color_idx] if isinstance(color_idx, int) else None
    rank = int(rank_idx) if isinstance(rank_idx, int) else None
    mask = [
        [bool(knowledge.color_plausible(c) and knowledge.rank_plausible(r)) for r in range(num_ranks)]
        for c in range(num_colors)
    ]
    return {"color": color, "rank": rank, "mask": mask}

def _coerce_color(color: Any) -> int:
    if isinstance(color, str):
        return pyhanabi.color_char_to_idx(color)
    return int(color)


def move_to_dict(move: pyhanabi.HanabiMove) -> Dict[str, Any]:
    """Serialize a HanabiMove into a dict understood by rl_env and tests."""
    if not isinstance(move, pyhanabi.HanabiMove):
        raise TypeError(f"Expected HanabiMove, got {type(move)}")

    move_type = move.type()
    payload: Dict[str, Any] = {"action_type": move_type.name, "type": move_type.name}

    if move_type in (pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD):
        payload["card_index"] = move.card_index()
    elif move_type == pyhanabi.HanabiMoveType.REVEAL_COLOR:
        payload["target_offset"] = move.target_offset()
        color_idx = move.color()
        payload["color"] = pyhanabi.COLOR_CHAR[color_idx] if color_idx >= 0 else color_idx
    elif move_type == pyhanabi.HanabiMoveType.REVEAL_RANK:
        payload["target_offset"] = move.target_offset()
        payload["rank"] = move.rank()
    elif move_type == pyhanabi.HanabiMoveType.DEAL:
        payload["target_offset"] = move.target_offset()
        color_idx = move.color()
        payload["color"] = pyhanabi.COLOR_CHAR[color_idx] if color_idx >= 0 else color_idx
        payload["rank"] = move.rank()
    elif move_type == pyhanabi.HanabiMoveType.DEAL_SPECIFIC:
        payload["card_index"] = move.card_index()
        payload["target_offset"] = move.target_offset()
        payload["color"] = move.color()
        payload["rank"] = move.rank()
    elif move_type == pyhanabi.HanabiMoveType.RETURN:
        payload["card_index"] = move.card_index()
        payload["target_offset"] = move.target_offset()
    else:
        raise ValueError(f"Unsupported move for serialization: {move_type}")

    return payload


def dict_to_move(game: pyhanabi.HanabiGame | None, spec: Dict[str, Any]) -> pyhanabi.HanabiMove:
    """Convert an action/move dict into a fresh HanabiMove."""
    move_type_name = spec.get("type", spec.get("action_type"))
    if move_type_name is None:
        raise ValueError(f"Missing move type in spec: {spec}")
    try:
        mtype = pyhanabi.HanabiMoveType[move_type_name.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown move type: {spec}") from exc

    game = game or pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
    num_colors = game.num_colors()
    num_ranks = game.num_ranks()

    if mtype == pyhanabi.HanabiMoveType.PLAY:
        return pyhanabi.HanabiMove.get_play_move(spec["card_index"])
    if mtype == pyhanabi.HanabiMoveType.DISCARD:
        return pyhanabi.HanabiMove.get_discard_move(spec["card_index"])
    if mtype == pyhanabi.HanabiMoveType.REVEAL_COLOR:
        color = _coerce_color(spec["color"])
        if color < 0 or color >= num_colors:
            raise ValueError(f"Reveal color out of range: {color}")
        return pyhanabi.HanabiMove.get_reveal_color_move(spec["target_offset"], color)
    if mtype == pyhanabi.HanabiMoveType.REVEAL_RANK:
        rank = spec["rank"]
        if rank < 0 or rank >= num_ranks:
            raise ValueError(f"Reveal rank out of range: {rank}")
        return pyhanabi.HanabiMove.get_reveal_rank_move(spec["target_offset"], rank)
    if mtype == pyhanabi.HanabiMoveType.DEAL_SPECIFIC:
        return pyhanabi.HanabiMove.get_deal_specific_move(
            spec.get("card_index"), spec["target_offset"], spec["color"], spec["rank"]
        )
    if mtype == pyhanabi.HanabiMoveType.RETURN:
        return pyhanabi.HanabiMove.get_return_move(spec["card_index"], spec["target_offset"])
    raise ValueError(f"Unsupported move type {move_type_name}")

def _clone_move_for_state(move: Any) -> pyhanabi.HanabiMove:
    """
    Build a fresh HanabiMove detached from any original state.

    Accepts an existing HanabiMove or a move/action dict (with type/action_type).
    Raises on unsupported inputs.
    """
    if isinstance(move, pyhanabi.HanabiMove):
        spec = move_to_dict(move)
    elif isinstance(move, dict):
        spec = dict(move)
        if "type" not in spec and "action_type" in spec:
            spec["type"] = spec["action_type"]
    else:
        raise ValueError(f"Unsupported move type for cloning: {move!r}")
    return dict_to_move(None, spec)


def apply_move_safe(state: pyhanabi.HanabiState, move: Any) -> None:
    """
    Apply a move to state, cloning/validating to avoid cross-state corruption.

    Supports:
      - pyhanabi.HanabiMove or action dict (applied via _clone_move_for_state)
      - (player, color, rank) tuples/lists for DEAL events (applied via deal_specific_card)
    """
    if isinstance(move, (list, tuple)) and len(move) == 3 and not isinstance(move, pyhanabi.HanabiMove):
        _debug(f"Trying deal {move}")
        state.deal_specific_card(*move)
        return
    cloned = _clone_move_for_state(move)
    _debug(str(state))
    _debug(f"cloned move to  {move}")
    state.apply_move(cloned)


def apply_move_to_state(state: pyhanabi.HanabiState, move_spec: Dict[str, Any]) -> None:
    move = _clone_move_for_state(move_spec)
    state.apply_move(move)


def _advance_chance_events(state: pyhanabi.HanabiState, deck_override: list[pyhanabi.HanabiCard] | None = None) -> None:
    """Advance chance nodes (card draws) until a player must act.

    If ``deck_override`` is provided, each dealt card is overwritten with the
    next card from that list, ensuring rollouts follow the sampled deck order.
    """
    while not state.is_terminal() and state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        before = [[(c.color(), c.rank()) for c in hand] for hand in state.player_hands()]
        state.deal_random_card()

@dataclass
class HanabiObservation:
    """Structured view of a player's observation and legal moves."""

    player_id: int
    current_player: int
    current_player_offset: int
    observed_hands: List[List[Dict[str, Any]]]
    card_knowledge: List[List[Dict[str, Any]]]
    discard_pile: List[Dict[str, Any]]
    fireworks: Dict[str, int]
    deck_size: int
    information_tokens: int
    life_tokens: int
    raw_observation: pyhanabi.HanabiObservation
    legal_moves: Sequence[pyhanabi.HanabiMove]
    legal_moves_dict: List[Dict[str, Any]]
    last_moves: List[Dict[str, Any]]

    def legal_move_dicts(self) -> List[Dict[str, Any]]:
        return list(self.legal_moves_dict)


def build_observation(state: pyhanabi.HanabiState, player_id: int) -> HanabiObservation:
    """Construct a structured observation for ``player_id`` from a state."""
    obs = state.observation(player_id)
    num_colors = HANABI_GAME_CONFIG["colors"]
    num_ranks = HANABI_GAME_CONFIG["ranks"]
    legal_moves = obs.legal_moves()
    observed_hands = [_hand_to_dict(hand) for hand in obs.observed_hands()]
    card_knowledge = [
        [_knowledge_to_dict(k, num_colors=num_colors, num_ranks=num_ranks) for k in player_knows]
        for player_knows in obs.card_knowledge()
    ]
    discard_pile = [_card_to_dict(card) for card in obs.discard_pile()]
    fireworks = _fireworks_to_dict(obs.fireworks())
    last_moves = [_history_item_to_dict(m) for m in obs.last_moves()]
    return HanabiObservation(
        player_id=player_id,
        current_player=state.cur_player(),
        current_player_offset=obs.cur_player_offset(),
        observed_hands=observed_hands,
        card_knowledge=card_knowledge,
        discard_pile=discard_pile,
        fireworks=fireworks,
        deck_size=obs.deck_size(),
        information_tokens=obs.information_tokens(),
        life_tokens=obs.life_tokens(),
        raw_observation=obs,
        legal_moves=legal_moves,
        legal_moves_dict=[move_to_dict(move) for move in legal_moves],
        last_moves=last_moves,
    )

class HanabiLookback1:
    def __init__(self, partial_config, seed):
        # seed the config
        self.config = partial_config.copy()
        self.config["seed"] = seed

        self.cur_game = pyhanabi.HanabiGame(self.config)
        self.cur_state = self.cur_game.new_initial_state()
        _advance_chance_events(self.cur_state)

        # lookback shit
        self.prev_game = None
        self.prev_state = None

        # move buffer
        self.last_move = None

    def _clone_move(self, move):
        """
        Rebuild a move object from a dict or HanabiMove to avoid reusing
        C++ objects tied to a different game/state.
        """
        return _clone_move_for_state(move)

    def apply_move(self, move):
        # if no previous state, init it
        if self.prev_state is None:
            self.prev_game = pyhanabi.HanabiGame(self.config)
            self.prev_state = self.prev_game.new_initial_state()
            _advance_chance_events(self.prev_state)

        # cum
        move_to_apply = self._clone_move(move)
        apply_move_safe(self.cur_state, move_to_apply)
        _advance_chance_events(self.cur_state)

        if self.last_move:
            prev_move = self._clone_move(self.last_move)
            apply_move_safe(self.prev_state, prev_move)
            _advance_chance_events(self.prev_state)

        # print("===== applied move =====")
        # print(move, self.last_move)
        # print("Current player:", self.cur_state.cur_player())
        # print("Previous player:", self.prev_state.cur_player())
        # print("=========================")

        self.last_move = move_to_dict(move_to_apply)
        
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
    return color, rank

'''
DEAL_TYPE_MOVES = [[[None for rank in range(HANABI_GAME_CONFIG["ranks"])] for color in range(HANABI_GAME_CONFIG["colors"])] for player in range(HANABI_GAME_CONFIG["players"])]
needed = HANABI_GAME_CONFIG["ranks"] * HANABI_GAME_CONFIG["colors"] * HANABI_GAME_CONFIG["players"]
game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
while needed > 0:
    fabricated_state = game.new_initial_state()
    for _ in range(HANABI_GAME_CONFIG["players"] * HANABI_GAME_CONFIG["hand_size"]):
        fabricated_state.deal_random_card()
    moves = fabricated_state.move_history()
    idx = 0
    for player in range(HANABI_GAME_CONFIG["players"]):
        for pos in range(HANABI_GAME_CONFIG["hand_size"]):
            move = moves[idx]
            idx += 1
            color, rank = unmask_card(move)
            if not DEAL_TYPE_MOVES[player][color][rank]:
                DEAL_TYPE_MOVES[player][color][rank] = move.move()
                needed -= 1

if DEBUG:
    print(DEAL_TYPE_MOVES, flush=True)

_save_game = game
_save_deal_move_000 = DEAL_TYPE_MOVES[0][0][0]
_save_deal_move_001 = DEAL_TYPE_MOVES[0][0][1]
_save_deal_move_002 = DEAL_TYPE_MOVES[0][0][2]
_save_deal_move_003 = DEAL_TYPE_MOVES[0][0][3]
_save_deal_move_004 = DEAL_TYPE_MOVES[0][0][4]
_save_deal_move_010 = DEAL_TYPE_MOVES[0][1][0]
_save_deal_move_011 = DEAL_TYPE_MOVES[0][1][1]
_save_deal_move_012 = DEAL_TYPE_MOVES[0][1][2]
_save_deal_move_013 = DEAL_TYPE_MOVES[0][1][3]
_save_deal_move_014 = DEAL_TYPE_MOVES[0][1][4]
_save_deal_move_020 = DEAL_TYPE_MOVES[0][2][0]
_save_deal_move_021 = DEAL_TYPE_MOVES[0][2][1]
_save_deal_move_022 = DEAL_TYPE_MOVES[0][2][2]
_save_deal_move_023 = DEAL_TYPE_MOVES[0][2][3]
_save_deal_move_024 = DEAL_TYPE_MOVES[0][2][4]
_save_deal_move_030 = DEAL_TYPE_MOVES[0][3][0]
_save_deal_move_031 = DEAL_TYPE_MOVES[0][3][1]
_save_deal_move_032 = DEAL_TYPE_MOVES[0][3][2]
_save_deal_move_033 = DEAL_TYPE_MOVES[0][3][3]
_save_deal_move_034 = DEAL_TYPE_MOVES[0][3][4]
_save_deal_move_040 = DEAL_TYPE_MOVES[0][4][0]
_save_deal_move_041 = DEAL_TYPE_MOVES[0][4][1]
_save_deal_move_042 = DEAL_TYPE_MOVES[0][4][2]
_save_deal_move_043 = DEAL_TYPE_MOVES[0][4][3]
_save_deal_move_044 = DEAL_TYPE_MOVES[0][4][4]
_save_deal_move_100 = DEAL_TYPE_MOVES[1][0][0]
_save_deal_move_101 = DEAL_TYPE_MOVES[1][0][1]
_save_deal_move_102 = DEAL_TYPE_MOVES[1][0][2]
_save_deal_move_103 = DEAL_TYPE_MOVES[1][0][3]
_save_deal_move_104 = DEAL_TYPE_MOVES[1][0][4]
_save_deal_move_110 = DEAL_TYPE_MOVES[1][1][0]
_save_deal_move_111 = DEAL_TYPE_MOVES[1][1][1]
_save_deal_move_112 = DEAL_TYPE_MOVES[1][1][2]
_save_deal_move_113 = DEAL_TYPE_MOVES[1][1][3]
_save_deal_move_114 = DEAL_TYPE_MOVES[1][1][4]
_save_deal_move_120 = DEAL_TYPE_MOVES[1][2][0]
_save_deal_move_121 = DEAL_TYPE_MOVES[1][2][1]
_save_deal_move_122 = DEAL_TYPE_MOVES[1][2][2]
_save_deal_move_123 = DEAL_TYPE_MOVES[1][2][3]
_save_deal_move_124 = DEAL_TYPE_MOVES[1][2][4]
_save_deal_move_130 = DEAL_TYPE_MOVES[1][3][0]
_save_deal_move_131 = DEAL_TYPE_MOVES[1][3][1]
_save_deal_move_132 = DEAL_TYPE_MOVES[1][3][2]
_save_deal_move_133 = DEAL_TYPE_MOVES[1][3][3]
_save_deal_move_134 = DEAL_TYPE_MOVES[1][3][4]
_save_deal_move_140 = DEAL_TYPE_MOVES[1][4][0]
_save_deal_move_141 = DEAL_TYPE_MOVES[1][4][1]
_save_deal_move_142 = DEAL_TYPE_MOVES[1][4][2]
_save_deal_move_143 = DEAL_TYPE_MOVES[1][4][3]
_save_deal_move_144 = DEAL_TYPE_MOVES[1][4][4]
'''




def fabricate(state: pyhanabi.HanabiState, player_id: int, fabricated_hand: [[pyhanabi.HanabiCard]], verbose: bool = False):
    """
    Return a fabricated game history that plays as if it proceed according to state, but instead had fabricated_hand drawn.
    """
    
    # Step 1: determine dealing order
    deck_idx = 0
    num_players, hand_size = HANABI_GAME_CONFIG["players"], HANABI_GAME_CONFIG["hand_size"]
    initial_deal_length = num_players * hand_size
    deal_to = []
    deck_tracking = [[] for _ in range(num_players)]
    last_acting_player = -1

    deck = []

    for move in state.move_history():
        _debug(f"{deck_tracking} {deal_to} {move}")
        if move.player() == -1:
            if deck_idx < initial_deal_length:
                deal_to.append(deck_idx // hand_size)
            elif last_acting_player != -1:
                deal_to.append(last_acting_player)

            deck_tracking[deal_to[-1]].append(deck_idx)
            deck.append(unmask_card(move))
            deck_idx += 1

        elif move.move().type() in [pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD]:
            last_acting_player = move.player()
            position = move.move().card_index()
            deck_tracking[last_acting_player].pop(position)
    _debug(f"{deck_tracking} {deal_to}")

    # Step 2: fabricate!
    for pos, card in zip(deck_tracking[player_id], fabricated_hand):
        deck[pos] = card

    fabricated_move_history = []
    deck_ptr = 0

    for move in state.move_history():
        if move.player() == -1:
            fabricated_move_history.append((deal_to[deck_ptr], *deck[deck_ptr]))
            deck_ptr += 1
        else:
            fabricated_move_history.append(move_to_dict(move.move()))

    _debug(f"{fabricated_move_history} {deck_ptr} {deck}")
            
    if verbose:
        return fabricated_move_history, deck
    return fabricated_move_history

def advance_state(state, fabricated_move_history):
    """
    Mutate state according to fabricated move history.
    """
    for move in fabricated_move_history:
        apply_move_safe(state, move)


class FabricateRollout:
    def __init__(self, state, player_id, fabricated_hand):
        self.fabricated_move_history, self.deck = fabricate(state, player_id, fabricated_hand, verbose=True)
        self.game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
        self.state = self.game.new_initial_state()

        # Fill in the rest of the deck with a shuffled set of remaining cards.
        full_counts = Counter(
            {
                (color, rank): self.game.num_cards(color, rank)
                for color in range(self.game.num_colors())
                for rank in range(self.game.num_ranks())
            }
        )
        used_counts = Counter(self.deck)

        remaining = []
        for card, total in full_counts.items():
            leftover = total - used_counts[card]
            if leftover < 0:
                raise ValueError(f"Fabricated deals exceed deck availability for card {card}")
            remaining.extend([card] * leftover)

        expected_remaining = state.deck_size()
        if expected_remaining and expected_remaining != len(remaining):
            raise ValueError(f"Remaining deck size mismatch: expected {expected_remaining}, got {len(remaining)}")

        random.shuffle(remaining)
        self.remaining_deck = remaining
        self.deck_ptr = 0
        self.deal_to = None

        _debug(
            f"fabricated init state={self.state} remaining_deck={self.remaining_deck} "
            f"deck_ptr={self.deck_ptr} deal_to={self.deal_to} history={self.fabricated_move_history}"
        )

        advance_state(self.state, self.fabricated_move_history)
        _debug("Advanced fabricated history")
        _debug(
            f"fabricated after advance state={self.state} remaining_deck={self.remaining_deck} "
            f"deck_ptr={self.deck_ptr} deal_to={self.deal_to}"
        )


    def is_terminal(self):
        return self.state.is_terminal()

    def cur_player(self):
        return self.state.cur_player()

    def apply_move(self, move):
        if self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            assert False, "Cannot apply non-deal move at chance node"

        move_obj = move
        if not isinstance(move_obj, pyhanabi.HanabiMove):
            move_obj = _clone_move_for_state(move_obj)

        if move_obj.type() in (pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD):
            self.deal_to = self.state.cur_player()

        _debug(f"cloned move to {move_obj}")
        
        apply_move_safe(self.state, move_obj)

    def advance_chance_events(self):
        while not self.state.is_terminal() and self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID and self.deck_ptr < len(self.remaining_deck):
            next_card = self.remaining_deck[self.deck_ptr]
            self.deck_ptr += 1
            self.state.deal_specific_card(self.deal_to, *next_card)


__all__ = [
    "HanabiObservation",
    "HanabiLookback1",
    "build_observation",
    "_advance_chance_events",
    "_clone_move_for_state",
    "move_to_dict",
    "dict_to_move",
    "apply_move_to_state",
    "apply_move_safe",
    "unmask_card",
    "fabricate",
    "FabricateRollout"
]
