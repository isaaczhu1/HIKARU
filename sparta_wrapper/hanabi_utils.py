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
class HanabiObservation(pyhanabi.HanabiObservation):
    """Structured view of a player's observation and legal moves."""

    player_id: int # the pov of the observation
    current_player: int # player to move
    current_player_offset: int # "roughly current_player - player_id"
    observed_hands: List[List[Dict[str, Any]]] # ??????????
    card_knowledge: List[List[Dict[str, Any]]] # ??????????
    discard_pile: List[Dict[str, Any]] # list of uh. dictionaries. somehow it stores what's discarded. ok.
    fireworks: Dict[str, int] # maps color to number of cards played in that color
    deck_size: int # self explanatory
    information_tokens: int # self explanatory (num hints remaining)
    life_tokens: int # self explanatory
    raw_observation: pyhanabi.HanabiObservation # the thing we wrapped around
    legal_moves: Sequence[pyhanabi.HanabiMove] # yes
    legal_moves_dict: List[Dict[str, Any]] # yes
    last_moves: List[Dict[str, Any]] # yes


    def __init__(self):
        super().__init__()

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
