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

def _advance_chance_events(state: pyhanabi.HanabiState, deck_override: list[pyhanabi.HanabiCard] | None = None) -> None:
    """Advance chance nodes (card draws) until a player must act.

    If ``deck_override`` is provided, each dealt card is overwritten with the
    next card from that list, ensuring rollouts follow the sampled deck order.
    """
    while not state.is_terminal() and state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        before = [[(c.color(), c.rank()) for c in hand] for hand in state.player_hands()]
        state.deal_random_card()
    
def unserialize(serialized):
    """
    unserialize a pyhanabi move
    Some preprocessing occurs:
        - *** DEAL MOVES BECOME DEAL SPECIFIC MOVES ***
        - serialized colors will be forced to int
    """
    method_map = {
        pyhanabi.HanabiMoveType.PLAY.name : pyhanabi.HanabiMove.get_play_move,
        pyhanabi.HanabiMoveType.DISCARD.name : pyhanabi.HanabiMove.get_discard_move,
        pyhanabi.HanabiMoveType.REVEAL_COLOR.name : pyhanabi.HanabiMove.get_reveal_color_move,
        pyhanabi.HanabiMoveType.REVEAL_RANK.name : pyhanabi.HanabiMove.get_reveal_rank_move,
        pyhanabi.HanabiMoveType.DEAL.name : pyhanabi.HanabiMove.get_deal_specific_move,
        pyhanabi.HanabiMoveType.RETURN.name : pyhanabi.HanabiMove.get_return_move,
        pyhanabi.HanabiMoveType.DEAL_SPECIFIC.name : pyhanabi.HanabiMove.get_deal_specific_move,
    }
    
    serialized_copy = serialized.copy()
    del serialized_copy["action_type"]
    
    if "color" in serialized_copy and isinstance(serialized_copy["color"], str):
        serialized_copy["color"] = pyhanabi.color_char_to_idx(serialized_copy["color"])
        
    return method_map[serialized["action_type"]](**serialized_copy)


def unmask_card(move):
    """
    move has a representation of the form <(Deal XY)> where X is color and Y is suit.
    Recover the color and suit, and map them back to the numerical values they are assigned in pyhanabi.
    """
    return pyhanabi.HanabiCard(color=move.color(), rank=move.rank())

__all__ = [
    "_advance_chance_events",
    "_clone_move_for_state",
    "unserialize",
    "unmask_card",
    "apply_move_to_state",
    "apply_move_safe",
]
