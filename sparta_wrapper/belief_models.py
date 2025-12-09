"""Approximate belief sampling for Hanabi SPARTA rollouts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from hanabi_learning_environment import pyhanabi

from sparta_wrapper.hanabi_utils import (
    HanabiObservation,
    _action_dict_to_move,
    build_observation,
)

from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG
_GAME_TEMPLATE = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
Card = Tuple[int, int]

ALL_CARDS: List[Card] = []
for color in range(_GAME_TEMPLATE.num_colors()):
    for rank in range(_GAME_TEMPLATE.num_ranks()):
        count = _GAME_TEMPLATE.num_cards(color, rank)
        for _ in range(count):
            ALL_CARDS.append((color, rank))

def _iter_all_hands(remaining_deck, knowledge, hand_size: int = HANABI_GAME_CONFIG["hand_size"]):
    """Generate all possible hands from remaining_deck."""
    if hand_size == 0:
        yield []
        return
    for i, card in enumerate(remaining_deck):
        # check if card is plausible
        if not knowledge[-hand_size].color_plausible(card[0]):
            continue
        if not knowledge[-hand_size].rank_plausible(card[1]):
            continue
        # check if card still exists
        if remaining_deck[card] <= 0:
            continue
        next_deck = remaining_deck.copy()
        next_deck[card] -= 1
        if next_deck[card] == 0:
            del next_deck[card]
        for sub_hand in _iter_all_hands(next_deck, knowledge, hand_size - 1):
            yield [card] + sub_hand

from hanabi_learning_environment import pyhanabi
from sparta_wrapper.hanabi_utils import build_observation
from sparta_wrapper.heuristic_blueprint import HeuristicBlueprint  # or your blueprint

def _predict_partner(blueprint_factory, state, my_id, partner_id, my_hand_guess):
    # my_hand_guess: list of (color_idx_or_char, rank)
    obs = build_observation(state, partner_id)

    # Override what the partner sees in your hand with the hypothesis.
    obs.observed_hands[my_id] = [
        {
            "color": c if isinstance(c, str) else pyhanabi.COLOR_CHAR[c],
            "rank": int(r),
        }
        for c, r in my_hand_guess
    ]

    # (Optional) If your hypothesis changes hint legality, rebuild legal moves accordingly;
    # for most play/discard choices the existing legal_moves are fine.

    return blueprint_factory().act(obs)


def sample_world_state(
    last_belief: List[List[pyhanabi.HanabiCard]],
    base_state: pyhanabi.HanabiState,
    obs: HanabiObservation,
    rng: random.Random,
    blueprint_factory: Callable[[], object] | None = None,
) -> List[List[pyhanabi.HanabiCard]]:
    """Sample hidden hand + deck consistent with public info/hints and (optionally) last partner move.

    If ``blueprint_factory`` is provided (which is the last actor's blueprint),
    we reject samples where the move is incompatible with the blueprint given the sampled hidden hand.

    last_belief: last possible list of hands that could be held for this player.
    obs: the observation resulting from the move.

    Returns the updated list of compatible hands.
    """
    cur_player_offset = obs.current_player_offset()

    if cur_player_offset == -1:
        raise ValueError("Bruh you're sampling world state on a chance node? Why?")

    observer_id = (state.cur_player() - cur_player_offset + obs.num_players()) % obs.num_players()
    knowledge = obs.raw_observation.card_knowledge()[observer_id]
    remaining_deck = _compute_remaining_deck(obs)
    
    last_move = obs.last_moves()[0]
    valid_hand_prefixes = last_belief

    if cur_player_offset == 1:
        # You made the last move.
        if last_move.type() == pyhanabi.HanabiMoveType.REVEAL_COLOR or last_move.type() == pyhanabi.HanabiMoveType.REVEAL_RANK:
            # Nothing happens.
            return last_belief
        else:
            # acted position, last card
            rem_position = last_move.card_index()
            last_card = (last_move.color(), last_move.rank())

            # get new possible hand prefixes
            valid_hand_prefixes = [belief[:rem_position] + belief[rem_position + 1:] 
                                    for belief in last_belief if belief[rem_position] == last_card]

    for hand in _iter_all_hands(remaining_deck, knowledge):
        # check that the last made move is consistent with this world state.
        blueprint = blueprint_factory() if blueprint_factory is not None else None

        

        




    raise NotImplementedError("todo")
    


# seems ok
def _compute_remaining_deck(obs: HanabiObservation) -> Dict[Card, int]:
    deck_counts: Dict[Card, int] = {}
    for card in ALL_CARDS:
        deck_counts[card] = deck_counts.get(card, 0) + 1

    for hand in obs.observed_hands:
        for card_dict in hand:
            if card_dict["color"] is not None:
                key = (
                    pyhanabi.color_char_to_idx(card_dict["color"]),
                    card_dict["rank"],
                )
                deck_counts[key] -= 1

    for card_dict in obs.discard_pile:
        if card_dict["color"] is None:
            continue
        key = (
            pyhanabi.color_char_to_idx(card_dict["color"]),
            card_dict["rank"],
        )
        deck_counts[key] -= 1

    for color_char, highest in obs.fireworks.items():
        if highest <= 0:
            continue
        color = pyhanabi.color_char_to_idx(color_char)
        for rank in range(highest):
            deck_counts[(color, rank)] -= 1
    return {card: count for card, count in deck_counts.items() if count > 0}


__all__ = ["WorldSample", "sample_world_state"]


def _is_compatible_with_last_move(
    base_state: pyhanabi.HanabiState,
    sample: WorldSample,
    last_move_dict: Dict[str, object],
    blueprint_factory: Callable[[], object],
) -> bool:
    """Check whether the partner's last move matches the blueprint under the sampled world."""
    # Clone state and apply sampled hands
    state_copy = base_state.copy()
    apply_sampled_hands(state_copy, sample)

    try:
        partner = int(last_move_dict.get("player", -1))
    except Exception:
        return False
    if partner < 0 or partner >= state_copy.num_players():
        return False

    try:
        expected_move = _action_dict_to_move(last_move_dict)
    except Exception:
        return False
    partner_obs = build_observation(state_copy, partner)
    if not partner_obs.legal_moves:
        partner_obs.legal_moves = (expected_move,)
        partner_obs.legal_moves_dict = [last_move_dict]
    blueprint = blueprint_factory()
    try:
        predicted = blueprint.act(partner_obs)
    except Exception:
        return False
    return predicted == expected_move
