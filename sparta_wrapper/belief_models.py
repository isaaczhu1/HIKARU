"""Approximate belief sampling for Hanabi SPARTA rollouts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from hanabi_learning_environment import pyhanabi

from sparta_wrapper.hanabi_utils import (
    HanabiObservation,
    _action_dict_to_move,
    _move_to_action_dict,
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

# tested
def _iter_all_hands(remaining_deck, knowledge, hand_size: int = HANABI_GAME_CONFIG["hand_size"]):
    """Generate all possible hands from remaining_deck."""
    if hand_size == 0:
        yield []
        return
    for i, card in enumerate(remaining_deck):
        # check if card is plausible
        # print(card, knowledge[-hand_size], knowledge[-hand_size].color_plausible(card[0]), knowledge[-hand_size].rank_plausible(card[1]))
        if not knowledge[-hand_size].color_plausible(card[0]):
            continue
        if not knowledge[-hand_size].rank_plausible(card[1]):
            continue
        # check if card still exists
        if remaining_deck[card] <= 0:
            continue
        # print("isHaram", hand_size, i, card)
        next_deck = remaining_deck.copy()
        # print(next_deck)
        next_deck[card] -= 1
        if next_deck[card] == 0:
            del next_deck[card]
        for sub_hand in _iter_all_hands(next_deck, knowledge, hand_size - 1):
            yield [card] + sub_hand

# tested
def _hand_multiplicity(remaining_deck, hand):
    """Count how many ways the given hand can be drawn from remaining_deck."""
    hand_cts = {}
    for card in hand:
        hand_cts[card] = hand_cts.get(card, 0) + 1
    mult = 1
    for card, ct in hand_cts.items():
        if remaining_deck.get(card, 0) < ct:
            return 0
        # compute n choose k
        n = remaining_deck[card]
        k = ct
        numer = 1
        denom = 1
        for i in range(k):
            numer *= n - i
            denom *= i + 1
        mult *= numer // denom
    return mult

# tested
def _sample_hand(remaining_deck, knowledge, hand_size, rng, takes):
    """Approximate sampling of a plausible hand via per-slot marginal draws."""

    def _sample_single():
        deck_counts = dict(remaining_deck)
        hand = []
        for pos in range(hand_size):
            slot_knowledge = knowledge[-hand_size + pos]
            candidates = [
                (card, ct)
                for card, ct in deck_counts.items()
                if ct > 0 and slot_knowledge.color_plausible(card[0]) and slot_knowledge.rank_plausible(card[1])
            ]
            if not candidates:
                return None
            total = sum(ct for _, ct in candidates)
            pick = rng.uniform(0, total)
            acc = 0.0
            chosen = candidates[-1][0]
            for card, ct in candidates:
                acc += ct
                if pick <= acc:
                    chosen = card
                    break
            hand.append(chosen)
            deck_counts[chosen] -= 1
            if deck_counts[chosen] <= 0:
                deck_counts.pop(chosen, None)
        return hand

    samples = []
    for _ in range(takes):
        hand = _sample_single()
        if hand is None:
            # If knowledge is too restrictive, ignore it and sample uniformly from remaining counts.
            deck_counts = dict(remaining_deck)
            hand = []
            for _ in range(hand_size):
                if not deck_counts:
                    break
                pool = list(deck_counts.items())
                total = sum(ct for _, ct in pool)
                pick = rng.uniform(0, total)
                acc = 0.0
                chosen = pool[-1][0]
                for card, ct in pool:
                    acc += ct
                    if pick <= acc:
                        chosen = card
                        break
                hand.append(chosen)
                deck_counts[chosen] -= 1
                if deck_counts[chosen] <= 0:
                    deck_counts.pop(chosen, None)
        samples.append(hand)
    return samples


from hanabi_learning_environment import pyhanabi
from sparta_wrapper.hanabi_utils import build_observation
from sparta_wrapper.heuristic_blueprint import HeuristicBlueprint  # or your blueprint


def _legal_moves_for_player(obs: HanabiObservation, actor_id: int) -> List[pyhanabi.HanabiMove]:
    """Rebuild legal moves assuming ``actor_id`` is to play given ``obs``."""
    moves: List[pyhanabi.HanabiMove] = []
    hand_size = len(obs.card_knowledge[actor_id])

    for idx in range(hand_size):
        moves.append(pyhanabi.HanabiMove.get_play_move(idx))
        if obs.information_tokens < HANABI_GAME_CONFIG["max_information_tokens"]:
            moves.append(pyhanabi.HanabiMove.get_discard_move(idx))

    if obs.information_tokens > 0:
        num_players = len(obs.observed_hands)
        for offset in range(1, num_players):
            target_id = (actor_id + offset) % num_players
            target_hand = obs.observed_hands[target_id]
            colors = {card["color"] for card in target_hand if card["color"] is not None}
            ranks = {card["rank"] for card in target_hand if card["rank"] is not None}
            for color in colors:
                moves.append(
                    pyhanabi.HanabiMove.get_reveal_color_move(
                        offset, pyhanabi.color_char_to_idx(color)
                    )
                )
            for rank in ranks:
                moves.append(pyhanabi.HanabiMove.get_reveal_rank_move(offset, int(rank)))

    # Deduplicate while preserving order to match pyhanabi move ordering expectations.
    deduped: List[pyhanabi.HanabiMove] = []
    seen = set()
    for move in moves:
        if move not in seen:
            deduped.append(move)
            seen.add(move)
    return deduped


def _predict_partner(blueprint_factory, state, my_id, partner_id, my_hand_guess):
    primed_blueprint = blueprint_factory(state, partner_id, my_id, my_hand_guess)
    observation_fabricate = build_observation(primed_blueprint.initial_fabricated_state, partner_id)
    return primed_blueprint.act(observation_fabricate, legal_moves=_legal_moves_for_player(observation_fabricate, partner_id))


def sample_world_state(
    lagging_state: pyhanabi.HanabiState,
    obs: HanabiObservation,
    rng: random.Random,
    blueprint_factory: Callable[[], object] | None = None,
    takes: int = 8,
    upstream_factor: int = 5,
    max_attempts: int = 100
) -> List[List[pyhanabi.HanabiCard]]:
    """Sample hidden hand + deck consistent with public info/hints and (optionally) last partner move.

    While max_attempts,
    we reject samples where the move is incompatible with the blueprint given the sampled hidden hand.

    If max_attempts = 0, this effectivelly nullifies the blueprint conditioning.

    state: state before the move
    obs: the observation resulting from the move.

    Returns the updated list of compatible hands.
    """
    if lagging_state is None:
        # No conditioning on last move
        knowledge = obs.raw_observation.card_knowledge()[0]
        return _sample_hand(_compute_remaining_deck(obs), knowledge, HANABI_GAME_CONFIG["hand_size"], rng, takes)

    cur_player_offset = obs.current_player_offset

    if cur_player_offset == -1:
        raise ValueError("Bruh you're sampling world state on a chance node? Why?")

    num_players = HANABI_GAME_CONFIG["players"]

    observer_id = (lagging_state.cur_player() + 1 - cur_player_offset + num_players) % num_players
    knowledge = obs.raw_observation.card_knowledge()[0]
    remaining_deck = _compute_remaining_deck(obs)
    
    ptr = 0
    while obs.last_moves[ptr]["player"] == pyhanabi.CHANCE_PLAYER_ID:
        ptr += 1
        if ptr >= len(obs.last_moves):
            raise ValueError("No last move found.")
    last_move = obs.last_moves[ptr]

    samples = []

    it = 0
    while len(samples) < takes:
        print("Try", it)
        if it >= max_attempts:
            # print(f"Failed to generate, pumping out {takes - len(samples)} samples.")
            # fill with anything
            fill_hands = _sample_hand(remaining_deck, knowledge, HANABI_GAME_CONFIG["hand_size"], rng, takes - len(samples))
            samples.extend(fill_hands)
            break
            
        upstream_takes = (takes - len(samples)) * upstream_factor
        candidate_hands = _sample_hand(remaining_deck, knowledge, HANABI_GAME_CONFIG["hand_size"], rng, upstream_takes)

        for hand in candidate_hands:
            predicted_move = _predict_partner(blueprint_factory, lagging_state, observer_id, lagging_state.cur_player(), hand)
            predicted_move_dict = _move_to_action_dict(predicted_move)
            print(last_move["move"])
            print("Predicted move dict:", predicted_move_dict)
            if last_move["move"] == predicted_move_dict:
                print("Accepted hand:", hand)
                samples.append(hand)
            else:
                print("Rejected hand:", hand)
            if len(samples) >= takes:
                break

        it += 1

    return samples[:takes]

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
