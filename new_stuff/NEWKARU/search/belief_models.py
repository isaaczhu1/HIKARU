"""Approximate belief sampling for Hanabi SPARTA rollouts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from hanabi_learning_environment import pyhanabi

from envs.full_hanabi_env import HanabiObservation


_GAME_TEMPLATE = pyhanabi.HanabiGame()
Card = Tuple[int, int]

ALL_CARDS: List[Card] = []
for color in range(_GAME_TEMPLATE.num_colors()):
    for rank in range(_GAME_TEMPLATE.num_ranks()):
        count = _GAME_TEMPLATE.num_cards(color, rank)
        for _ in range(count):
            ALL_CARDS.append((color, rank))


@dataclass
class WorldState:
    hands: List[List[pyhanabi.HanabiCard]]
    deck: List[pyhanabi.HanabiCard]


def sample_world_state(obs: HanabiObservation, rng: random.Random) -> WorldState:
    """Samples a concrete assignment consistent with public info and hints."""

    remaining = _compute_remaining_deck(obs)
    hands: List[List[pyhanabi.HanabiCard]] = []
    for pid, knowledge in enumerate(obs.card_knowledge):
        hand: List[pyhanabi.HanabiCard] = []
        for slot, card_info in enumerate(knowledge):
            observed = obs.observed_hands[pid][slot]
            if observed["color"] is not None:
                card = pyhanabi.HanabiCard(
                    pyhanabi.color_char_to_idx(observed["color"]),
                    observed["rank"],
                )
            else:
                card = _sample_card_matching(card_info, remaining, rng)
            hand.append(card)
        hands.append(hand)

    deck: List[pyhanabi.HanabiCard] = []
    for (color, rank), count in remaining.items():
        deck.extend([pyhanabi.HanabiCard(color, rank)] * count)
    rng.shuffle(deck)
    return WorldState(hands=hands, deck=deck)


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


def _sample_card_matching(card_info: Dict[str, int], counts: Dict[Card, int], rng: random.Random) -> pyhanabi.HanabiCard:
    candidates = []
    for (color, rank), count in counts.items():
        if card_info["color"] is not None:
            if pyhanabi.color_char_to_idx(card_info["color"]) != color:
                continue
        if card_info["rank"] is not None and card_info["rank"] != rank:
            continue
        candidates.extend([(color, rank)] * count)
    if not candidates:
        candidates.extend(card for card, count in counts.items() for _ in range(count))
    choice_color, choice_rank = rng.choice(candidates)
    counts[(choice_color, choice_rank)] -= 1
    if counts[(choice_color, choice_rank)] == 0:
        del counts[(choice_color, choice_rank)]
    return pyhanabi.HanabiCard(choice_color, choice_rank)


__all__ = ["WorldState", "sample_world_state"]
