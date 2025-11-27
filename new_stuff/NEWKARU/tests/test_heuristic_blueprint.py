from __future__ import annotations

from typing import List

import pytest

from hanabi_learning_environment import pyhanabi

from blueprints.heuristic_blueprint import HeuristicBlueprint
from envs.full_hanabi_env import HanabiObservation, _move_to_action_dict


DEFAULT_FIREWORKS = {c: 0 for c in pyhanabi.COLOR_CHAR[:5]}


def make_observation(
    hand_knowledge: List[dict],
    partner_hand: List[dict],
    partner_knowledge: List[dict],
    fireworks: dict | None = None,
    information_tokens: int = 8,
    legal_moves: list | None = None,
) -> HanabiObservation:
    hand_size = len(hand_knowledge)
    observed_hands = [[{"color": None, "rank": None} for _ in range(hand_size)], partner_hand]
    all_knowledge = [hand_knowledge, partner_knowledge]
    fireworks = fireworks or dict(DEFAULT_FIREWORKS)
    if legal_moves is None:
        legal_moves = []
        for idx in range(hand_size):
            legal_moves.append(pyhanabi.HanabiMove.get_play_move(idx))
            legal_moves.append(pyhanabi.HanabiMove.get_discard_move(idx))
    legal_dicts = [_move_to_action_dict(move) for move in legal_moves]
    return HanabiObservation(
        player_id=0,
        current_player=0,
        current_player_offset=0,
        observed_hands=observed_hands,
        card_knowledge=all_knowledge,
        discard_pile=[],
        fireworks=fireworks,
        deck_size=40,
        information_tokens=information_tokens,
        life_tokens=3,
        raw_observation=None,  # type: ignore[arg-type]
        legal_moves=legal_moves,
        legal_moves_dict=legal_dicts,
        last_moves=[],
    )


def color_hint(color: str) -> pyhanabi.HanabiMove:
    return pyhanabi.HanabiMove.get_reveal_color_move(1, pyhanabi.color_char_to_idx(color))


def rank_hint(rank: int) -> pyhanabi.HanabiMove:
    return pyhanabi.HanabiMove.get_reveal_rank_move(1, rank)


@pytest.fixture
def blueprint() -> HeuristicBlueprint:
    return HeuristicBlueprint()


def test_blueprint_plays_known_card(blueprint: HeuristicBlueprint) -> None:
    hand_knowledge = [
        {"color": "R", "rank": 0},
        {"color": None, "rank": None},
    ]
    partner_hand = [{"color": "G", "rank": 0}, {"color": "B", "rank": 1}]
    partner_knowledge = [{"color": None, "rank": None} for _ in partner_hand]
    legal_moves = [
        pyhanabi.HanabiMove.get_play_move(0),
        pyhanabi.HanabiMove.get_play_move(1),
        pyhanabi.HanabiMove.get_discard_move(0),
        pyhanabi.HanabiMove.get_discard_move(1),
    ]
    obs = make_observation(hand_knowledge, partner_hand, partner_knowledge, legal_moves=legal_moves)
    move = blueprint.act(obs)
    assert move.type() == pyhanabi.HanabiMoveType.PLAY
    assert move.card_index() == 0


def test_blueprint_hints_immediate_partner_play(blueprint: HeuristicBlueprint) -> None:
    hand_knowledge = [{"color": None, "rank": None} for _ in range(2)]
    partner_hand = [
        {"color": "G", "rank": 0},  # playable
        {"color": "R", "rank": 1},
    ]
    partner_knowledge = [{"color": None, "rank": None} for _ in partner_hand]
    legal_moves = [
        pyhanabi.HanabiMove.get_play_move(0),
        pyhanabi.HanabiMove.get_discard_move(0),
        color_hint("G"),
        rank_hint(0),
    ]
    fireworks = dict(DEFAULT_FIREWORKS)
    fireworks["G"] = 0
    obs = make_observation(
        hand_knowledge,
        partner_hand,
        partner_knowledge,
        fireworks=fireworks,
        information_tokens=8,
        legal_moves=legal_moves,
    )
    move = blueprint.act(obs)
    assert move.type() == pyhanabi.HanabiMoveType.REVEAL_COLOR
    assert move.color() == pyhanabi.color_char_to_idx("G")


def test_blueprint_prefers_missing_attribute_hint(blueprint: HeuristicBlueprint) -> None:
    hand_knowledge = [{"color": None, "rank": None} for _ in range(2)]
    partner_hand = [
        {"color": "G", "rank": 0},
        {"color": "R", "rank": 1},
    ]
    partner_knowledge = [
        {"color": "G", "rank": None},  # already knows color
        {"color": None, "rank": None},
    ]
    legal_moves = [
        pyhanabi.HanabiMove.get_play_move(0),
        pyhanabi.HanabiMove.get_discard_move(0),
        color_hint("G"),
        rank_hint(0),
    ]
    obs = make_observation(hand_knowledge, partner_hand, partner_knowledge, legal_moves=legal_moves)
    move = blueprint.act(obs)
    assert move.type() == pyhanabi.HanabiMoveType.REVEAL_RANK
    assert move.rank() == 0


def test_blueprint_discards_known_useless(blueprint: HeuristicBlueprint) -> None:
    hand_knowledge = [
        {"color": "R", "rank": 0},
        {"color": None, "rank": None},
    ]
    partner_hand = [{"color": "G", "rank": 1} for _ in range(2)]
    partner_knowledge = [{"color": None, "rank": None} for _ in partner_hand]
    fireworks = dict(DEFAULT_FIREWORKS)
    fireworks["R"] = 2  # pile already beyond rank 0/1
    legal_moves = [
        pyhanabi.HanabiMove.get_play_move(0),
        pyhanabi.HanabiMove.get_discard_move(0),
        pyhanabi.HanabiMove.get_discard_move(1),
    ]
    obs = make_observation(hand_knowledge, partner_hand, partner_knowledge, fireworks=fireworks, legal_moves=legal_moves, information_tokens=0)
    move = blueprint.act(obs)
    assert move.type() == pyhanabi.HanabiMoveType.DISCARD
    assert move.card_index() == 0


def test_blueprint_general_hint_when_possible(blueprint: HeuristicBlueprint) -> None:
    hand_knowledge = [{"color": None, "rank": None} for _ in range(2)]
    partner_hand = [
        {"color": "R", "rank": 1},
        {"color": "B", "rank": 2},
    ]
    partner_knowledge = [
        {"color": None, "rank": None},
        {"color": "B", "rank": None},
    ]
    legal_moves = [
        pyhanabi.HanabiMove.get_discard_move(0),
        pyhanabi.HanabiMove.get_discard_move(1),
        color_hint("R"),
        rank_hint(2),
    ]
    obs = make_observation(hand_knowledge, partner_hand, partner_knowledge, legal_moves=legal_moves)
    move = blueprint.act(obs)
    assert move.type() in {pyhanabi.HanabiMoveType.REVEAL_COLOR, pyhanabi.HanabiMoveType.REVEAL_RANK}


def test_blueprint_discards_oldest_unhinted_without_tokens(blueprint: HeuristicBlueprint) -> None:
    hand_knowledge = [{"color": None, "rank": None} for _ in range(3)]
    partner_hand = [{"color": "R", "rank": 2} for _ in range(3)]
    partner_knowledge = [{"color": None, "rank": None} for _ in partner_hand]
    legal_moves = [
        pyhanabi.HanabiMove.get_discard_move(0),
        pyhanabi.HanabiMove.get_discard_move(1),
        pyhanabi.HanabiMove.get_discard_move(2),
    ]
    obs = make_observation(hand_knowledge, partner_hand, partner_knowledge, legal_moves=legal_moves, information_tokens=0)
    move = blueprint.act(obs)
    assert move.type() == pyhanabi.HanabiMoveType.DISCARD
    assert move.card_index() == 0
