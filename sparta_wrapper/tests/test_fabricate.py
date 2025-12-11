import copy
from typing import Iterable, List, Tuple

from hanabi_learning_environment import pyhanabi

from sparta_wrapper.hanabi_utils import advance_state, fabricate, move_to_dict, unmask_card
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG

Card = Tuple[int, int]


def _hand_tuples(state: pyhanabi.HanabiState) -> List[List[Card]]:
    return [[(c.color(), c.rank()) for c in hand] for hand in state.player_hands()]


def _initial_state(seed: int) -> pyhanabi.HanabiState:
    cfg = dict(HANABI_GAME_CONFIG)
    cfg["seed"] = seed
    game = pyhanabi.HanabiGame(cfg)
    state = game.new_initial_state()
    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        state.deal_random_card()
    return state


def _replay(history: Iterable, cfg: dict) -> pyhanabi.HanabiState:
    game = pyhanabi.HanabiGame(cfg)
    state = game.new_initial_state()
    advance_state(state, history)
    return state


def _deal_trace(state: pyhanabi.HanabiState) -> List[Card]:
    return [unmask_card(item) for item in state.move_history() if item.player() == -1]


def _apply_move_and_advance(state: pyhanabi.HanabiState, move: pyhanabi.HanabiMove) -> None:
    state.apply_move(move)
    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        state.deal_random_card()


def test_fabricate_replays_initial_state_with_new_hand():
    state = _initial_state(seed=111)
    original_hands = _hand_tuples(state)

    fabricated_hand = [(4, 4), (4, 3), (4, 2), (4, 1), (4, 0)]
    fabricated_history = fabricate(state, 1, fabricated_hand)

    replay = _replay(fabricated_history, HANABI_GAME_CONFIG)
    replay_hands = _hand_tuples(replay)

    assert replay_hands[1] == fabricated_hand, "Target hand should be replaced"
    assert replay_hands[0] == original_hands[0], "Other hands should be unchanged"

    # Non-deal moves should be identical after replay.
    original_actions = [move_to_dict(item.move()) for item in state.move_history() if item.player() != -1]
    replay_actions = [move_to_dict(item.move()) for item in replay.move_history() if item.player() != -1]
    assert original_actions == replay_actions


def test_fabricate_replaces_current_hand_after_discards():
    state = _initial_state(seed=222)

    play_move = next(m for m in state.legal_moves() if m.type() == pyhanabi.HanabiMoveType.PLAY)
    _apply_move_and_advance(state, play_move)

    other_hand = _hand_tuples(state)[1]
    fabricated_hand = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    fabricated_history = fabricate(state, 0, fabricated_hand)
    replay = _replay(fabricated_history, HANABI_GAME_CONFIG)

    replay_hands = _hand_tuples(replay)
    assert replay_hands[0] == fabricated_hand, "Fabrication should match the player's actual current hand"
    assert replay_hands[1] == other_hand, "Non-target hands should remain untouched"


def test_fabricate_verbose_returns_consistent_deck():
    state = _initial_state(seed=333)

    fabricated_hand = [(0, 4), (0, 3), (0, 2), (0, 1), (0, 0)]
    fabricated_history, fabricated_deck = fabricate(state, 1, fabricated_hand, verbose=True)

    deal_events = [event for event in fabricated_history if isinstance(event, tuple)]
    assert fabricated_deck == [tuple(card) for (_, *card) in deal_events]
    assert len(fabricated_deck) == len(_deal_trace(state))
