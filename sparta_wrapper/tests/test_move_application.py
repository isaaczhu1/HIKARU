import pytest

from hanabi_learning_environment import pyhanabi

from sparta_wrapper.hanabi_utils import (
    HANABI_GAME_CONFIG,  # type: ignore[attr-defined]
    _clone_move_for_state,
    apply_move_safe,
    apply_move_to_state,
    move_to_dict,
    dict_to_move,
)


def _initial_state():
    game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
    state = game.new_initial_state()
    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        state.deal_random_card()
    return game, state


def test_clone_move_validates_and_rebuilds():
    # Legal move from one state should be cloneable for another state.
    _, state_a = _initial_state()
    move = state_a.observation(state_a.cur_player()).legal_moves()[0]
    rebuilt = _clone_move_for_state(move)
    assert isinstance(rebuilt, pyhanabi.HanabiMove)
    assert rebuilt.type() == move.type()

    # Out-of-range hint rank should raise before applying.
    with pytest.raises(ValueError):
        _clone_move_for_state({"action_type": "REVEAL_RANK", "target_offset": 1, "rank": -1})


def test_apply_move_safe_across_states():
    # Applying a move object captured from one state onto another should be safe.
    _, state_a = _initial_state()
    move = state_a.observation(state_a.cur_player()).legal_moves()[0]

    _, state_b = _initial_state()
    apply_move_safe(state_b, move)  # should not throw or corrupt

    history = state_b.move_history()
    assert history  # move history should include the applied action
    assert history[-1].move().type() == move.type()


def test_apply_move_safe_handles_deal_tuple():
    _, state = _initial_state()
    obs = state.observation(state.cur_player())
    plays_or_discards = [
        m for m in obs.legal_moves()
        if m.type() in (pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD)
    ]
    assert plays_or_discards, "Expected at least one play/discard move"

    apply_move_safe(state, plays_or_discards[0])
    assert state.cur_player() == pyhanabi.CHANCE_PLAYER_ID  # draw step

    # Deal a specific card to satisfy the chance node without crashing.
    apply_move_safe(state, (0, 0, 0))
    assert state.cur_player() != pyhanabi.CHANCE_PLAYER_ID


def test_move_to_dict_roundtrip():
    game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
    state = game.new_initial_state()
    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        state.deal_random_card()
    move = state.observation(state.cur_player()).legal_moves()[0]

    move_spec = move_to_dict(move)
    rebuilt = dict_to_move(game, move_spec)
    assert isinstance(rebuilt, pyhanabi.HanabiMove)
    assert rebuilt.type() == move.type()


def test_apply_move_to_state_roundtrip():
    game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
    state_a = game.new_initial_state()
    while state_a.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        state_a.deal_random_card()
    move = state_a.observation(state_a.cur_player()).legal_moves()[0]
    move_spec = move_to_dict(move)

    state_b = game.new_initial_state()
    while state_b.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        state_b.deal_random_card()
    apply_move_to_state(state_b, move_spec)
    history = state_b.move_history()
    assert history
    assert history[-1].move().type() == move.type()


def test_apply_move_to_state_invalid_type():
    game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
    state = game.new_initial_state()
    with pytest.raises(ValueError):
        apply_move_to_state(state, {"type": "NOT_A_MOVE"})
