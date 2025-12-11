import os
import sys

import pytest

from hanabi_learning_environment import pyhanabi

sys.path.append(os.path.dirname(__file__))
from helpers import apply_move_to_state  # noqa: E402


HANABI_GAME_CONFIG = {
    "players": 2,
    "colors": 2,
    "ranks": 3,
    "hand_size": 2,
}


def move_to_dict(move):
  return {
      "type": move.type().name,
      "card_index": move.card_index(),
      "target_offset": move.target_offset(),
      "color": move.color(),
      "rank": move.rank(),
  }


def dict_to_move(game, spec):
  move_type_name = spec.get("type", spec.get("action_type"))
  if move_type_name is None:
    raise ValueError(f"Missing move type in spec: {spec}")
  mtype = pyhanabi.HanabiMoveType[move_type_name]
  if mtype == pyhanabi.HanabiMoveType.PLAY:
    return pyhanabi.HanabiMove.get_play_move(spec["card_index"])
  if mtype == pyhanabi.HanabiMoveType.DISCARD:
    return pyhanabi.HanabiMove.get_discard_move(spec["card_index"])
  if mtype == pyhanabi.HanabiMoveType.REVEAL_COLOR:
    color = spec["color"]
    if color < 0 or color >= HANABI_GAME_CONFIG["colors"]:
      raise ValueError(f"Reveal color out of range: {color}")
    return pyhanabi.HanabiMove.get_reveal_color_move(
        spec["target_offset"], color)
  if mtype == pyhanabi.HanabiMoveType.REVEAL_RANK:
    rank = spec["rank"]
    if rank < 0 or rank >= HANABI_GAME_CONFIG["ranks"]:
      raise ValueError(f"Reveal rank out of range: {rank}")
    return pyhanabi.HanabiMove.get_reveal_rank_move(
        spec["target_offset"], rank)
  if mtype == pyhanabi.HanabiMoveType.DEAL_SPECIFIC:
    return pyhanabi.HanabiMove.get_deal_specific_move(
        spec.get("card_index"), spec["target_offset"], spec["color"], spec["rank"])
  if mtype == pyhanabi.HanabiMoveType.RETURN:
    return pyhanabi.HanabiMove.get_return_move(
        spec["card_index"], spec["target_offset"])
  raise ValueError(f"Unsupported move type {spec['type']}")


def _clone_move_for_state(move):
  if isinstance(move, pyhanabi.HanabiMove):
    spec = move_to_dict(move)
  elif isinstance(move, dict):
    spec = dict(move)
    if "type" not in spec and "action_type" in spec:
      spec["type"] = spec["action_type"]
  else:
    raise ValueError(f"Unsupported move type for cloning: {move!r}")
  return dict_to_move(pyhanabi.HanabiGame(HANABI_GAME_CONFIG), spec)


def apply_move_safe(state, move):
  if isinstance(move, (list, tuple)) and len(move) == 3 and not isinstance(move, pyhanabi.HanabiMove):
    state.deal_specific_card(*move)
    return
  cloned = _clone_move_for_state(move)
  state.apply_move(cloned)


def _initial_game_state():
  game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
  state = game.new_initial_state()
  while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
    state.deal_random_card()
  return game, state


def test_clone_move_validates_and_rebuilds():
  # Legal move from one state should be cloneable for another state.
  game_a, state_a = _initial_game_state()
  move = state_a.legal_moves()[0]
  rebuilt = _clone_move_for_state(move)
  assert isinstance(rebuilt, pyhanabi.HanabiMove)
  assert rebuilt.type() == move.type()

  # Out-of-range hint rank should raise before applying.
  with pytest.raises(ValueError):
    _clone_move_for_state({"action_type": "REVEAL_RANK", "target_offset": 1, "rank": -1})


def test_apply_move_safe_across_states():
  # Applying a move object captured from one state onto another should be safe.
  game_a, state_a = _initial_game_state()
  move = state_a.legal_moves()[0]

  game_b, state_b = _initial_game_state()
  apply_move_safe(state_b, move)  # should not throw or corrupt

  history = state_b.move_history()
  assert history  # move history should include the applied action
  assert history[-1].move().type() == move.type()


def test_apply_move_safe_handles_deal_tuple():
  game, state = _initial_game_state()
  legal_moves = state.legal_moves()
  assert legal_moves, "Expected at least one legal move"

  apply_move_safe(state, legal_moves[0])
  assert state.cur_player() == pyhanabi.CHANCE_PLAYER_ID

  # Deal a specific card to satisfy the chance node without crashing.
  apply_move_safe(state, (0, 0, 0))
  assert state.cur_player() != pyhanabi.CHANCE_PLAYER_ID


def test_move_to_dict_roundtrip():
  game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
  state = game.new_initial_state()
  while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
    state.deal_random_card()
  move = state.legal_moves()[0]

  move_spec = move_to_dict(move)
  rebuilt = dict_to_move(game, move_spec)
  assert isinstance(rebuilt, pyhanabi.HanabiMove)
  assert rebuilt.type() == move.type()


def test_apply_move_to_state_roundtrip():
  game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
  state_a = game.new_initial_state()
  while state_a.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
    state_a.deal_random_card()
  move = state_a.legal_moves()[0]
  move_spec = move_to_dict(move)

  game_b = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
  state_b = game_b.new_initial_state()
  while state_b.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
    state_b.deal_random_card()
  apply_move_to_state(state_b, move_spec)
  history = state_b.move_history()
  assert history
  assert history[-1].move().type() == move.type()


def test_apply_move_to_state_invalid_type():
  game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
  state = game.new_initial_state()
  while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
    state.deal_random_card()
  with pytest.raises(ValueError):
    apply_move_to_state(state, {"type": "NOT_A_MOVE"})
