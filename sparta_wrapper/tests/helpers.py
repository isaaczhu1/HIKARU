"""Helper utilities for applying serialized moves safely."""

from hanabi_learning_environment import pyhanabi


def apply_move_to_state(state, move_spec):
  """Apply a move (dict) to a HanabiState using fresh C++ move objects.

  Args:
    state: HanabiState instance to mutate.
    move_spec: dict with keys:
      - type: str name of HanabiMoveType (e.g., 'PLAY', 'REVEAL_COLOR').
      - card_index: optional int
      - target_offset: optional int (player relative, or absolute for DEAL_SPECIFIC/RETURN)
      - color: optional int
      - rank: optional int
  """
  try:
    mtype = pyhanabi.HanabiMoveType[move_spec["type"]]
  except Exception as exc:  # defensive conversion
    raise ValueError(f"Unknown move type: {move_spec}") from exc
  if mtype == pyhanabi.HanabiMoveType.PLAY:
    move = pyhanabi.HanabiMove.get_play_move(move_spec["card_index"])
  elif mtype == pyhanabi.HanabiMoveType.DISCARD:
    move = pyhanabi.HanabiMove.get_discard_move(move_spec["card_index"])
  elif mtype == pyhanabi.HanabiMoveType.REVEAL_COLOR:
    move = pyhanabi.HanabiMove.get_reveal_color_move(
        move_spec["target_offset"], move_spec["color"])
  elif mtype == pyhanabi.HanabiMoveType.REVEAL_RANK:
    move = pyhanabi.HanabiMove.get_reveal_rank_move(
        move_spec["target_offset"], move_spec["rank"])
  elif mtype == pyhanabi.HanabiMoveType.DEAL_SPECIFIC:
    move = pyhanabi.HanabiMove.get_deal_specific_move(
        move_spec.get("card_index"), move_spec["target_offset"],
        move_spec["color"], move_spec["rank"])
  elif mtype == pyhanabi.HanabiMoveType.RETURN:
    move = pyhanabi.HanabiMove.get_return_move(
        move_spec["card_index"], move_spec["target_offset"])
  else:
    raise ValueError(f"Unsupported move type {move_spec['type']}")

  state.apply_move(move)
