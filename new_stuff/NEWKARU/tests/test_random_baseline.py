from __future__ import annotations

from hanabi_learning_environment import pyhanabi

from blueprints.random_baseline import RandomHintDiscardBlueprint
from envs.full_hanabi_env import HanabiObservation, _move_to_action_dict


def make_obs(info_tokens: int = 1) -> HanabiObservation:
    hand_size = 2
    observed_hands = [[{"color": None, "rank": None} for _ in range(hand_size)] for _ in range(2)]
    knowledge = [[{"color": None, "rank": None} for _ in range(hand_size)] for _ in range(2)]
    legal = [
        pyhanabi.HanabiMove.get_discard_move(0),
        pyhanabi.HanabiMove.get_discard_move(1),
        pyhanabi.HanabiMove.get_play_move(0),
        pyhanabi.HanabiMove.get_play_move(1),
        pyhanabi.HanabiMove.get_reveal_color_move(1, pyhanabi.color_char_to_idx("R")),
    ]
    return HanabiObservation(
        player_id=0,
        current_player=0,
        current_player_offset=0,
        observed_hands=observed_hands,
        card_knowledge=knowledge,
        discard_pile=[],
        fireworks={c: 0 for c in pyhanabi.COLOR_CHAR[:5]},
        deck_size=40,
        information_tokens=info_tokens,
        life_tokens=3,
        raw_observation=None,  # type: ignore[arg-type]
        legal_moves=legal,
        legal_moves_dict=[_move_to_action_dict(m) for m in legal],
        last_moves=[],
    )


def test_random_uses_hint_when_tokens() -> None:
    bp = RandomHintDiscardBlueprint(seed=0)
    obs = make_obs(info_tokens=2)
    move = bp.act(obs)
    assert move.type() in {pyhanabi.HanabiMoveType.REVEAL_COLOR, pyhanabi.HanabiMoveType.REVEAL_RANK}


def test_random_discards_when_no_tokens() -> None:
    bp = RandomHintDiscardBlueprint(seed=0)
    obs = make_obs(info_tokens=0)
    move = bp.act(obs)
    assert move.type() == pyhanabi.HanabiMoveType.DISCARD
