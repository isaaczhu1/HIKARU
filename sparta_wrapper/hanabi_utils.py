"""Shared Hanabi helpers for SPARTA and heuristic blueprints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from hanabi_learning_environment import pyhanabi


def _card_to_dict(card: pyhanabi.HanabiCard) -> Dict[str, Any]:
    return {
        "color": pyhanabi.COLOR_CHAR[card.color()] if card.color() >= 0 else None,
        "rank": card.rank(),
    }


def _hand_to_dict(hand: Sequence[pyhanabi.HanabiCard]) -> List[Dict[str, Any]]:
    return [_card_to_dict(card) for card in hand]


def _fireworks_to_dict(fireworks: Sequence[int]) -> Dict[str, int]:
    colors = pyhanabi.COLOR_CHAR[: len(fireworks)]
    return {color: level for color, level in zip(colors, fireworks)}


def _history_item_to_dict(item: pyhanabi.HanabiHistoryItem) -> Dict[str, Any]:
    payload = {
        "player": item.player(),
        "move": _move_to_action_dict(item.move()),
        "scored": item.scored(),
        "information_token": item.information_token(),
        "color": pyhanabi.COLOR_CHAR[item.color()] if item.color() >= 0 else None,
        "rank": item.rank(),
        "card_info_revealed": item.card_info_revealed(),
        "card_info_newly_revealed": item.card_info_newly_revealed(),
    }
    return payload


def _knowledge_to_dict(knowledge: pyhanabi.HanabiCardKnowledge) -> Dict[str, Any]:
    color_idx = knowledge.color()
    rank_idx = knowledge.rank()
    color = pyhanabi.COLOR_CHAR[color_idx] if isinstance(color_idx, int) else None
    rank = int(rank_idx) if isinstance(rank_idx, int) else None
    return {"color": color, "rank": rank}


def _move_to_action_dict(move: pyhanabi.HanabiMove) -> Dict[str, Any]:
    move_type = move.type()
    payload: Dict[str, Any] = {"action_type": move_type.name}
    if move_type in (pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD):
        payload["card_index"] = move.card_index()
    elif move_type == pyhanabi.HanabiMoveType.REVEAL_COLOR:
        payload["target_offset"] = move.target_offset()
        payload["color"] = pyhanabi.COLOR_CHAR[move.color()]
    elif move_type == pyhanabi.HanabiMoveType.REVEAL_RANK:
        payload["target_offset"] = move.target_offset()
        payload["rank"] = move.rank()
    elif move_type == pyhanabi.HanabiMoveType.DEAL:
        payload["target_offset"] = move.target_offset()
        payload["color"] = pyhanabi.COLOR_CHAR[move.color()]
        payload["rank"] = move.rank()
    return payload


def _action_dict_to_move(data: Dict[str, Any]) -> pyhanabi.HanabiMove:
    try:
        move_type = pyhanabi.HanabiMoveType[data["action_type"].upper()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown action type: {data}") from exc

    if move_type == pyhanabi.HanabiMoveType.PLAY:
        return pyhanabi.HanabiMove.get_play_move(int(data["card_index"]))
    if move_type == pyhanabi.HanabiMoveType.DISCARD:
        return pyhanabi.HanabiMove.get_discard_move(int(data["card_index"]))
    if move_type == pyhanabi.HanabiMoveType.REVEAL_COLOR:
        color = data["color"]
        if isinstance(color, str):
            color_idx = pyhanabi.color_char_to_idx(color)
        else:
            color_idx = int(color)
        return pyhanabi.HanabiMove.get_reveal_color_move(
            int(data["target_offset"]), color_idx
        )
    if move_type == pyhanabi.HanabiMoveType.REVEAL_RANK:
        return pyhanabi.HanabiMove.get_reveal_rank_move(
            int(data["target_offset"]), int(data["rank"])
        )
    raise ValueError(f"Unsupported move payload: {data}")


def _advance_chance_events(state: pyhanabi.HanabiState, deck_override: list[pyhanabi.HanabiCard] | None = None) -> None:
    """Advance chance nodes (card draws) until a player must act.

    If ``deck_override`` is provided, each dealt card is overwritten with the
    next card from that list, ensuring rollouts follow the sampled deck order.
    """
    while not state.is_terminal() and state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        before = [[(c.color(), c.rank()) for c in hand] for hand in state.player_hands()]
        state.deal_random_card()
        if deck_override is not None:
            _overwrite_last_deal(state, before, deck_override)


def _overwrite_last_deal(state: pyhanabi.HanabiState, before, deck_override: list[pyhanabi.HanabiCard]) -> None:
    """Find the newly dealt card slot and overwrite with supplied deck order."""
    if not deck_override:
        raise RuntimeError("Deck override exhausted during rollout; belief deck should cover all draws.")
    after = [[(c.color(), c.rank()) for c in hand] for hand in state.player_hands()]
    target = None
    for pid, (prev, curr) in enumerate(zip(before, after)):
        if len(prev) != len(curr):  # safety: unlikely in standard Hanabi after initial deal
            idx = len(curr) - 1
            target = (pid, idx)
            break
        for idx, (p, q) in enumerate(zip(prev, curr)):
            if p != q:
                target = (pid, idx)
                break
        if target:
            break
    if target is None:
        # Fallback: nothing changed; drop a card from override to keep alignment
        deck_override.pop(0)
        return
    pid, idx = target
    try:
        card = deck_override.pop(0)
    except IndexError:
        return
    state.player_hands()[pid][idx] = card


@dataclass
class HanabiObservation:
    """Structured view of a player's observation and legal moves."""

    player_id: int
    current_player: int
    current_player_offset: int
    observed_hands: List[List[Dict[str, Any]]]
    card_knowledge: List[List[Dict[str, Any]]]
    discard_pile: List[Dict[str, Any]]
    fireworks: Dict[str, int]
    deck_size: int
    information_tokens: int
    life_tokens: int
    raw_observation: pyhanabi.HanabiObservation
    legal_moves: Sequence[pyhanabi.HanabiMove]
    legal_moves_dict: List[Dict[str, Any]]
    last_moves: List[Dict[str, Any]]

    def legal_move_dicts(self) -> List[Dict[str, Any]]:
        return list(self.legal_moves_dict)


def build_observation(state: pyhanabi.HanabiState, player_id: int) -> HanabiObservation:
    """Construct a structured observation for ``player_id`` from a state."""
    obs = state.observation(player_id)
    legal_moves = obs.legal_moves()
    observed_hands = [_hand_to_dict(hand) for hand in obs.observed_hands()]
    card_knowledge = [
        [_knowledge_to_dict(k) for k in player_knows] for player_knows in obs.card_knowledge()
    ]
    discard_pile = [_card_to_dict(card) for card in obs.discard_pile()]
    fireworks = _fireworks_to_dict(obs.fireworks())
    last_moves = [_history_item_to_dict(m) for m in obs.last_moves()]

    return HanabiObservation(
        player_id=player_id,
        current_player=state.cur_player(),
        current_player_offset=obs.cur_player_offset(),
        observed_hands=observed_hands,
        card_knowledge=card_knowledge,
        discard_pile=discard_pile,
        fireworks=fireworks,
        deck_size=obs.deck_size(),
        information_tokens=obs.information_tokens(),
        life_tokens=obs.life_tokens(),
        raw_observation=obs,
        legal_moves=legal_moves,
        legal_moves_dict=[_move_to_action_dict(move) for move in legal_moves],
        last_moves=last_moves,
    )

class HanabiLookback1:
    def __init__(self, partial_config, seed):
        # seed the config
        self.config = partial_config.copy()
        self.config["seed"] = seed

        self.cur_game = pyhanabi.HanabiGame(self.config)
        self.cur_state = self.cur_game.new_initial_state()
        _advance_chance_events(self.cur_state)

        # lookback shit
        self.prev_game = None
        self.prev_state = None

        # move buffer
        self.last_move = None

    def apply_move(self, move):
        # if no previous state, init it
        if self.prev_state is None:
            self.prev_game = pyhanabi.HanabiGame(self.config)
            self.prev_state = self.prev_game.new_initial_state()
            _advance_chance_events(self.prev_state)

        # cum
        self.cur_state.apply_move(move)
        _advance_chance_events(self.cur_state)

        if self.last_move:
            self.prev_state.apply_move(self.last_move)
            _advance_chance_events(self.prev_state)

        # print("===== applied move =====")
        # print(move, self.last_move)
        # print("Current player:", self.cur_state.cur_player())
        # print("Previous player:", self.prev_state.cur_player())
        # print("=========================")

        self.last_move = move
        


__all__ = [
    "HanabiObservation",
    "HanabiLookback1",
    "build_observation",
    "_advance_chance_events",
    "_move_to_action_dict",
    "_action_dict_to_move",
]

