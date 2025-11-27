"""Thin wrapper around DeepMind's Hanabi Python API for 2-player experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from hanabi_learning_environment import pyhanabi


DEFAULT_GAME_CONFIG: Dict[str, Any] = {
    "players": 2,
    "colors": 5,
    "ranks": 5,
    "hand_size": 5,
    "max_information_tokens": 8,
    "max_life_tokens": 3,
    "seed": -1,
}


def _card_to_dict(card: pyhanabi.HanabiCard) -> Dict[str, Any]:
    return {"color": pyhanabi.COLOR_CHAR[card.color()] if card.color() >= 0 else None,
            "rank": card.rank()}


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


class FullHanabiEnv:
    """Convenience wrapper for the pyhanabi player API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, *, state: Optional[pyhanabi.HanabiState] = None,
                 auto_reset: bool = True) -> None:
        self._base_config = dict(DEFAULT_GAME_CONFIG)
        if config:
            self._base_config.update(config)
        self._game = pyhanabi.HanabiGame(self._base_config)
        self._state: Optional[pyhanabi.HanabiState] = None
        if state is not None:
            self._state = state.copy()
        elif auto_reset:
            self.reset()

    # ------------------------------------------------------------------
    # Core environment control
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> HanabiObservation:
        config = dict(self._base_config)
        if seed is not None:
            config["seed"] = seed
        self._game = pyhanabi.HanabiGame(config)
        self._state = self._game.new_initial_state()
        self._advance_chance_events()
        return self.current_observation()

    def step(self, action: Union[int, Dict[str, Any], pyhanabi.HanabiMove]) -> Tuple[Optional[HanabiObservation], float, bool, Dict[str, Any]]:
        self._require_state()
        move = self._convert_action(action)
        last_score = self._state.score()
        self._state.apply_move(move)
        self._advance_chance_events()
        reward = self._state.score() - last_score
        done = self._state.is_terminal()
        observation = None if done else self.current_observation()
        info = {
            "score": self._state.score(),
            "last_move": _move_to_action_dict(move),
        }
        return observation, float(reward), done, info

    def clone(self) -> "FullHanabiEnv":
        self._require_state()
        return FullHanabiEnv(config=self._base_config, state=self._state, auto_reset=False)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def current_player(self) -> int:
        self._require_state()
        return self._state.cur_player()

    def is_terminal(self) -> bool:
        self._require_state()
        return self._state.is_terminal()

    def score(self) -> int:
        self._require_state()
        return self._state.score()

    def legal_moves(self, as_dict: bool = False) -> Sequence[Union[pyhanabi.HanabiMove, Dict[str, Any]]]:
        self._require_state()
        moves = self._state.legal_moves()
        if as_dict:
            return [_move_to_action_dict(move) for move in moves]
        return moves

    def player_hands(self) -> List[List[Dict[str, Any]]]:
        self._require_state()
        return [_hand_to_dict(hand) for hand in self._state.player_hands()]

    def discard_pile(self) -> List[Dict[str, Any]]:
        self._require_state()
        return [_card_to_dict(card) for card in self._state.discard_pile()]

    def fireworks(self) -> Dict[str, int]:
        self._require_state()
        return _fireworks_to_dict(self._state.fireworks())

    def information_tokens(self) -> int:
        self._require_state()
        return self._state.information_tokens()

    def life_tokens(self) -> int:
        self._require_state()
        return self._state.life_tokens()

    def deck_size(self) -> int:
        self._require_state()
        return self._state.deck_size()

    def num_players(self) -> int:
        self._require_state()
        return self._state.num_players()

    def current_observation(self) -> HanabiObservation:
        self._require_state()
        return self.observation_for_player(self.current_player())

    def observation_for_player(self, player_id: int) -> HanabiObservation:
        self._require_state()
        observation = self._state.observation(player_id)
        observed_hands = [_hand_to_dict(hand) for hand in observation.observed_hands()]
        card_knowledge = [
            [_knowledge_to_dict(knowledge) for knowledge in player_k]
            for player_k in observation.card_knowledge()
        ]
        discard_pile = [_card_to_dict(card) for card in observation.discard_pile()]
        legal_moves = observation.legal_moves()
        last_moves = [_history_item_to_dict(item) for item in observation.last_moves()]
        return HanabiObservation(
            player_id=player_id,
            current_player=self.current_player(),
            current_player_offset=observation.cur_player_offset(),
            observed_hands=observed_hands,
            card_knowledge=card_knowledge,
            discard_pile=discard_pile,
            fireworks=_fireworks_to_dict(observation.fireworks()),
            deck_size=observation.deck_size(),
            information_tokens=observation.information_tokens(),
            life_tokens=observation.life_tokens(),
            raw_observation=observation,
            legal_moves=legal_moves,
            legal_moves_dict=[_move_to_action_dict(move) for move in legal_moves],
            last_moves=last_moves,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _require_state(self) -> None:
        if self._state is None:  # pragma: no cover - defensive
            raise RuntimeError("Environment state is uninitialized. Call reset() first.")

    def _convert_action(self, action: Union[int, Dict[str, Any], pyhanabi.HanabiMove]) -> pyhanabi.HanabiMove:
        if isinstance(action, pyhanabi.HanabiMove):
            return action
        if isinstance(action, dict):
            return _action_dict_to_move(action)
        if isinstance(action, int):
            return self._game.get_move(action)
        raise TypeError(f"Unsupported action type: {action}")

    def _advance_chance_events(self) -> None:
        self._require_state()
        while not self._state.is_terminal() and self._state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self._state.deal_random_card()


__all__ = [
    "FullHanabiEnv",
    "HanabiObservation",
    "DEFAULT_GAME_CONFIG",
    "_move_to_action_dict",
    "_action_dict_to_move",
]
