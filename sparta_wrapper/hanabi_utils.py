"""Shared Hanabi helpers for SPARTA and heuristic blueprints."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from hanabi_learning_environment import pyhanabi
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG


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


def _knowledge_to_dict(
    knowledge: pyhanabi.HanabiCardKnowledge, num_colors: int = 5, num_ranks: int = 5
) -> Dict[str, Any]:
    color_idx = knowledge.color()
    rank_idx = knowledge.rank()
    color = pyhanabi.COLOR_CHAR[color_idx] if isinstance(color_idx, int) else None
    rank = int(rank_idx) if isinstance(rank_idx, int) else None
    mask = [
        [bool(knowledge.color_plausible(c) and knowledge.rank_plausible(r)) for r in range(num_ranks)]
        for c in range(num_colors)
    ]
    return {"color": color, "rank": rank, "mask": mask}


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
        if color_idx < 0 or color_idx >= HANABI_GAME_CONFIG["colors"]:
            raise ValueError(f"Reveal color out of range: {color_idx}")
        return pyhanabi.HanabiMove.get_reveal_color_move(
            int(data["target_offset"]), color_idx
        )
    if move_type == pyhanabi.HanabiMoveType.REVEAL_RANK:
        rank = int(data["rank"])
        if rank < 0 or rank >= HANABI_GAME_CONFIG["ranks"]:
            raise ValueError(f"Reveal rank out of range: {rank}")
        return pyhanabi.HanabiMove.get_reveal_rank_move(
            int(data["target_offset"]), rank
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
    raise NotImplementedError("_overwrite_last_deal is deprecated")
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
    num_colors = HANABI_GAME_CONFIG["colors"]
    num_ranks = HANABI_GAME_CONFIG["ranks"]
    legal_moves = obs.legal_moves()
    observed_hands = [_hand_to_dict(hand) for hand in obs.observed_hands()]
    card_knowledge = [
        [_knowledge_to_dict(k, num_colors=num_colors, num_ranks=num_ranks) for k in player_knows]
        for player_knows in obs.card_knowledge()
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
        
def unmask_card(move):
    """
    move has a representation of the form <(Deal XY)> where X is color and Y is suit.
    Recover the color and suit, and map them back to the numerical values they are assigned in pyhanabi.
    """
    if isinstance(move, pyhanabi.HanabiMove):
        move = str(move)
    s = str(move).strip()
    # Expect formats like "<(Deal W4)>" or "(Deal W4)"
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1]
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]

    parts = s.split()
    if len(parts) != 2 or parts[0].lower() != "deal":
        raise ValueError(f"Unsupported move format for unmask_card: {move}")

    card = parts[1]
    if len(card) != 2:
        raise ValueError(f"Unsupported card token in move: {move}")

    color_char = card[0]
    rank_char = card[1]

    color = pyhanabi.color_char_to_idx(color_char)
    rank = int(rank_char) - 1
    return color, rank

DEAL_TYPE_MOVES = [[[None for rank in range(HANABI_GAME_CONFIG["ranks"])] for color in range(HANABI_GAME_CONFIG["colors"])] for player in range(HANABI_GAME_CONFIG["players"])]
needed = HANABI_GAME_CONFIG["ranks"] * HANABI_GAME_CONFIG["colors"] * HANABI_GAME_CONFIG["players"]
game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
while needed > 0:
    fabricated_state = game.new_initial_state()
    for _ in range(HANABI_GAME_CONFIG["players"] * HANABI_GAME_CONFIG["hand_size"]):
        fabricated_state.deal_random_card()
    moves = fabricated_state.move_history()
    idx = 0
    for player in range(HANABI_GAME_CONFIG["players"]):
        for pos in range(HANABI_GAME_CONFIG["hand_size"]):
            move = moves[idx]
            idx += 1
            color, rank = unmask_card(move)
            if not DEAL_TYPE_MOVES[player][color][rank]:
                DEAL_TYPE_MOVES[player][color][rank] = move.move()
                needed -= 1

print(DEAL_TYPE_MOVES)

'''
_save_game = game
_save_deal_move_000 = DEAL_TYPE_MOVES[0][0][0]
_save_deal_move_001 = DEAL_TYPE_MOVES[0][0][1]
_save_deal_move_002 = DEAL_TYPE_MOVES[0][0][2]
_save_deal_move_003 = DEAL_TYPE_MOVES[0][0][3]
_save_deal_move_004 = DEAL_TYPE_MOVES[0][0][4]
_save_deal_move_010 = DEAL_TYPE_MOVES[0][1][0]
_save_deal_move_011 = DEAL_TYPE_MOVES[0][1][1]
_save_deal_move_012 = DEAL_TYPE_MOVES[0][1][2]
_save_deal_move_013 = DEAL_TYPE_MOVES[0][1][3]
_save_deal_move_014 = DEAL_TYPE_MOVES[0][1][4]
_save_deal_move_020 = DEAL_TYPE_MOVES[0][2][0]
_save_deal_move_021 = DEAL_TYPE_MOVES[0][2][1]
_save_deal_move_022 = DEAL_TYPE_MOVES[0][2][2]
_save_deal_move_023 = DEAL_TYPE_MOVES[0][2][3]
_save_deal_move_024 = DEAL_TYPE_MOVES[0][2][4]
_save_deal_move_030 = DEAL_TYPE_MOVES[0][3][0]
_save_deal_move_031 = DEAL_TYPE_MOVES[0][3][1]
_save_deal_move_032 = DEAL_TYPE_MOVES[0][3][2]
_save_deal_move_033 = DEAL_TYPE_MOVES[0][3][3]
_save_deal_move_034 = DEAL_TYPE_MOVES[0][3][4]
_save_deal_move_040 = DEAL_TYPE_MOVES[0][4][0]
_save_deal_move_041 = DEAL_TYPE_MOVES[0][4][1]
_save_deal_move_042 = DEAL_TYPE_MOVES[0][4][2]
_save_deal_move_043 = DEAL_TYPE_MOVES[0][4][3]
_save_deal_move_044 = DEAL_TYPE_MOVES[0][4][4]
_save_deal_move_100 = DEAL_TYPE_MOVES[1][0][0]
_save_deal_move_101 = DEAL_TYPE_MOVES[1][0][1]
_save_deal_move_102 = DEAL_TYPE_MOVES[1][0][2]
_save_deal_move_103 = DEAL_TYPE_MOVES[1][0][3]
_save_deal_move_104 = DEAL_TYPE_MOVES[1][0][4]
_save_deal_move_110 = DEAL_TYPE_MOVES[1][1][0]
_save_deal_move_111 = DEAL_TYPE_MOVES[1][1][1]
_save_deal_move_112 = DEAL_TYPE_MOVES[1][1][2]
_save_deal_move_113 = DEAL_TYPE_MOVES[1][1][3]
_save_deal_move_114 = DEAL_TYPE_MOVES[1][1][4]
_save_deal_move_120 = DEAL_TYPE_MOVES[1][2][0]
_save_deal_move_121 = DEAL_TYPE_MOVES[1][2][1]
_save_deal_move_122 = DEAL_TYPE_MOVES[1][2][2]
_save_deal_move_123 = DEAL_TYPE_MOVES[1][2][3]
_save_deal_move_124 = DEAL_TYPE_MOVES[1][2][4]
_save_deal_move_130 = DEAL_TYPE_MOVES[1][3][0]
_save_deal_move_131 = DEAL_TYPE_MOVES[1][3][1]
_save_deal_move_132 = DEAL_TYPE_MOVES[1][3][2]
_save_deal_move_133 = DEAL_TYPE_MOVES[1][3][3]
_save_deal_move_134 = DEAL_TYPE_MOVES[1][3][4]
_save_deal_move_140 = DEAL_TYPE_MOVES[1][4][0]
_save_deal_move_141 = DEAL_TYPE_MOVES[1][4][1]
_save_deal_move_142 = DEAL_TYPE_MOVES[1][4][2]
_save_deal_move_143 = DEAL_TYPE_MOVES[1][4][3]
_save_deal_move_144 = DEAL_TYPE_MOVES[1][4][4]
'''




def fabricate(state: pyhanabi.HanabiState, player_id: int, fabricated_hand: [[pyhanabi.HanabiCard]], verbose: bool = False):
    """
    Return a fabricated game state that plays as if it proceed according to state, but instead had fabricated_hand drawn.
    """
    
    # Step 1: determine dealing order

    deck_idx = 0
    num_players, hand_size = HANABI_GAME_CONFIG["players"], HANABI_GAME_CONFIG["hand_size"]
    initial_deal_length = num_players * hand_size

    deal_to = []
    rcv = [[] for _ in range(num_players)]
    last_acting_player = -1

    deck = []

    for move in state.move_history():
        if move.player() == -1:
            if deck_idx < initial_deal_length:
                deal_to.append(deck_idx // hand_size)
            elif last_acting_player != -1:
                deal_to.append(last_acting_player)

            rcv[deal_to[-1]].append(deck_idx)
            deck.append(unmask_card(move))
            deck_idx += 1

        elif move.move().type() in [pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD]:
            last_acting_player = move.player()

    # Step 2: fabricate!
    for pos, card in zip(rcv[player_id][-hand_size:], fabricated_hand):
        deck[pos] = card

    fabricated_move_history = []
    deck_ptr = 0

    for move in state.move_history():
        if move.player() == -1:
            print(deal_to, deck_ptr, deck)
            fabricated_move_history.append((deal_to[deck_ptr], *deck[deck_ptr]))
            deck_ptr += 1
        else:
            fabricated_move_history.append(_move_to_action_dict(move.move()))
            
    if verbose:
        return fabricated_move_history, deck
    return fabricated_move_history

def advance_state(state, fabricated_move_history):
    """
    Mutate state according to fabricated move history.
    """
    for move in fabricated_move_history:
        if isinstance(move, dict):
            move = _action_dict_to_move(move)
        if isinstance(move, pyhanabi.HanabiMove):
            state.apply_move(move)
        else:
            state.deal_specific_card(*move)


class FabricateRollout:
    def __init__(self, state, player_id, fabricated_hand):
        self.fabricated_move_history, self.deck = fabricate(state, player_id, fabricated_hand, verbose=True)
        self.game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
        self.state = self.game.new_initial_state()

        # Fill in the rest of the deck with a shuffled set of remaining cards.
        full_counts = Counter(
            {
                (color, rank): self.game.num_cards(color, rank)
                for color in range(self.game.num_colors())
                for rank in range(self.game.num_ranks())
            }
        )
        used_counts = Counter(self.deck)

        remaining = []
        for card, total in full_counts.items():
            leftover = total - used_counts[card]
            if leftover < 0:
                raise ValueError(f"Fabricated deals exceed deck availability for card {card}")
            remaining.extend([card] * leftover)

        expected_remaining = state.deck_size()
        if expected_remaining and expected_remaining != len(remaining):
            raise ValueError(f"Remaining deck size mismatch: expected {expected_remaining}, got {len(remaining)}")

        random.shuffle(remaining)
        self.remaining_deck = remaining
        self.deck_ptr = 0

        advance_state(self.state, self.fabricated_move_history)

    def is_terminal(self):
        return self.state.is_terminal()

    def cur_player(self):
        return self.state.cur_player()

    def apply_move(self, move):
        if self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            assert False, "Cannot apply non-deal move at chance node"

        if move.type() == pyhanabi.HanabiMoveType.PLAY or move.type() == pyhanabi.HanabiMoveType.DISCARD:
            self.deal_to = self.state.cur_player()
        
        self.state.apply_move(move)

    def advance_chance_events(self):
        while not self.state.is_terminal() and self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID and self.deck_ptr < len(self.remaining_deck):
            next_card = self.remaining_deck[self.deck_ptr]
            self.deck_ptr += 1
            self.state.deal_specific_card(self.deal_to, *next_card)


__all__ = [
    "HanabiObservation",
    "HanabiLookback1",
    "build_observation",
    "_advance_chance_events",
    "_move_to_action_dict",
    "_action_dict_to_move",
    "unmask_card",
    "fabricate"
]
