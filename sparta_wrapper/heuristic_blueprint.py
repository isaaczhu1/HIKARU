"""Simple rule-based blueprint for 2-player Hanabi."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from hanabi_learning_environment import pyhanabi

from sparta_wrapper.hanabi_utils import HanabiObservation
import random


@dataclass
class BlueprintConfig:
    target_offset: int = 1  # partner index in 2-player Hanabi


class HeuristicBlueprint:
    """Implements the five-rule policy from HEURISTIC_BLUEPRINT_DESC.md."""

    def __init__(self, config: Optional[BlueprintConfig] = None) -> None:
        self.config = config or BlueprintConfig()
        self.rule_counts: Counter[str] = Counter()
        self._last_rule: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def act(self, observation: HanabiObservation, rng: random.Random) -> pyhanabi.HanabiMove:
        """Select an action consistent with the heuristic rules."""

        # Rule 1: play a card we know is playable.
        play_move = self._find_known_playable_card(observation)
        if play_move is not None:
            self._record_rule_usage("rule1_play_known")
            return play_move

        # Rule 2: hint partner about a playable card.
        if observation.information_tokens > 0:
            hint_move = self._hint_partner_playable_card(observation)
            if hint_move is not None:
                self._record_rule_usage("rule2_hint_partner_playable")
                return hint_move

        # Rule 3: discard a card known to be useless.
        discard_useless = self._discard_known_useless(observation)
        if discard_useless is not None:
            self._record_rule_usage("rule3_discard_useless")
            return discard_useless

        # Rule 4: give a general informative hint if possible.
        if observation.information_tokens > 0:
            general_hint = self._hint_general_information(observation)
            if general_hint is not None:
                self._record_rule_usage("rule4_general_hint")
                return general_hint

        # Rule 5: discard the oldest unhinted (or fallback) card.
        fallback = self._discard_oldest_unhinted(observation)
        if fallback is not None:
            self._record_rule_usage("rule5_discard_fallback")
            return fallback

        # As a last resort, return the first legal move (should never happen).
        self._record_rule_usage("rule_fallback_default")
        return observation.legal_moves[0]

    def reset_rule_counts(self) -> None:
        self.rule_counts.clear()
        self._last_rule = None

    def get_rule_counts(self) -> Dict[str, int]:
        return dict(self.rule_counts)

    def last_rule(self) -> Optional[str]:
        return self._last_rule

    # ------------------------------------------------------------------
    # Rule helpers
    # ------------------------------------------------------------------
    def _find_known_playable_card(self, obs: HanabiObservation) -> Optional[pyhanabi.HanabiMove]:
        hand_knowledge = obs.card_knowledge[0]
        for idx, knowledge in enumerate(hand_knowledge):
            color = knowledge.get("color")
            rank = knowledge.get("rank")
            if color is None or rank is None:
                continue
            needed_rank = obs.fireworks.get(color)
            if needed_rank is not None and rank == needed_rank:
                move = self._play_move_for_card(obs, idx)
                if move is not None:
                    return move
        return None

    def _hint_partner_playable_card(self, obs: HanabiObservation) -> Optional[pyhanabi.HanabiMove]:
        playable_indices = self._partner_playable_indices(obs)
        if not playable_indices:
            return None

        partner_knowledge = obs.card_knowledge[1]
        best_choice: Optional[pyhanabi.HanabiMove] = None
        best_key: Optional[tuple] = None

        for move, move_dict in zip(obs.legal_moves, obs.legal_moves_dict):
            if not self._is_partner_hint(move_dict):
                continue
            covered = self._hint_covered_indices(obs, move_dict)
            if not covered:
                continue

            move_type = move_dict["action_type"]
            for idx in covered:
                if idx not in playable_indices:
                    continue
                knowledge = partner_knowledge[idx]
                needs_color = knowledge.get("color") is None
                needs_rank = knowledge.get("rank") is None

                if not needs_color and not needs_rank:
                    continue

                reveals_color = move_type == "REVEAL_COLOR"
                reveals_rank = move_type == "REVEAL_RANK"
                if reveals_color and not needs_color:
                    continue
                if reveals_rank and not needs_rank:
                    continue

                priority = 0 if reveals_color else 1
                key = (priority, len(covered))
                if best_key is None or key < best_key:
                    best_choice = move
                    best_key = key
        return best_choice

    def _discard_known_useless(self, obs: HanabiObservation) -> Optional[pyhanabi.HanabiMove]:
        hand_knowledge = obs.card_knowledge[0]
        for idx, knowledge in enumerate(hand_knowledge):
            color = knowledge.get("color")
            rank = knowledge.get("rank")
            if color is None or rank is None:
                continue
            needed_rank = obs.fireworks.get(color)
            if needed_rank is not None and rank < needed_rank:
                move = self._discard_move_for_card(obs, idx)
                if move is not None:
                    return move
        return None

    def _hint_general_information(self, obs: HanabiObservation) -> Optional[pyhanabi.HanabiMove]:
        best_move: Optional[pyhanabi.HanabiMove] = None
        best_key: Optional[tuple] = None
        partner_hand = obs.observed_hands[1]
        partner_knowledge = obs.card_knowledge[1]

        for move, move_dict in zip(obs.legal_moves, obs.legal_moves_dict):
            if not self._is_partner_hint(move_dict):
                continue
            covered = self._hint_covered_indices(obs, move_dict)
            if not covered:
                continue
            if move_dict["action_type"] == "REVEAL_COLOR":
                informative = [idx for idx in covered if partner_knowledge[idx].get("color") is None]
            else:  # rank hint
                informative = [idx for idx in covered if partner_knowledge[idx].get("rank") is None]
            if not informative:
                continue

            min_rank = min(partner_hand[idx]["rank"] for idx in informative)
            key = (min_rank, len(covered))
            if best_key is None or key < best_key:
                best_move = move
                best_key = key
        return best_move

    def _discard_oldest_unhinted(self, obs: HanabiObservation) -> Optional[pyhanabi.HanabiMove]:
        hand_knowledge = obs.card_knowledge[0]
        target_index: Optional[int] = None
        for idx, knowledge in enumerate(hand_knowledge):
            if knowledge.get("color") is None and knowledge.get("rank") is None:
                target_index = idx
                break
        if target_index is None:
            target_index = 0
        return self._discard_move_for_card(obs, target_index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _play_move_for_card(self, obs: HanabiObservation, card_index: int) -> Optional[pyhanabi.HanabiMove]:
        for move in obs.legal_moves:
            if move.type() == pyhanabi.HanabiMoveType.PLAY and move.card_index() == card_index:
                return move
        return None

    def _discard_move_for_card(self, obs: HanabiObservation, card_index: int) -> Optional[pyhanabi.HanabiMove]:
        for move in obs.legal_moves:
            if move.type() == pyhanabi.HanabiMoveType.DISCARD and move.card_index() == card_index:
                return move
        return None

    def _partner_playable_indices(self, obs: HanabiObservation) -> List[int]:
        playable = []
        partner_hand = obs.observed_hands[1]
        for idx, card in enumerate(partner_hand):
            needed_rank = obs.fireworks.get(card["color"])
            if needed_rank is not None and card["rank"] == needed_rank:
                playable.append(idx)
        return playable

    def _is_partner_hint(self, move_dict: Dict[str, object]) -> bool:
        if move_dict["action_type"] not in {"REVEAL_COLOR", "REVEAL_RANK"}:
            return False
        return int(move_dict.get("target_offset", -1)) == self.config.target_offset

    def _hint_covered_indices(self, obs: HanabiObservation, move_dict: Dict[str, object]) -> List[int]:
        target_hand = obs.observed_hands[1]
        covered: List[int] = []
        if move_dict["action_type"] == "REVEAL_COLOR":
            for idx, card in enumerate(target_hand):
                if card["color"] == move_dict.get("color"):
                    covered.append(idx)
        elif move_dict["action_type"] == "REVEAL_RANK":
            for idx, card in enumerate(target_hand):
                if card["rank"] == move_dict.get("rank"):
                    covered.append(idx)
        return covered

    def _record_rule_usage(self, rule_name: str) -> None:
        self.rule_counts[rule_name] += 1
        self._last_rule = rule_name


__all__ = ["HeuristicBlueprint", "BlueprintConfig"]
