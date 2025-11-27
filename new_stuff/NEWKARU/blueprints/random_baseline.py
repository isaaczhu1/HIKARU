"""Random hint/discard baseline for debugging."""

from __future__ import annotations

import random
from typing import Optional

from hanabi_learning_environment import pyhanabi

from envs.full_hanabi_env import HanabiObservation


class RandomHintDiscardBlueprint:
    """On each turn: random hint if possible, else discard random card."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def act(self, observation: HanabiObservation) -> pyhanabi.HanabiMove:
        hint_moves = [
            move
            for move in observation.legal_moves
            if move.type() in {pyhanabi.HanabiMoveType.REVEAL_COLOR, pyhanabi.HanabiMoveType.REVEAL_RANK}
        ]
        if observation.information_tokens > 0 and hint_moves:
            return self._rng.choice(hint_moves)

        discard_moves = [move for move in observation.legal_moves if move.type() == pyhanabi.HanabiMoveType.DISCARD]
        if discard_moves:
            return self._rng.choice(discard_moves)

        play_moves = [move for move in observation.legal_moves if move.type() == pyhanabi.HanabiMoveType.PLAY]
        if play_moves:
            return self._rng.choice(play_moves)

        return observation.legal_moves[0]


__all__ = ["RandomHintDiscardBlueprint"]
