# encode.py
# -----------------------------------------------------------------------------
# Observation + Action encoding utilities for Hanabi (DeepMind HLE).
# - ObservationEncoder: state -> flat float32 vector for one seat.
# - ActionMapper: FIXED template mapping (no HanabiMove construction; no legal_moves in __init__).
#   It maps ids <-> move "signatures" via either safe attribute reads or to_string() parsing.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import re
import numpy as np
from hanabi_learning_environment import pyhanabi


# ======== Observation Encoder (unchanged from previous version) ========

class ObservationEncoder:
    def __init__(self, game: pyhanabi.HanabiGame, include_touched_flags: bool = True):
        self.game = game
        self.include_touched = include_touched_flags
        self.C = game.num_colors()
        self.R = game.num_ranks()
        self.P = game.num_players()
        self.H = game.hand_size()
        self.CR = self.C * self.R
        self.obs_dim = self._calc_dim()

    def _calc_dim(self) -> int:
        board = self.C + 1 + 1 + 1 + (self.C * self.R)
        seat_turn = self.P + 1
        partner = self.H * (self.C * self.R)
        self_know = self.H * (self.C + self.R)
        touched = self.H * 2 if self.include_touched else 0
        return int(board + seat_turn + partner + self_know + touched)

    def encode(
        self,
        state: "pyhanabi.HanabiState",
        seat: int,
        norm_turn: float = 0.0,
        touched_flags: np.ndarray | None = None,
    ) -> np.ndarray:
        obs = state.observation(seat)
        C, R, H, CR, P = self.C, self.R, self.H, self.CR, self.P

        vec: List[float] = []

        fw = obs.fireworks()
        for c in range(C):
            vec.append(float(fw[c]))
        vec.append(float(obs.information_tokens()) / 8.0)
        vec.append(float(obs.life_tokens()) / 3.0)
        try:
            deck_sz = state.deck_size()
        except Exception:
            deck_sz = 0
        vec.append(float(deck_sz) / 50.0)

        disc = np.zeros((C, R), dtype=np.float32)
        for card in obs.discard_pile():
            c, r = card.color(), card.rank()
            if 0 <= c < C and 0 <= r < R:
                disc[c, r] += 1.0
        vec.extend(disc.flatten().tolist())

        seat_oh = np.zeros(P, dtype=np.float32)
        if 0 <= seat < P:
            seat_oh[seat] = 1.0
        vec.extend(seat_oh.tolist())
        vec.append(float(norm_turn))

        partner_id = (seat + 1) % P
        partner_slots = obs.player_hand(partner_id)
        partner_oh = np.zeros((H, CR), dtype=np.float32)
        for i, card in enumerate(partner_slots[:H]):
            c, r = card.color(), card.rank()
            if 0 <= c < C and 0 <= r < R:
                partner_oh[i, c * R + r] = 1.0
        vec.extend(partner_oh.flatten().tolist())

        self_mask = np.zeros((H, C + R), dtype=np.float32)
        try:
            ck = obs.card_knowledge()[seat]
        except Exception:
            ck = []
        for i in range(min(H, len(ck))):
            info = ck[i]
            for c in range(C):
                self_mask[i, c] = 1.0 if info.color_plausible(c) else 0.0
            for r in range(R):
                self_mask[i, C + r] = 1.0 if info.rank_plausible(r) else 0.0
        vec.extend(self_mask.flatten().tolist())

        if self.include_touched:
            if touched_flags is None:
                vec.extend([0.0] * (H * 2))
            else:
                arr = np.asarray(touched_flags, dtype=np.float32)
                if arr.shape != (H, 2):
                    buf = np.zeros((H, 2), dtype=np.float32)
                    hmin = min(H, arr.shape[0]); cmin = min(2, arr.shape[1] if arr.ndim == 2 else 0)
                    buf[:hmin, :cmin] = arr[:hmin, :cmin]
                    arr = buf
                vec.extend(arr.flatten().tolist())

        return np.asarray(vec, dtype=np.float32)


# ======== ActionMapper (template ids, picks actual move from legal set) ========

class ActionMapper:
    """
    Fixed mapping for 2p Hanabi without ever constructing HanabiMove objects.

      id ranges (H = hand size, C colors, R ranks):
        0..H-1               -> PLAY card_index = i
        H..2H-1              -> DISCARD card_index = i - H
        2H..2H+C-1           -> REVEAL_COLOR color = i - 2H, target_offset = 1
        2H+C..2H+C+R-1       -> REVEAL_RANK  rank  = i - (2H+C), target_offset = 1

    At runtime:
      - move_to_id(move): infer (kind, idx/color/rank) via safe accessors or string parsing
      - id_to_move(state, id): scan current state's legal_moves() and pick the one
        that matches the template (kind + parameter). If none match, fall back to first legal.
    """

    _PLAY = "PLAY"
    _DISCARD = "DISCARD"
    _REVEAL_COLOR = "REVEAL_COLOR"
    _REVEAL_RANK = "REVEAL_RANK"

    def __init__(self, game: "pyhanabi.HanabiGame"):
        self.game = game
        self.C = game.num_colors()
        self.R = game.num_ranks()
        self.P = game.num_players()
        self.H = game.hand_size()
        assert self.P == 2, "ActionMapper is defined for 2 players."

        # Precompute id range cutoffs
        self._play_lo, self._play_hi = 0, self.H
        self._disc_lo, self._disc_hi = self.H, 2 * self.H
        self._rc_lo, self._rc_hi = 2 * self.H, 2 * self.H + self.C
        self._rr_lo, self._rr_hi = self._rc_hi, self._rc_hi + self.R
        self.num_moves = self._rr_hi

        # Precompile regexes for robust string parsing across builds
        self._re_play = re.compile(r"PLAY\s+(\d+)", re.IGNORECASE)
        self._re_discard = re.compile(r"DISCARD\s+(\d+)", re.IGNORECASE)
        self._re_reveal_color = re.compile(r"REVEAL[_\s]?COLOR\s+(\d+)", re.IGNORECASE)
        self._re_reveal_rank = re.compile(r"REVEAL[_\s]?RANK\s+(\d+)", re.IGNORECASE)
        self._re_any_int = re.compile(r"(-?\d+)")

    # ---- public API ----

    def move_to_id(self, move: "pyhanabi.HanabiMove") -> int:
        kind, param = self._signature_from_move(move)
        if kind == self._PLAY:
            return self._play_lo + np.clip(int(param), 0, self.H - 1)
        if kind == self._DISCARD:
            return self._disc_lo + np.clip(int(param), 0, self.H - 1)
        if kind == self._REVEAL_COLOR:
            return self._rc_lo + np.clip(int(param), 0, self.C - 1)
        if kind == self._REVEAL_RANK:
            return self._rr_lo + np.clip(int(param), 0, self.R - 1)
        # Unknown -> map to first legal bucket deterministically
        return self._play_lo

    def id_to_move(self, state: "pyhanabi.HanabiState", idx: int) -> "pyhanabi.HanabiMove":
        """
        Find the legal move matching template encoded by idx.
        If none match (should be rare), return the first legal move.
        """
        desired = self._template_from_id(idx)
        legal = state.legal_moves()
        if not legal:
            raise RuntimeError("id_to_move called on a state with no legal moves.")

        # Try exact match first
        for m in legal:
            if self._signature_from_move(m) == desired:
                return m

        # Be a bit lenient: some builds print different strings; match by kind only.
        kind, _ = desired
        for m in legal:
            k, p = self._signature_from_move(m)
            if k == kind:
                return m

        # Fall back to first legal
        return legal[0]

    # ---- helpers ----

    def _template_from_id(self, idx: int) -> Tuple[str, int]:
        if self._play_lo <= idx < self._play_hi:
            return (self._PLAY, idx - self._play_lo)
        if self._disc_lo <= idx < self._disc_hi:
            return (self._DISCARD, idx - self._disc_lo)
        if self._rc_lo <= idx < self._rc_hi:
            return (self._REVEAL_COLOR, idx - self._rc_lo)
        if self._rr_lo <= idx < self._rr_hi:
            return (self._REVEAL_RANK, idx - self._rr_lo)
        # Clamp out-of-range ids
        return (self._PLAY, 0)

    def _signature_from_move(self, move: "pyhanabi.HanabiMove") -> Tuple[str, int]:
        """
        Extract (kind, param) from move via safe attribute reads; if those fail, use to_string() regexes.
        param is: card_index for PLAY/DISCARD; color for REVEAL_COLOR; rank for REVEAL_RANK.
        """
        # First try attribute/callable accessors (varies by build)
        def _safe(attr) -> Optional[int]:
            v = getattr(move, attr, None)
            try:
                v = v() if callable(v) else v
            except Exception:
                v = None
            if isinstance(v, int):
                return v
            return None

        # Detect move "type" by string; attrs for type enums are not stable across builds
        s = ""
        try:
            s = move.to_string()
        except Exception:
            s = ""

        up = s.upper()

        # PLAY
        if "PLAY" in up:
            ci = _safe("card_index")
            if ci is None:
                # try regex fallback
                m = self._re_play.search(up) or self._re_any_int.search(up)
                ci = int(m.group(1)) if m else 0
            return (self._PLAY, ci)

        # DISCARD
        if "DISCARD" in up:
            ci = _safe("card_index")
            if ci is None:
                m = self._re_discard.search(up) or self._re_any_int.search(up)
                ci = int(m.group(1)) if m else 0
            return (self._DISCARD, ci)

        # REVEAL COLOR
        if "REVEAL" in up and "COLOR" in up:
            color = _safe("color")
            if color is None:
                m = self._re_reveal_color.search(up) or self._re_any_int.search(up)
                color = int(m.group(1)) if m else 0
            return (self._REVEAL_COLOR, color)

        # REVEAL RANK
        if "REVEAL" in up and "RANK" in up:
            rank = _safe("rank")
            if rank is None:
                m = self._re_reveal_rank.search(up) or self._re_any_int.search(up)
                rank = int(m.group(1)) if m else 0
            return (self._REVEAL_RANK, rank)

        # Unknown kind: default bucket
        return (self._PLAY, 0)
