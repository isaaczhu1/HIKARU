# encode.py
import numpy as np
from hanabi_learning_environment import pyhanabi

class ActionMapper:
    """
    Create a fixed global id space for ALL possible Hanabi moves in this game config.
    Maps pyhanabi.Move <-> int id.
    """
    def __init__(self, game: pyhanabi.HanabiGame):
        self._id2move = []
        self._move2id = {}
        # Enumerate *all* moves in canonical order using a dummy state:
        st = game.new_initial_state()
        # HLE exposes all moves through game methods; here we brute-force by scanning a new state,
        # then also scanning through playable hand indices, hints, discards, plays for all seats.
        # Simpler: union of legal moves across randomized states until stable (small in 2p).
        seen = set()
        for _ in range(200):  # enough random steps to see all move templates
            for m in st.legal_moves():
                key = m.to_string()
                if key not in seen:
                    seen.add(key)
                    self._move2id[key] = len(self._id2move)
                    self._id2move.append(pyhanabi.HanabiMove.from_move(m))
            if st.is_terminal(): break
            st.apply_move(np.random.choice(st.legal_moves()))
        self.num_moves = len(self._id2move)

    def move_to_id(self, move: pyhanabi.HanabiMove) -> int:
        return self._move2id[move.to_string()]

    def id_to_move(self, idx: int) -> pyhanabi.HanabiMove:
        return self._id2move[idx]

class ObservationEncoder:
    """
    Convert HLE observation (for a seat) -> flat float32 vector.
    Start minimal for Phase 1; swap to richer later.
    """
    def __init__(self, game: pyhanabi.HanabiGame, mode="minimal"):
        self.game = game
        self.mode = mode
        self.obs_dim = self._calc_dim()

    def _calc_dim(self) -> int:
        # minimal: fireworks (C), info (1), life (1), seat one-hot (2),
        # discard histogram (C*R), partner hand one-hot (hand_size * C*R),
        # self-knowledge mask one-hot (hand_size * (C+R))  [very rough]
        C, R, P, H = self.game.num_colors(), self.game.num_ranks(), self.game.num_players(), self.game.hand_size()
        base = C + 1 + 1 + P
        disc = C * R
        partner = H * (C * R)
        selfknow = H * (C + R)
        return base + disc + partner + selfknow

    def encode(self, state: pyhanabi.HanabiState, seat: int) -> np.ndarray:
        C, R, P, H = self.game.num_colors(), self.game.num_ranks(), self.game.num_players(), self.game.hand_size()
        obs = state.observation(seat)
        vec = []

        # Fireworks
        fw = obs.fireworks()
        vec.extend([fw[c] for c in range(C)])
        # Tokens
        vec.append(obs.information_tokens()); vec.append(obs.life_tokens())
        # Seat one-hot
        seat_oh = np.zeros(P, np.float32); seat_oh[seat] = 1
        vec.extend(seat_oh.tolist())

        # Discard histogram (C×R)
        disc = np.zeros((C, R), np.float32)
        for card in obs.discard_pile():
            disc[card.color(), card.rank()] += 1
        vec.extend(disc.flatten().tolist())

        # Partner visible hand (one-hot per slot over C×R)
        partner_id = 1 - seat
        partner_slots = obs.player_hand(partner_id)
        partner_onehots = np.zeros((H, C * R), np.float32)
        for i, card in enumerate(partner_slots):
            if card.color() >= 0 and card.rank() >= 0:
                partner_onehots[i, card.color() * R + card.rank()] = 1.0
        vec.extend(partner_onehots.flatten().tolist())

        # Self knowledge (very rough): color-hinted mask and rank-hinted mask
        # Each slot: (C + R) binary flags saying which colors/ranks are still possible
        self_mask = np.zeros((H, C + R), np.float32)
        # If you want exact card-knowledge domains, read obs.card_knowledge() and intersect with deck remains.
        for i in range(H):
            info = obs.card_knowledge()[seat][i]
            # info.color_plausible(c), info.rank_plausible(r) exist in HLE
            for c in range(C):
                self_mask[i, c] = 1.0 if info.color_plausible(c) else 0.0
            for r in range(R):
                self_mask[i, C + r] = 1.0 if info.rank_plausible(r) else 0.0
        vec.extend(self_mask.flatten().tolist())

        return np.asarray(vec, dtype=np.float32)
