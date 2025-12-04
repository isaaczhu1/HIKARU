import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Box, Discrete

# Global action mapping (keeps your schema)
A_PLAY_0, A_DISCARD_0 = 0, 1
A_REVEAL_C0, A_REVEAL_C1 = 2, 3
A_REVEAL_R1, A_REVEAL_R2 = 4, 5
NUM_MOVES = 6
SENTINEL_NONE = NUM_MOVES

# 2 colors × ranks 1..2
DECK_CARDS = [(0, 1), (0, 2), (1, 1), (1, 2)]


class TinyHanabi2x2(gym.Env):
    """
    Extremely small Hanabi-like toy:
      - 2 players, 1 card per hand
      - colors = {0,1}, ranks = {1,2}
      - 6 actions: PLAY, DISCARD, 2 color hints, 2 rank hints
    Rewards:
      - +1 for a successful play (advances the stack)
      - 0 otherwise (no explicit penalty for misplays/discards in this toy)

    Observation dict matches your main wrapper:
      {
        "obs":              float32[obs_dim],
        "legal_mask":       float32[NUM_MOVES],
        "seat":             int (0 or 1),
        "prev_other_action":int in [0..NUM_MOVES] (NUM_MOVES = sentinel),
      }
    """

    metadata = {}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.n_players = 2
        self.hand_size = 1
        self.colors = 2
        self.ranks = 2

        obs_dim = self._obs_dim()
        self.observation_space = Dict({
            "obs":              Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
            "legal_mask":       Box(0.0, 1.0, shape=(NUM_MOVES,), dtype=np.float32),
            "seat":             Discrete(2),
            "prev_other_action": Discrete(NUM_MOVES + 1),
        })
        self.action_space = Discrete(NUM_MOVES)

        self._seed = seed
        self.reset(seed=seed)

    # --------------------------------------------------------------------- #
    # Gym API
    # --------------------------------------------------------------------- #

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._seed = seed
        self.np_random, _ = gym.utils.seeding.np_random(self._seed)

        # Fresh deck and hands
        self.deck = list(DECK_CARDS)
        self.np_random.shuffle(self.deck)

        self.hands = [[], []]  # each is a list of (color, rank)
        for p in range(self.n_players):
            self._draw(p)

        # Discards and stacks
        self.discard_counts = np.zeros((self.colors, self.ranks), dtype=np.int32)
        self.stacks = np.zeros(self.colors, dtype=np.int32)  # highest rank per color (0..2)

        # Tokens (simple Hanabi-like behavior)
        self.info_tokens = 8
        self.life_tokens = 3

        # Turn + bookkeeping
        self.cur_player = 0
        self.prev_other_action = SENTINEL_NONE
        # last_hint_flags[p, slot, :] = [touched_color, touched_rank]
        self.last_hint_flags = np.zeros((self.n_players, self.hand_size, 2), dtype=np.int32)
        self.done = False

        return self._build_obs(), {}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called on done TinyHanabi2x2 env; call reset().")

        cp = self.cur_player
        op = 1 - cp
        reward = 0.0

        # Enforce legality (no action should be illegal under random test,
        # but we guard anyway).
        legal = self._legal_mask()
        if legal[action] > 0.0:
            # --- PLAY ---
            if action == A_PLAY_0 and self.hands[cp]:
                c, r = self.hands[cp][0]
                # Correct play if next rank on that stack
                if self.stacks[c] + 1 == r:
                    self.stacks[c] = r
                    reward = 1.0
                else:
                    # Misplay: count as discard + life lost
                    self.discard_counts[c, r - 1] += 1
                    self.life_tokens = max(0, self.life_tokens - 1)
                # Card replaced → reset hint flags for this slot
                self.hands[cp].pop(0)
                self._draw(cp)  # will clear hint flags for that new card

            # --- DISCARD ---
            elif action == A_DISCARD_0 and self.hands[cp]:
                c, r = self.hands[cp][0]
                self.discard_counts[c, r - 1] += 1
                # Simple "info recovery": discarding regains a token (up to 8)
                self.info_tokens = min(8, self.info_tokens + 1)
                self.hands[cp].pop(0)
                self._draw(cp)  # will clear hint flags for new card

            # --- REVEAL COLOR ---
            elif action in (A_REVEAL_C0, A_REVEAL_C1) and self.info_tokens > 0:
                hint_c = 0 if action == A_REVEAL_C0 else 1
                self.info_tokens -= 1
                # New hint: clear all flags globally, then set for the opponent's slot if match
                self.last_hint_flags[:] = 0
                if self.hands[op] and self.hands[op][0][0] == hint_c:
                    self.last_hint_flags[op, 0, 0] = 1

            # --- REVEAL RANK ---
            elif action in (A_REVEAL_R1, A_REVEAL_R2) and self.info_tokens > 0:
                hint_r = 1 if action == A_REVEAL_R1 else 2
                self.info_tokens -= 1
                self.last_hint_flags[:] = 0
                if self.hands[op] and self.hands[op][0][1] == hint_r:
                    self.last_hint_flags[op, 0, 1] = 1
            # else: illegal under current tokens/hand, treated as no-op
        # else: illegal under mask, treated as no-op

        # Bookkeeping for prev_other_action: at the *next* state, the other
        # player will see the action we just took.
        self.prev_other_action = action
        # Alternate seat
        self.cur_player = 1 - self.cur_player

        # Termination conditions:
        #  - all stacks complete (2 in each color)
        #  - no cards left in deck or hands
        #  - life tokens exhausted
        if np.all(self.stacks == self.ranks):
            self.done = True
        if all(len(h) == 0 for h in self.hands) and len(self.deck) == 0:
            self.done = True
        if self.life_tokens <= 0:
            self.done = True

        obs = self._build_obs()
        info = {"score": int(self.stacks.sum())}
        return obs, reward, self.done, False, info

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _draw(self, p: int):
        """Draw into slot 0 if space and deck non-empty. Reset hint flags for that slot."""
        if self.deck and len(self.hands[p]) < self.hand_size:
            self.hands[p].append(self.deck.pop())
        # New card in slot 0: any previous hint info on that slot is invalid.
        self.last_hint_flags[p, 0] = 0

    def _legal_mask(self):
        """
        Legal moves:
          - PLAY/DISCARD if you have a card.
          - Hints always legal if info_tokens > 0 (in this toy).
        """
        m = np.zeros(NUM_MOVES, dtype=np.float32)
        if self.hands[self.cur_player]:
            m[A_PLAY_0] = 1.0
            m[A_DISCARD_0] = 1.0
        if self.info_tokens > 0:
            m[A_REVEAL_C0] = 1.0
            m[A_REVEAL_C1] = 1.0
            m[A_REVEAL_R1] = 1.0
            m[A_REVEAL_R2] = 1.0
        return m

    def _obs_dim(self):
        """
        Observation layout:
          - info_tokens, life_tokens, deck_size         (3)
          - stacks[2]                                   (2)
          - discards[2 * 2] counts                     (4)
          - opponent color one-hot[2]                  (2)
          - opponent rank one-hot[2]                   (2)
          - touched flags on opponent slot[2]          (2)
        Total = 3 + 2 + 4 + 2 + 2 + 2 = 15
        """
        return 3 + 2 + 4 + 2 + 2 + 2

    def _encode_obs_vec(self):
        cp = self.cur_player
        op = 1 - cp

        # Info tokens, life tokens, deck size
        vec = [
            float(self.info_tokens),
            float(self.life_tokens),
            float(len(self.deck)),
        ]

        # Stacks (highest rank per color)
        vec.extend([float(self.stacks[0]), float(self.stacks[1])])

        # Discard counts flattened [C,R]
        vec.extend(self.discard_counts.flatten().astype(np.float32).tolist())

        # Opponent's card (color, rank) as one-hots
        if self.hands[op]:
            c, r = self.hands[op][0]
            color_oh = [1.0 if i == c else 0.0 for i in range(self.colors)]
            rank_oh = [1.0 if (i + 1) == r else 0.0 for i in range(self.ranks)]
        else:
            color_oh = [0.0] * self.colors
            rank_oh = [0.0] * self.ranks
        vec.extend(color_oh)
        vec.extend(rank_oh)

        # Last hint flags for opponent's slot (touched_color, touched_rank)
        vec.extend(self.last_hint_flags[op, 0].astype(np.float32).tolist())

        return np.asarray(vec, dtype=np.float32)

    def _build_obs(self):
        return {
            "obs": self._encode_obs_vec(),
            "legal_mask": self._legal_mask(),
            "seat": np.int64(self.cur_player),
            "prev_other_action": np.int64(self.prev_other_action),
        }
