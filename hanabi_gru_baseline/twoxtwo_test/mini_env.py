import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Box, Discrete

# Global action mapping (keeps your schema)
A_PLAY_0, A_DISCARD_0 = 0, 1
A_REVEAL_C0, A_REVEAL_C1 = 2, 3
A_REVEAL_R1, A_REVEAL_R2 = 4, 5
NUM_MOVES = 6
SENTINEL_NONE = NUM_MOVES

DECK_CARDS = [(0,1), (0,2), (1,1), (1,2)]  # 2 colors × ranks 1..2

class TinyHanabi2x2(gym.Env):
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
            "prev_other_action":Discrete(NUM_MOVES + 1),
        })
        self.action_space = Discrete(NUM_MOVES)

        self._seed = seed
        self.reset(seed=seed)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._seed = seed
        self.np_random, _ = gym.utils.seeding.np_random(self._seed)

        self.deck = list(DECK_CARDS)
        self.np_random.shuffle(self.deck)

        self.hands = [[], []]
        for p in range(2):
            self._draw(p)

        self.discard_counts = np.zeros((self.colors, self.ranks), dtype=np.int32)
        self.stacks = np.zeros(self.colors, dtype=np.int32)  # highest rank per color (0..2)
        self.info_tokens, self.life_tokens = 8, 3
        self.cur_player = 0
        self.prev_other_action = SENTINEL_NONE
        self.last_hint_flags = np.zeros((2, self.hand_size, 2), dtype=np.int32)  # (touched_color, touched_rank)
        self.done = False

        return self._build_obs(), {}

    def step(self, action: int):
        assert not self.done, "Call reset()"
        cp, op = self.cur_player, 1 - self.cur_player
        reward = 0.0

        legal = self._legal_mask()
        if legal[action] > 0.0:
            if action == A_PLAY_0:
                c, r = self.hands[cp][0]
                if self.stacks[c] + 1 == r:
                    self.stacks[c] = r
                    reward = 1.0
                else:
                    self.discard_counts[c, r-1] += 1
                self.hands[cp].pop(0); self._draw(cp)

            elif action == A_DISCARD_0:
                c, r = self.hands[cp][0]
                self.discard_counts[c, r-1] += 1
                self.hands[cp].pop(0); self._draw(cp)

            elif action in (A_REVEAL_C0, A_REVEAL_C1):
                hint_c = 0 if action == A_REVEAL_C0 else 1
                self.last_hint_flags[:] = 0
                if self.hands[op] and self.hands[op][0][0] == hint_c:
                    self.last_hint_flags[op, 0, 0] = 1

            elif action in (A_REVEAL_R1, A_REVEAL_R2):
                hint_r = 1 if action == A_REVEAL_R1 else 2
                self.last_hint_flags[:] = 0
                if self.hands[op] and self.hands[op][0][1] == hint_r:
                    self.last_hint_flags[op, 0, 1] = 1

        self.prev_other_action = action
        self.cur_player = 1 - self.cur_player

        if all(len(h) == 0 for h in self.hands) and len(self.deck) == 0:
            self.done = True
        if np.all(self.stacks == self.ranks):
            self.done = True

        obs = self._build_obs()
        info = {"score": int(self.stacks.sum())}
        return obs, reward, self.done, False, info

    # internals
    def _draw(self, p: int):
        if self.deck and len(self.hands[p]) < self.hand_size:
            self.hands[p].append(self.deck.pop())

    def _legal_mask(self):
        m = np.zeros(NUM_MOVES, dtype=np.float32)
        if self.hands[self.cur_player]:
            m[A_PLAY_0] = 1.0
            m[A_DISCARD_0] = 1.0
        m[A_REVEAL_C0] = m[A_REVEAL_C1] = 1.0
        m[A_REVEAL_R1] = m[A_REVEAL_R2] = 1.0
        return m

    def _obs_dim(self):
        # [info_tokens, life_tokens, deck_size] (3)
        # stacks[2] (as scalars)
        # discards[2*2] (counts)
        # opp color one-hot[2], opp rank one-hot[2]
        # touched flags on opp slot[2]
        return 3 + 2 + 4 + 2 + 2 + 2

    def _encode_obs_vec(self):
        cp, op = self.cur_player, 1 - self.cur_player
        vec = [float(self.info_tokens), float(self.life_tokens), float(len(self.deck))]
        vec.extend([float(self.stacks[0]), float(self.stacks[1])])
        vec.extend(self.discard_counts.flatten().astype(np.float32).tolist())
        if self.hands[op]:
            c, r = self.hands[op][0]
            color_oh = [1.0 if i == c else 0.0 for i in range(2)]
            rank_oh  = [1.0 if i+1 == r else 0.0 for i in range(2)]
        else:
            color_oh = [0.0, 0.0]; rank_oh = [0.0, 0.0]
        vec.extend(color_oh); vec.extend(rank_oh)
        vec.extend(self.last_hint_flags[op, 0].astype(np.float32).tolist())
        return np.array(vec, dtype=np.float32)

    def _build_obs(self):
        return {
            "obs": self._encode_obs_vec(),
            "legal_mask": self._legal_mask(),
            "seat": np.int64(self.cur_player),
            "prev_other_action": np.int64(self.prev_other_action),
        }
