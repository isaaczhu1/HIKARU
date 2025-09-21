# envs.py
import gymnasium as gym
import numpy as np
from hanabi_learning_environment import pyhanabi
from encode import ObservationEncoder, ActionMapper

class HanabiGym2P(gym.Env):
    def __init__(self, seed=0, obs_conf="minimal"):
        self.cfg = dict(colors=5, ranks=5, players=2, hand_size=5,
                        max_information_tokens=8, max_life_tokens=3,
                        observation_type=pyhanabi.ObservationType.CARD_KNOWLEDGE.value, seed=seed)
        self.game = pyhanabi.HanabiGame(self.cfg)
        self.state = self.game.new_initial_state()
        self.obs_encoder = ObservationEncoder(self.game, mode=obs_conf)
        self.amap = ActionMapper(self.game)   # global action id <-> Move
        self.num_moves = self.amap.num_moves
        self.observation_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(-np.inf, np.inf, (self.obs_encoder.obs_dim,), np.float32),
            "legal_mask": gym.spaces.Box(0,1,(self.num_moves,), np.float32),
            "seat": gym.spaces.Discrete(2),
            "prev_other_action": gym.spaces.Discrete(self.num_moves + 1),  # 0..num_moves-1, num_moves = "none"
        })
        self.action_space = gym.spaces.Discrete(self.num_moves)
        self.rng = np.random.RandomState(seed)
        # Track previous actions per seat
        self.prev_action = [-1, -1]  # -1 means none

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self.state = self.game.new_initial_state()
        self.prev_action = [-1, -1]
        return self._pack_obs(), {}

    def step(self, a_id: int):
        # Convert action id -> HLE Move
        move = self.amap.id_to_move(a_id)
        reward_before = self.state.score()
        self.state.apply_move(move)
        reward_after = self.state.score()
        reward = float(reward_after - reward_before)  # +1 only on successful play, else 0
        seat = self.state.cur_player()  # now next seat after move
        # Record prev action for the seat that just acted
        self.prev_action[1 - seat] = a_id  # the "other seat" relative to the *next* actor

        terminated = self.state.is_terminal()
        truncated = False
        if terminated:
            # optional: add a small terminal bonus = final score
            pass
        return self._pack_obs(), reward, terminated, truncated, {}

    def _pack_obs(self):
        seat = self.state.cur_player()
        obs_vec = self.obs_encoder.encode(self.state, seat)   # np.float32
        legal_mask = np.zeros(self.num_moves, np.float32)
        for m in self.state.legal_moves():
            legal_mask[self.amap.move_to_id(m)] = 1.0
        prev_other = self.prev_action[seat]  # last action by the opponent (from this seat's POV)
        prev_other = self.num_moves if prev_other < 0 else prev_other  # map none -> num_moves sentinel
        return {
            "obs": obs_vec,
            "legal_mask": legal_mask,
            "seat": seat,
            "prev_other_action": prev_other
        }
