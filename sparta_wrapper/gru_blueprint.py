from sparta_wrapper.sparta_config import DEVICE
from sparta_wrapper.shared_model import _get_shared_model, _resolve_device
from sparta_wrapper.hanabi_utils import unserialize
import torch
from hanabi_learning_environment import pyhanabi

# codex really wants this object for some reason
class _GameShim:
    """Light shim to mirror rl_env's ObservationEncoder game handle contract."""

    def __init__(self, c_game):
        self._game = c_game

    @property
    def c_game(self):
        return self._game

class GRUBlueprint:
    def __init__(self, ckpt_path, model_cfg, hanabi_cfg):
        self.ckpt_path = ckpt_path
        self.model_cfg = model_cfg
        self.hanabi_cfg = hanabi_cfg

        self.shared_model = _get_shared_model(ckpt_path, model_cfg, _resolve_device(DEVICE))
        self.device = self.shared_model.device

        self._sentinel_none = self.shared_model.num_moves

        self._h = self.shared_model.initial_state()
        self._encoder_cache = {}
        
    def logits(self, obs, state, prev_self_action=None):
        """Select an action for the current player given HanabiObservation."""
        obs_vec = self._encode_vectorized_observation(obs)

        seat = torch.tensor([[obs.get_player()]], device=self.device, dtype=torch.long)

        # Build legal mask and map legal moves to ids
        legal_ids = [self._id_from_move(m) for m in obs.legal_moves()]
        legal_mask = torch.zeros(self.shared_model.num_moves, device=self.device, dtype=torch.float32)
        
        for gid in legal_ids:
            if 0 <= gid < self.shared_model.num_moves:
                legal_mask[gid] = 1.0
        legal_mask = legal_mask.view(1, 1, -1)

        # Previous other action: use last move from opponent if available
        prev_other_id = self._extract_prev_other_id(obs, legal_ids)
        prev_other = torch.tensor([[prev_other_id]], device=self.device, dtype=torch.long)

        # Optional previous self action embedding
        prev_self = None
        if getattr(self.shared_model.net, "include_prev_self", False):
            prev_self_id = self._prev_self_id if prev_self_action is None else int(prev_self_action)
            prev_self = torch.tensor([[prev_self_id]], device=self.device, dtype=torch.long)

        logits, _, h_new = self.shared_model.forward(
            obs_vec=obs_vec,
            seat=seat,
            prev_other=prev_other,
            h=self._h,
            prev_self=prev_self,
        )

        # Detach GRU state to prevent graph accumulation during rollout.
        self._h = h_new.detach()
        logits = logits.squeeze(0)  # [1, num_moves]

        # Mask illegal actions
        very_neg = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(legal_mask.squeeze(0) < 0.5, very_neg)
        return masked
        
    def act(self, obs):
        """
        return the move from logits
        """
        logits = self.logits(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action_id = int(dist.sample().item())

        # Track prev self for next step if applicable
        if getattr(self.shared_model.net, "include_prev_self", False):
            self._prev_self_id = action_id

        # Map chosen id back to actual pyhanabi move
        move = self._move_from_id(action_id, obs.legal_moves())
        return move
    
    def reset_episode(self):
        self._h = self.shared_model.initial_state()
        if getattr(self.shared_model.net, "include_prev_self", False):
            self._prev_self_id = self._sentinel_none

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _encode_vectorized_observation(self, observation):
        """
        Match the baseline HanabiEnv2P encoding by using pyhanabi.ObservationEncoder
        (same as rl_env's 'vectorized' field), then pad/trim to the trained obs_dim.
        """
        # Cache encoder per underlying game to avoid reconstructing every call.
        game_ptr = observation._game
        encoder = self._encoder_cache.get(game_ptr)
        if encoder is None:
            encoder = pyhanabi.ObservationEncoder(
                _GameShim(game_ptr), pyhanabi.ObservationEncoderType.CANONICAL
            )
            self._encoder_cache[game_ptr] = encoder

        vec = encoder.encode(observation)
        obs_vec = torch.tensor(vec, dtype=torch.float32, device=self.device).view(1, 1, -1)
        target = self.shared_model.obs_dim
        cur = obs_vec.shape[-1]
        if cur != target:
            if cur < target:
                pad = torch.zeros(1, 1, target, device=self.device, dtype=torch.float32)
                pad[..., :cur] = obs_vec
                obs_vec = pad
            else:
                obs_vec = obs_vec[..., :target]
        return obs_vec

    def _id_from_move(self, move: pyhanabi.HanabiMove) -> int:
        t = move.type()
        if t == pyhanabi.HanabiMoveType.PLAY:
            return move.card_index()
        if t == pyhanabi.HanabiMoveType.DISCARD:
            return self.shared_model.hand_size + move.card_index()
        if t == pyhanabi.HanabiMoveType.REVEAL_COLOR:
            return 2 * self.shared_model.hand_size + move.color()
        if t == pyhanabi.HanabiMoveType.REVEAL_RANK:
            return 2 * self.shared_model.hand_size + self.shared_model.colors + move.rank()
        return -1

    def _move_from_id(self, gid: int, legal_moves):
        for m in legal_moves:
            if self._id_from_move(m) == gid:
                return m
        # Fallback to first legal move if mapping failed
        return legal_moves[0]

    def _extract_prev_other_id(self, observation, legal_ids):
        # last_moves are ordered most recent first; find an opponent move
        for item in observation.last_moves():
            pid = item.player()
            move_dict = item.move().to_dict()
            if pid is None or move_dict is None:
                continue
            if pid == observation.get_player() or pid == pyhanabi.CHANCE_PLAYER_ID:
                continue
            move = unserialize(move_dict)
            if move is None:
                continue
            gid = self._id_from_move(move)
            if gid >= 0:
                return gid
        return self._sentinel_none