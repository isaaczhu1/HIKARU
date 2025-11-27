# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HanabiGRUPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_moves: int,
        hidden: int = 256,
        action_emb_dim: int = 32,
        seat_emb_dim: int = 8,
        include_prev_self: bool = False,
    ):
        super().__init__()
        self.num_moves = num_moves
        self.hidden = hidden
        self.include_prev_self = include_prev_self

        # +1 for sentinel "none"
        self.prev_other_emb = nn.Embedding(num_moves + 1, action_emb_dim)
        if include_prev_self:
            self.prev_self_emb = nn.Embedding(num_moves + 1, action_emb_dim)

        self.seat_emb = nn.Embedding(2, seat_emb_dim)

        self.obs_fe = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
        )

        in_gru = hidden + action_emb_dim + seat_emb_dim
        if include_prev_self:
            in_gru += action_emb_dim

        # Batch-first so inputs are [B, T, *]
        self.gru = nn.GRU(input_size=in_gru, hidden_size=hidden, batch_first=True)

        self.pi = nn.Linear(hidden, num_moves)
        self.v  = nn.Linear(hidden, 1)

    def initial_state(self, batch: int, device=None):
        # [num_layers=1, B, H]
        return torch.zeros(1, batch, self.gru.hidden_size, device=device)

    def forward(self, obs_vec, seat, prev_other, h, prev_self=None):
        """
        obs_vec:     [B, T, obs_dim]  (float)
        seat:        [B, T]           (long, 0/1)
        prev_other:  [B, T]           (long in [0..num_moves], where num_moves is sentinel)
        prev_self:   [B, T] or None   (long, optional; only used if include_prev_self=True)
        h:           [1, B, hidden]
        Returns:
          logits: [B, T, num_moves]
          value:  [B, T]      (squeezed on last dim)
          h_new:  [1, B, hidden]
        """
        # ---- Ensure shapes/dtypes ----
        if obs_vec.dim() != 3:
            # Accept [B, obs_dim] and promote to [B,1,obs_dim] defensively
            obs_vec = obs_vec.unsqueeze(1)
        obs_vec = obs_vec.float()

        if seat.dim() != 2:
            seat = seat.view(seat.shape[0], -1)
        if prev_other.dim() != 2:
            prev_other = prev_other.view(prev_other.shape[0], -1)

        seat = seat.long()
        prev_other = prev_other.long()

        if self.include_prev_self:
            if prev_self is None:
                # use sentinel "none" if not provided
                prev_self = torch.full_like(prev_other, fill_value=self.num_moves)
            elif prev_self.dim() != 2:
                prev_self = prev_self.view(prev_self.shape[0], -1)
            prev_self = prev_self.long()

        # ---- Feature/embed paths (all 3-D: [B,T,*]) ----
        x = self.obs_fe(obs_vec)               # [B,T,H]
        e_seat  = self.seat_emb(seat)          # [B,T,E_seat]
        e_other = self.prev_other_emb(prev_other)  # [B,T,E_act]

        parts = [x, e_seat, e_other]
        if self.include_prev_self:
            e_self = self.prev_self_emb(prev_self)  # [B,T,E_act]
            parts.append(e_self)

        # All parts are [B,T,*] → safe to concat on last dim
        z = torch.cat(parts, dim=-1)           # [B,T,H_concat]

        # ---- GRU ----
        y, h_n = self.gru(z, h)                # y: [B,T,H], h_n: [1,B,H]

        # ---- Heads ----
        logits = self.pi(y)                    # [B,T,num_moves]
        value  = self.v(y).squeeze(-1)         # [B,T]

        return logits, value, h_n
