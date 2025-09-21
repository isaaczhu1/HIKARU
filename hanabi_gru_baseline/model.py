# model.py
import torch, torch.nn as nn, torch.nn.functional as F

class HanabiGRUPolicy(nn.Module):
    def __init__(self, obs_dim, num_moves, hidden=256,
                 action_emb_dim=32, seat_emb_dim=8, include_prev_self=False):
        super().__init__()
        self.num_moves = num_moves
        self.include_prev_self = include_prev_self

        # +1 for "none" sentinel id
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

        self.gru = nn.GRU(input_size=in_gru, hidden_size=hidden, batch_first=True)
        self.pi = nn.Linear(hidden, num_moves)
        self.v  = nn.Linear(hidden, 1)

    def initial_state(self, batch: int, device=None):
        h = torch.zeros(1, batch, self.gru.hidden_size, device=device)
        return h

    def forward(self, obs_vec, seat, prev_other, h, prev_self=None):
        """
        obs_vec: [B, T, obs_dim]
        seat:    [B, T] int64
        prev_other: [B, T] int64 in [0..num_moves] (num_moves == "none")
        prev_self:  (optional) same shape
        h: [1, B, hidden]
        """
        B, T, _ = obs_vec.shape
        x = self.obs_fe(obs_vec)  # [B,T,H]

        e_seat  = self.seat_emb(seat)                 # [B,T,seat_emb_dim]
        e_other = self.prev_other_emb(prev_other)     # [B,T,action_emb_dim]
        parts = [x, e_seat, e_other]
        if self.include_prev_self:
            parts.append(self.prev_self_emb(prev_self))
        z = torch.cat(parts, dim=-1)                  # [B,T,H+...]
        y, h_n = self.gru(z, h)                       # [B,T,H], [1,B,H]

        logits = self.pi(y)                           # [B,T,num_moves]
        value  = self.v(y).squeeze(-1)                # [B,T]
        return logits, value, h_n
