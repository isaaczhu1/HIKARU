# utils.py
import torch

def masked_categorical(logits, legal_mask):
    very_neg = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(legal_mask < 0.5, very_neg)
    dist = torch.distributions.Categorical(logits=masked_logits)
    action = dist.sample()
    logp = dist.log_prob(action)
    return action, logp, dist
