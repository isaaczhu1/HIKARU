"""
Data loading and preprocessing for SPARTA rollouts.

Expected input: one or more JSONL (newline-delimited JSON) files, or a single
JSON array, where each record corresponds to one SPARTA decision timestep and
contains at least:
  - game_id (int)
  - t (int) timestep within the game
  - seat (int)
  - obs_vec (list/array of floats)
  - action_sparta (int) the chosen action id
  - legal_mask (list/array of length num_moves) OR legal_action_ids (list[int])
  - reward (float) shaped reward for this step
  - done (bool) whether the episode ended after this step

Optional fields we will use when present:
  - prev_other_action (int) sentinel/last opponent action id
  - action_blueprint (int) blueprint argmax action
  - Q_values (list[float]) SPARTA estimates (not required for v1)

Outputs:
  - A flat list of per-step samples ready for supervision, each including
    obs_vec, seat, prev_other_action, legal_mask, action_target, value_target,
    and an optional is_override flag when blueprint action is provided.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


def _read_json_records(path: Path) -> List[Dict]:
    """Read a JSONL file (or a JSON array) into a list of dicts."""
    text = path.read_text().strip()
    if not text:
        return []
    # If it looks like a JSON array, load directly.
    if text[0] in "[{":
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return [payload]
        except json.JSONDecodeError:
            pass
    # Fallback: treat as JSONL.
    records: List[Dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def load_sparta_logs(paths: Sequence[str]) -> List[Dict]:
    """Load and concatenate SPARTA step records from one or more paths."""
    all_records: List[Dict] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(f"SPARTA log not found: {p}")
        all_records.extend(_read_json_records(path))
    return all_records


def _group_by_game(records: Iterable[Dict]) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = {}
    for rec in records:
        gid = int(rec.get("game_id", -1))
        if gid < 0:
            raise ValueError(f"Missing or invalid game_id in record: {rec}")
        grouped.setdefault(gid, []).append(rec)
    # Sort each game's steps by t to ensure correct temporal order.
    for steps in grouped.values():
        steps.sort(key=lambda r: int(r.get("t", 0)))
    return grouped


def _ensure_mask(num_moves: int, rec: Dict) -> np.ndarray:
    """Build a legal mask array from either legal_mask or legal_action_ids."""
    if "legal_mask" in rec and rec["legal_mask"] is not None:
        mask = np.asarray(rec["legal_mask"], dtype=np.float32).reshape(-1)
        if mask.shape[0] < num_moves:
            padded = np.zeros((num_moves,), dtype=np.float32)
            padded[: mask.shape[0]] = mask
            mask = padded
        elif mask.shape[0] > num_moves:
            mask = mask[:num_moves]
        return mask

    action_ids = rec.get("legal_action_ids")
    if action_ids is None:
        raise ValueError(f"No legal_mask or legal_action_ids in record: {rec}")
    mask = np.zeros((num_moves,), dtype=np.float32)
    for a in action_ids:
        a_int = int(a)
        if 0 <= a_int < num_moves:
            mask[a_int] = 1.0
    return mask


def _normalize_obs(obs_vec: Sequence[float], obs_dim: int) -> np.ndarray:
    """Pad or truncate the observation vector to obs_dim."""
    vec = np.asarray(obs_vec, dtype=np.float32).reshape(-1)
    if vec.shape[0] == obs_dim:
        return vec
    if vec.shape[0] < obs_dim:
        out = np.zeros((obs_dim,), dtype=np.float32)
        out[: vec.shape[0]] = vec
        return out
    return vec[:obs_dim]


def _compute_returns(rewards: List[float], discount: float = 1.0) -> List[float]:
    """Compute (optionally discounted) returns G_t from a list of rewards."""
    G: List[float] = [0.0 for _ in rewards]
    running = 0.0
    for idx in reversed(range(len(rewards))):
        running = rewards[idx] + discount * running
        G[idx] = running
    return G


def build_training_samples(
    records: Sequence[Dict],
    num_moves: int,
    obs_dim: int,
    *,
    discount: float = 1.0,
) -> List[Dict]:
    """
    Transform raw SPARTA step records into supervised training samples.

    Each output sample contains:
      - obs_vec: np.ndarray[obs_dim]
      - seat: int
      - prev_other_action: int (reconstructed if absent)
      - legal_mask: np.ndarray[num_moves]
      - action_target: int (SPARTA action)
      - value_target: float (return from this step)
      - is_override: int (1 if blueprint differs, else 0; omitted if unknown)
    """
    grouped = _group_by_game(records)
    samples: List[Dict] = []

    for steps in grouped.values():
        rewards = [float(s.get("reward", 0.0)) for s in steps]
        returns = _compute_returns(rewards, discount=discount)

        last_action_by_seat = {0: num_moves, 1: num_moves}

        for idx, rec in enumerate(steps):
            seat = int(rec.get("seat", 0))
            if seat not in (0, 1):
                raise ValueError(f"Unexpected seat {seat} in record {rec}")

            obs_vec = _normalize_obs(rec["obs_vec"], obs_dim)
            legal_mask = _ensure_mask(num_moves, rec)

            action_sparta = int(rec["action_sparta"])
            prev_other = rec.get("prev_other_action")
            if prev_other is None:
                prev_other = last_action_by_seat[1 - seat]
            prev_other = int(prev_other)

            action_blueprint = rec.get("action_blueprint")
            is_override = None
            if action_blueprint is not None:
                is_override = int(action_sparta != int(action_blueprint))

            samples.append(
                {
                    "obs_vec": obs_vec,
                    "seat": seat,
                    "prev_other_action": prev_other,
                    "legal_mask": legal_mask,
                    "action_target": action_sparta,
                    "value_target": float(returns[idx]),
                    "is_override": is_override,
                }
            )

            # Update opponent-action tracker after the move.
            last_action_by_seat[seat] = action_sparta

    return samples
