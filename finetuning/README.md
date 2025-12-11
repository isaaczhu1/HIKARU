## SPARTA Supervised Fine-Tuning

This directory contains a minimal pipeline to distill SPARTA rollouts into the existing Hanabi GRU policy. It does **not** run SPARTA itself; it only consumes logs produced by a SPARTA wrapper.

### Expected log format (per decision)
Each record should include:
- `game_id`: int
- `t`: timestep within the game
- `seat`: current player id
- `obs_vec`: vectorized observation (same shape as `HanabiEnv2P._pack_obs["obs"]`)
- `legal_mask` **or** `legal_action_ids`: legality aligned to your action-id mapping
- `action_sparta`: int action id chosen by SPARTA
- `reward`: shaped reward for this step
- `done`: bool (game ended after this step)

Optional:
- `prev_other_action`: sentinel or last opponent action id (if omitted we reconstruct from the log)
- `action_blueprint`: blueprint argmax action (used to mark overrides)
- `Q_values`: SPARTA Q estimates (ignored for v1)

Files can be JSONL (one record per line) or a JSON array.

### Preprocessing
`finetuning.data.build_training_samples` groups steps by `game_id`, sorts by `t`, computes undiscounted returns, reconstructs `prev_other_action` if missing, and outputs per-step supervision targets:
- `obs_vec`, `seat`, `prev_other_action`, `legal_mask`
- `action_target` (SPARTA action), `value_target` (return)
- `is_override` (1 if SPARTA != blueprint when provided)

### Training
Run the trainer (example):
```bash
python -m finetuning.train_sparta \
  --data sparta_games.jsonl \
  --ckpt runs/hanabi/standard_train/ckpt_020000.pt \
  --out runs/hanabi/finetune_sparta/ckpt_ft.pt \
  --epochs 2 \
  --batch-size 1024 \
  --lr 5e-5
```

Loss per batch:
- Policy imitation: cross-entropy to `action_sparta` over masked logits (optional override weighting)
- Value regression: MSE to returns
- Small entropy bonus on the masked policy
Total: `L = L_pi + lambda_value * L_value - lambda_entropy * H`

The GRU state is reset per sample (seq_len=1) for simplicity. Gradients are clipped with the same norm as PPO.

### Outputs
`train_sparta.py` saves a checkpoint compatible with `HanabiGRUPolicy` (same architecture as the source PPO model). Use it as a new blueprint or evaluate it directly with your existing eval scripts. Iterate by generating new SPARTA logs with the updated blueprint and repeating the fine-tune.
