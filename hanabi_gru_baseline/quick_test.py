from hanabi_envs import HanabiEnv2P
import numpy as np

e = HanabiEnv2P(seed=0)
obs = e.reset()
print("Seat at reset:", obs["seat"], "obs_shape:", obs["obs"].shape)
print("Sum legal at reset:", float(obs["legal_mask"].sum()))

assert obs["legal_mask"].sum() > 0, "No legal actions at reset!"

for t in range(6):
    legal = obs["legal_mask"]
    idxs = np.flatnonzero(legal)
    a = int(idxs[0]) if idxs.size > 0 else 0
    obs, r, done, info = e.step(a)
    print(f"t={t} seat={obs['seat']} sum_legal={float(obs['legal_mask'].sum())} reward={r} done={done}")
    if done:
        obs = e.reset()
        print("  (reset) seat:", obs["seat"], "sum_legal:", float(obs["legal_mask"].
        sum()))
