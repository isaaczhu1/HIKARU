"""
Quick script to list all action IDs and their decoded rl_env payloads
according to HanabiEnv2P's mapping.
"""

from hanabi_gru_baseline.hanabi_envs import HanabiEnv2P


def list_actions():
    env = HanabiEnv2P(seed=0, obs_conf="minimal")
    actions = []
    for gid in range(env.num_moves):
        actions.append((gid, env._rl_action_from_id(gid)))
    for gid, action in actions:
        print(f"{gid:02d}: {action}")


if __name__ == "__main__":
    list_actions()
