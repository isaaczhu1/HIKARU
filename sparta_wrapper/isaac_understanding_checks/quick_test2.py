from hanabi_learning_environment import pyhanabi
from hanabi_gru_baseline.hanabi_envs import HanabiEnv2P

env2p = HanabiEnv2P(seed=0)
ot = env2p._env.game.observation_type()
print("observation_type =", ot, "name =", pyhanabi.AgentObservationType(ot).name)
print(env2p._env.game.parameter_string())
print("vectorized shape =", env2p._env.vectorized_observation_shape())
