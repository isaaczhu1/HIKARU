from hanabi_learning_environment import pyhanabi
state = pyhanabi.HanabiGame({'players':2}).new_initial_state()
print([name for name in dir(state) if not name.startswith('__')])
