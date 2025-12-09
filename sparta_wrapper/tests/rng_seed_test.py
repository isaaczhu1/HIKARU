import random
frequencies = {}
rng = random.Random(42)
for i in range(1000):
    s = rng.getstate()
    c = rng.choice([1,2])
    if c not in frequencies:
        frequencies[c] = 0
    frequencies[c] += 1
    rng.setstate(s)
print(frequencies)