reps = []
losses = []
with open('eval.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        try:
            rep = float(line.split("R/ep(mean,last100)=")[1].split(" ")[0])
            loss = float(line.split("loss_v=")[1].split(" ")[0])
            reps.append(rep)
            losses.append(loss)
        except (IndexError, ValueError):
            continue

import matplotlib.pyplot as plt

plt.plot(reps)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode')

plt.savefig('reward_per_episode.png')
plt.close()

plt.plot(losses)
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Loss per Episode')

plt.savefig('loss_per_episode.png')
plt.close()