import pickle
import torch
import matplotlib.pyplot as plt

with open("classics/DQN_smallerLR.pkl", "rb") as f:
    training_metrics = pickle.load(f)


rewards_per_episode=training_metrics["rewards_per_episode"]


steps=[]
for i,st in training_metrics["steps_list"]:
    steps.append(st)
episodes = range(len(rewards_per_episode))
success_rate=training_metrics["success_rate"]
episodes_success=[]
for ep,st in training_metrics["steps_list"]:
    episodes_success.append(ep)

print(f'The agent reached the goal {training_metrics["success_rate"].count(1)} times and that is out of {len(training_metrics["success_rate"])} episodes')
# Plot average reward
plt.figure()
plt.plot(episodes, rewards_per_episode, label="Rewards")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Agent Learning Curve")
plt.legend()
plt.savefig("pngs_for_thesis/LCURVE.png")
#plt.show()

# Plot success rate
plt.figure()
plt.plot(episodes, success_rate, label="Success Rate")
plt.xlabel("Episodes")
plt.ylabel("Success Rate")
plt.title("Agent Success Rate")
plt.legend()
plt.savefig("pngs_for_thesis/SR.png")
#plt.show()


plt.figure()
plt.plot(episodes_success, steps, label="Steps")
plt.xlabel("Episodes")
plt.ylabel("Steps")
plt.title("Steps to reach goal")
plt.legend()
plt.savefig("pngs_for_thesis/Steps.png")
#plt.show()