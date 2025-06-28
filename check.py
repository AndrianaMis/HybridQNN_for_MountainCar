import pickle
import torch
import matplotlib.pyplot as plt

with open("quantum/HybridDQN_withH.pkl", "rb") as f:
    training_metrics = pickle.load(f)


rewards_per_episode=training_metrics["rewards_per_episode"]
losses_per_episode=training_metrics["losses"]

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
plt.plot(episodes, rewards_per_episode, label="Rewards", color="b")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Agent Learning Curve")
plt.legend()
plt.savefig("quantum/pngs/withHadamard/LearningCurve.png")

#plt.show()
plt.figure()
plt.plot(episodes, losses_per_episode, label="Loss")
plt.xlabel("Episodes")
plt.ylabel("Total Loss")
plt.title("Agent Loss Curve")
plt.legend()
plt.savefig("quantum/pngs/withHadamard/LossCurve.png")

# Plot success rate
plt.figure()
plt.plot(episodes, success_rate, label="Success Rate", color="g")
plt.xlabel("Episodes")
plt.ylabel("Success Rate")
plt.title("Agent Success Rate")
plt.legend()
plt.savefig("quantum/pngs/withHadamard/SuccesRate.png")
#plt.show()


plt.figure()
plt.plot(episodes_success, steps, label="Steps", color="r")
plt.xlabel("Episodes")
plt.ylabel("Steps")
plt.title("Steps to reach goal")
plt.legend()
plt.savefig("quantum/pngs/new/Steps.png")
#plt.show()