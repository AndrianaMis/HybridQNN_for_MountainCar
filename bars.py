import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load metrics from pkl files
def load_metrics(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

# Compute relevant average metrics
def summarize_metrics(metrics):
    median_penalty = np.abs(np.median(metrics["rewards_per_episode"]))
    success_rate = np.sum(metrics["success_rate"])   
    avg_steps = np.mean(metrics["steps_list"])
    
    return [median_penalty, success_rate, avg_steps]

# File paths
classical_path = 'classics/DQN_smallerLR.pkl'
hybrid_path = 'quantum/HybridDQN_withH.pkl'

# Load and summarize
classical_metrics = summarize_metrics(load_metrics(classical_path))
hybrid_metrics = summarize_metrics(load_metrics(hybrid_path))

# Bar plot
labels = ['Absolute Median Episode Penalty', 'Success Rate', 'Average Steps per Episode']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
metric_colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd']  # red, green, blue, purple

bars1 = ax.bar(x - width/2, classical_metrics, width, label='Classical', color="#0D5080")  # blue
bars2 = ax.bar(x + width/2, hybrid_metrics, width, label='Hybrid', color="#820c0c")        # orange

ax.set_ylabel('Metric Value')
ax.set_title('Performance Comparison: Classical vs Hybrid RL Models')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Annotate bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
