import gymnasium as gym
import numpy as np
from collections import deque
import random
import pickle 
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit import ParameterVector

from qiskit.compiler import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
import time
from q2 import Hybrid_VQC_nn , create_qc, ReplayBuffer


# Assume `hybrid_dqn` is your hybrid model
pretrained_weights = torch.load("../classics/dqn_weights.pth")



GAMMA = 0.99
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
REPLAY_BUFFER_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 0.99
MAX_EPISODES=200
MIN_REWARDS=-500
TARGET_UPDATE_FREQ=100   
t=0
def train(replay_buffer, policy, target, optimizer):
    if replay_buffer.size() < BATCH_SIZE:
        return
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)
    rewards_tensor = torch.FloatTensor(rewards)
    next_states_tensor = torch.FloatTensor(next_states)
    dones_tensor = torch.FloatTensor(dones)

    q_values = policy(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    next_q_values = target(next_states_tensor).max(1)[0]
    target_q_values = rewards_tensor + (1 - dones_tensor) * GAMMA * next_q_values
    global t
    t+=1
    loss = nn.MSELoss()(q_values, target_q_values)
    if t%20==0:
        print(f"loss: {loss}")
        print(f"thetas: {policy.vqcs.weights_param}")

    optimizer.zero_grad()
    loss.backward()

    # Ensure only trainable (unfrozen) layers get updated
    for param in policy.parameters():
        if param.requires_grad:
            param.grad = param.grad  # Keep gradients only for trainable parameters

    
def run():
    env = gym.make("MountainCar-v0", render_mode=None) 
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize Hybrid Model
    policy = Hybrid_VQC_nn(n_states,1,2, n_actions)  # Assuming HybridDQN includes quantum layers
    target = Hybrid_VQC_nn(n_states,1,2, n_actions)
    policy.vqcs.weights_param.requires_grad_=True
    target.vqcs.weights_param.requires_grad_=True

    # Load Pretrained Classical Weights
    policy.load_state_dict(pretrained_weights, strict=False)  # strict=False allows missing layers
    target.load_state_dict(pretrained_weights, strict=False)  # strict=False allows missing layers

    target.eval()
    # Freeze Classical Layers
    for name, param in policy.named_parameters():
        if "vqcs" not in name:  # Adjust based on your quantum layer names
           # print(f"{name} -> {param}")

            param.requires_grad = False  # Freeze classical layers

    target.load_state_dict(policy.state_dict())
    target.eval()

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    steps_list = []
    epsilon = EPSILON_START
    rewards_per_episode = []
    success_rate = []

    for i in range(MAX_EPISODES):
        state = env.reset()[0]
        terminated = False
        truncated = False
        rewards = 0
        goal_reached = False
        steps = 0

        while not terminated and rewards > MIN_REWARDS:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = torch.argmax(policy(state_tensor)).item()

            new_state, reward, terminated, truncated, _ = env.step(action)
            if abs(new_state[0] - state[0]) < 1e-3:
                reward -= 0.1
            previous_distance = abs(state[0] - 0.5)
            current_distance = abs(new_state[0] - 0.5)
            distance_reward = previous_distance - current_distance
            reward += distance_reward * 0.5

            replay_buffer.add((state, action, reward, new_state, terminated))
            state = new_state
            rewards += reward

            train(replay_buffer, policy, target, optimizer)

            pos, vel = state
            steps += 1
            if pos >= 0.5:
                print(f'Goal reached in episode {i}')
                goal_reached = True

        if goal_reached:
            success_rate.append(1)
            steps_list.append((i, steps))
        else:
            success_rate.append(0)

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        rewards_per_episode.append(rewards)

        if i % TARGET_UPDATE_FREQ == 0:
            target.load_state_dict(policy.state_dict())
        print(f"Episode {i}, Total Reward: {rewards}, Epsilon: {epsilon:.4f}")


    env.close()


run()
