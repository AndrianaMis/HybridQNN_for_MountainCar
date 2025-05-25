import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import pickle 
import matplotlib.pyplot as plt
import sys
# Hyperparameters
import time
GAMMA = 0.99
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
REPLAY_BUFFER_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 0.99
MAX_EPISODES=1000
MIN_REWARDS=-500
TARGET_UPDATE_FREQ=100   



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def size(self):
        return len(self.buffer)

def train(replay_buffer, policy, target, optimizer):
    train_start=time.time()
    if replay_buffer.size() < BATCH_SIZE:    #το τραινινγκ γίνεται μόνο όταν υπάρχει αρκετό υλικό 
        return
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)
    rewards_tensor = torch.FloatTensor(rewards)
    next_states_tensor = torch.FloatTensor(next_states)
    dones_tensor = torch.FloatTensor(dones)

    #Q(s,a) -> Q(s,a) + a[r + γmaxQ(s',a) - Q(s,a) ] 

    # Compute Q(s, a)
    q_values = policy(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)   #Το policy (main) netwrok βγάζει όλα τα Q values που υπάρχουν για την δοσμένη στειτ

    
    # Compute maxQ(s', a') from target network
    next_q_values = target(next_states_tensor).max(1)[0]  #Το target βγάζει τα q values για το next state και διαλέγεται το max 
    target_q_values = rewards_tensor + (1 - dones_tensor) * GAMMA * next_q_values   #INSIDE []

    # Compute loss
    loss = nn.MSELoss()(q_values, target_q_values)   #The TD ERROR : r + γmaxQ(s',a) - Q(s,a)    is simply the loss
  #  print(f"loss: {loss}")
    # Optimize the model
    #The whole update (after ->) is done by optimizer
    optimizer.zero_grad()
    l_s=time.time()
    loss.backward()
    l_e=time.time()
    l_t=l_e-l_s
   # print(f"loss back time: {l_t}")
    optimizer.step()
    train_end=time.time()
    train_t=train_end-train_start
    #print(f"Total train time: {train_t}")





def run():
    
    env = gym.make("MountainCar-v0", render_mode=None) 
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(n_states)
    policy=DQN(n_states,n_actions)
    target=DQN(n_states,n_actions)

    target.load_state_dict(policy.state_dict())
    target.eval()
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    steps_list=[]
    epsilon=EPSILON_START
    
    rewards_per_episode=[]
    sucess_rate=[]
    for i in range(MAX_EPISODES):
        state=env.reset()[0] 
        
        terminated=False
        truncated=False
        rewards=0
        goal_reached=False
        steps=0
        while (not terminated and rewards> MIN_REWARDS):

            if np.random.rand() < epsilon:
                action=env.action_space.sample()  #O = drive left , 1=stay , 2=drive rigth
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = torch.argmax(policy(state_tensor)).item()
                


            
            new_state, reward, terminated, truncated,_=env.step(action)
           # if abs(new_state[0] - state[0]) < 1e-3:
            #    reward -= 0.1
            previous_distance = abs(state[0] - 0.5)
            current_distance = abs(new_state[0] - 0.5)
            distance_reward = previous_distance - current_distance
            reward += distance_reward * 0.2  # Scale appropriately


            replay_buffer.add((state, action, reward, new_state, terminated))
            state=new_state
            rewards+=reward

            train(replay_buffer,policy,target,optimizer)
            pos,vel=state
            steps+=1
            if(pos>=0.5):
                print(f'Goal reached in episode {i}')
                goal_reached=True

        if goal_reached==True:
            sucess_rate.append(1)
            steps_list.append((i,steps))
        else:
            sucess_rate.append(0)
        epsilon=max(EPSILON_END, epsilon*EPSILON_DECAY)
        rewards_per_episode.append(rewards)
        if i % TARGET_UPDATE_FREQ == 0:
            target.load_state_dict(policy.state_dict())

      #  print(f"Episode {i}, Total Reward: {rewards}, Epsilon: {epsilon:.4f}")
    env.close()

    window=10
    smoothed_rewards_per_episode=np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')
    parameters_used ={
        "lr":LEARNING_RATE,
        "gamma":GAMMA,
        "epsilon_start":EPSILON_START,
        "epsilon_end":EPSILON_END,
        "epsilon_decay":EPSILON_DECAY,
        "batch":BATCH_SIZE,
        "replay_size":REPLAY_BUFFER_SIZE,
        "episodes": MAX_EPISODES,
        "min_rewards": MIN_REWARDS,
        "target_net_update_freq": TARGET_UPDATE_FREQ


    }
    metrics= {
        "rewards_per_episode": rewards_per_episode,
        "success_rate": sucess_rate,
        "steps_list": steps_list,
        "smoothed_rewards": smoothed_rewards_per_episode.tolist(),
        "parameters": parameters_used
    }
    metrics_file="DQN_no.pkl"
    with open(metrics_file, "wb") as f:
        pickle.dump(metrics, f)
    
    model_file="dqn_weights.pth"
    torch.save(policy.state_dict(),model_file)






if __name__=='__main__':

    run()