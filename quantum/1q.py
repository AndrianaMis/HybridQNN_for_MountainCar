
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
from qiskit.quantum_info import Statevector, Pauli

from qiskit.compiler import transpile

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-5
LR_GD= 0.0001
BATCH_SIZE = 16
REPLAY_BUFFER_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 0.95
MAX_EPISODES=1000
MIN_REWARDS=-500
TARGET_UPDATE_FREQ=100
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
import time












# Define observables for a 4-qubit system
observable1 = SparsePauliOp('ZIII')  # Pauli-Z on qubit 1
observable2 = SparsePauliOp('IZII')  # Pauli-Z on qubit 2
observable3 = SparsePauliOp('IIZI')  # Pauli-Z on qubit 3
observable4 = SparsePauliOp('IIIZ')  # Pauli-Z on qubit 4

obs_1=SparsePauliOp('ZI')
obs_2=SparsePauliOp('IZ')

def dummy():
    torch.set_num_threads(8)  # Use 8 CPU threads for matrix computations

    policy=Hybrid_VQC_nn(2,1,2,3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for i in range(5):  # Just 5 iterations to check if it works
        optimizer.zero_grad()
        
        state_batch = torch.rand(5, 2)  # Small random batch
        target = torch.rand(5, 1)  # Fake targets

# Time Forward Pass
        start_time = time.time()
      #  print(f"state  batch : {state_batch}")
        output = policy(state_batch)
        print(f"Forward Pass Time: {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        loss = loss_fn(output, target)
        loss.backward()
        
        print(f"Backward Pass Time: {time.time() - start_time:.4f} seconds")
        
        
 #       grads = policy.vqcs.compute_grads(policy.vqcs.weights_param)   #now updateing the quantum weights
  #      with torch.no_grad():
   #         policy.vqcs.weights_param -= 0.01 * grads


        optimizer.step()
        print(f"Iteration {i+1}: Loss = {loss.item()}")



# Combine observables into a list

#We will create the Quantum VQC layer that has a quantum circuit as a member
class VQCLayer(nn.Module):
    def __init__(self, num_qubits, depth):
        super().__init__()
        self.num_qubits=num_qubits
        self.depth=depth
        self.theta = torch.randn(2, dtype=torch.float32) * 0.1

    def forward(self, x):
        # x is expected to be [batch_size, 2] (for x1 and x2)
        outputs = []
        for x1, x2 in x:
            out = qc(x1.item(), x2.item(), self.theta.detach().numpy())
            outputs.append(out)
        return torch.tensor(outputs, dtype=torch.float32).view(-1, 1)

def compute_batch_gradient(qc, states, thetas, delta_theta=1e-2):
    """
    Compute the average gradient of the quantum circuit over a batch of states.

    Args:
        qc (function): Quantum circuit function: qc(x1, x2, thetas) → float
        states (torch.Tensor): Tensor of shape [batch_size, 2], each row is [x1, x2]
        thetas (torch.Tensor): Parameters of the quantum circuit (1D tensor)
        delta_theta (float): Small delta for parameter shift

    Returns:
        torch.Tensor: Averaged gradient over the batch [same shape as thetas]
    """
    num_params = len(thetas)
    gradients = torch.zeros(num_params)

    for state in states:
        x1, x2 = state[0].item(), state[1].item()

        for i in range(num_params):
            theta_plus = thetas.clone()
            theta_minus = thetas.clone()

            theta_plus[i] += delta_theta
            theta_minus[i] -= delta_theta

            f_plus = qc(x1, x2, theta_plus)
            f_minus = qc(x1, x2, theta_minus)

            grad = (f_plus - f_minus) / (2 * delta_theta)
            gradients[i] += grad

    # Average over the batch
    gradients /= len(states)
    return gradients


def qc(x1,x2,thetas):

    qc = QuantumCircuit(1)

    # Encoding inputs (normalize to [-π, π])
    x1 = np.clip(x1, -1, 1) * np.pi
    x2 = np.clip(x2, -1, 1) * np.pi
    qc.rx(x1, 0)
    qc.ry(x2, 0)

    # Parametrized circuit (only 2 thetas)
    qc.rx(thetas[0].item(), 0)
    qc.rz(thetas[1].item(), 0)


    # Measurement
    state = Statevector.from_instruction(qc)

    # Compute expectation value of Pauli Z
    expectation_value = state.expectation_value(Pauli("Z")).real

    return expectation_value


#"function" for mapping the EXPECTATION VALUES OF OBSERVABLES INTO THE 3 ACTION STATE PAIRS. 
class Net(nn.Module):      #extraction 
    def __init__(self, input_d, output_dim):
        super(Net, self).__init__()
        self.fc1=nn.Linear(input_d, 8)
        self.fc2=nn.Linear(8, output_dim) #introducing a hidden layer of 8 neurons 
    def forward(self, quantum_outputs):
        x = torch.relu(self.fc1(quantum_outputs))
        return self.fc2(x)

class Hybrid_VQC_nn(nn.Module):
    def __init__(self, input_dim, vqc_depth,num_qubits, output_dim):
        super(Hybrid_VQC_nn, self).__init__()
        self.vqcs=VQCLayer(num_qubits, vqc_depth)             
        self.num_qubits=num_qubits
        self.post = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim)  # outputs 3 Q-values
        )    
    def forward(self, x):
        vqc_out = self.vqcs(x)         # [batch, 1]
        q_values = self.post(vqc_out)  # [batch, 2]
        return q_values
            

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


training=False
st=0
def train(replay_buffer, policy, target, optimizer):
    train_start=time.time()
  #  print("Training")
    if replay_buffer.size() < BATCH_SIZE:    #το τραινινγκ γίνεται μόνο όταν υπάρχει αρκετό υλικό 
        return
    
  #  print("\n\nWe begin training!!!")
    global training
    training=True
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    states_tensor = torch.FloatTensor(states).requires_grad_()
    actions_tensor = torch.tensor(actions, dtype=torch.long)
    rewards_tensor = torch.FloatTensor(rewards)
    next_states_tensor = torch.FloatTensor(next_states)
    dones_tensor = torch.FloatTensor(dones)

    #Q(s,a) -> Q(s,a) + a[r + γmaxQ(s',a) - Q(s,a) ] 

    # Compute Q(s, a)
    #states_tensor should be [batchsize, 2]
 #   print(f"states tensor: {states_tensor.shape}")
    net_s=time.time()
  #  print("actions_tensor:", actions_tensor)
  #  print("q_values shape:", policy(states_tensor).shape)
    q_values = policy(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)   #Το policy (main) netwrok βγάζει όλα τα Q values που υπάρχουν για την δοσμένη στειτ

 
  #  print(f"During training, policcy gaves us the 3 qvalues: {q_values}")
      # Compute maxQ(s', a') using the target network
    with torch.no_grad():  
        next_q_values = target(next_states_tensor).max(1)[0]  
    target_q_values = rewards_tensor + (1 - dones_tensor) * GAMMA * next_q_values  
    net_e=time.time()
    net_t=net_e-net_s
  #  print(f"Networkds time: {net_t}") 
  #  print(f"tagert q values: {target_q_values} and shape {target_q_values.shape}")

    # Compute loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(q_values, target_q_values)
    global st
    st+=1
  #  if st % 100==0: print(f"loss: {loss}")


    # Optimize the model
    optimizer.zero_grad()
    start_time=time.time()
    loss.backward()  # Ensure gradients are computed correctly
    end_time=time.time()
    t=end_time-start_time
  #  print(f"Loss backward time: {t}")
    
    o_s=time.time()
    optimizer.step()
    o_e=time.time()
    o_t=o_e-o_s
   # print(f"opt time: {o_t}")
 #   print("training done")
    quantum_gradients = compute_batch_gradient(qc, states_tensor, policy.vqcs.theta)
    with torch.no_grad():
        policy.vqcs.theta[0] -= LEARNING_RATE* quantum_gradients[0]
        policy.vqcs.theta[1] -= LEARNING_RATE* quantum_gradients[1]

       # print(f"thetas: {policy.vqcs.theta}")

    train_end=time.time()
    train_t=train_end-train_start
  #  print(f"Total train time: {train_t}")


def run():
    
    env = gym.make("MountainCar-v0", render_mode=None) 
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(n_actions)
    depth=1
    qubits=2
    policy=Hybrid_VQC_nn(n_states,depth, qubits, n_actions)
    target=Hybrid_VQC_nn(n_states,depth, qubits, n_actions)
   # policy.load_state_dict(torch.load('../Hybrid.pth'))   #load pre trained weigths
    target.load_state_dict(policy.state_dict())
    target.eval()
    '''optimizer = torch.optim.Adam([
        {'params': policy.feature_extractor.parameters(), 'lr': 0.001},
        {'params': policy.vqcs.parameters(), 'lr': 0.01},  # Increase LR for quantum
        {'params': policy.fc.parameters(), 'lr': 0.001}
    ])
    '''
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
              #  print("xploration")

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = torch.argmax(policy(state_tensor)).item()
                


            new_state, reward, terminated, truncated,_=env.step(action)
            if abs(new_state[0] - state[0]) < 1e-3:
                reward -= 0.1
            previous_distance = abs(state[0] - 0.5)
            current_distance = abs(new_state[0] - 0.5)
            distance_reward = previous_distance - current_distance
            reward += distance_reward * 0.9  # Scale appropriately


            replay_buffer.add((state, action, reward, new_state, terminated))
            state=new_state
            rewards+=reward

            train(replay_buffer,policy,target,optimizer)
     #       print(f"Done first training!!!! training:{training}")

    
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

        print(f"Episode {i}, Total Reward: {rewards}, Epsilon: {epsilon:.4f}")
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
    metrics_file="HybridDQN.pkl"
    with open(metrics_file, "wb") as f:
        pickle.dump(metrics, f)
    
    model_file="HybridDQN.pth"
    torch.save(policy.state_dict(),model_file)





if __name__=='__main__':
   # dummy()
    run()