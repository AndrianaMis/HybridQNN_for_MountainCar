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

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-5
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 0.95
MAX_EPISODES=1000
MIN_REWARDS=-1000
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

    policy=Hybrid_VQC_nn(2,3,4,3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for i in range(5):  # Just 5 iterations to check if it works
        optimizer.zero_grad()
        
        state_batch = torch.rand(4, 2)  # Small random batch
        target = torch.rand(4, 1)  # Fake targets

# Time Forward Pass
        start_time = time.time()
        output = policy(state_batch)
        print(f"Forward Pass Time: {time.time() - start_time:.4f} seconds")

# Time Backward Pass
        start_time = time.time()
        loss = loss_fn(output, target)
        
        loss.backward()
        print(f"Backward Pass Time: {time.time() - start_time:.4f} seconds")

        optimizer.step()

        print(f"Iteration {i+1}: Loss = {loss.item()}")



# Combine observables into a list

#We will create the Quantum VQC layer that has a quantum circuit as a member
class VQCLayer(nn.Module):
    def __init__(self, num_qubits, depth):
        super().__init__()
        self.num_qubits=num_qubits
        self.depth=depth
        self.backend = AerSimulator(method="statevector", device="GPU")
        self.qc , self.inputs, self.weights=create_qc(num_qubits, depth)
        self.weights_param = nn.Parameter(torch.randn(num_qubits * depth) * 0.1)

    def forward(self, x):
        batch=x.shape[0]
        outputs=[]   #[batch, qubits]
        for i in range(batch):
            # Convert input and weights to NumPy
            input_values = {self.inputs[j]: x[i, j].item() for j in range(self.num_qubits)}
            weight_values = {self.weights[j]: self.weights_param[j].item() for j in range(self.num_qubits * self.depth)}
            param_bindings = {**input_values, **weight_values}

            # Transpile and run circuit
           # transpiled_circuit = transpile(self.qc, self.backend)
            result = self.backend.run(self.qc).result()
            statevector = result.get_statevector()
            print(f"Statevector shape: {np.array(statevector).shape}")
            # Convert quantum output to PyTorch tensor (returning real part)
            outputs.append(torch.tensor(statevector.real, device=device))

        return torch.stack(outputs)

def create_qc(num_qubits, depth):
       #Both inputs and weights (trainable parameters) are created as Qiskit's Parameters.
    qc=QuantumCircuit(num_qubits)
    inputs=ParameterVector('x', num_qubits)   #not to be optimized
    weights=ParameterVector('θ',num_qubits * depth) 
        


#depth is for data reuploading and repetition  of the layer 
    for d in range(depth):
        for i in range(num_qubits):            #data re-uploading
            qc.ry(inputs[i],i)
        #qc.barrier()
       # qc.barrier()

        for i in range(num_qubits):
            qc.ry(weights[i+ num_qubits*d], i)
       # qc.barrier()
        for i in range(num_qubits - 1):  # CZ Entanglement
            qc.cz(i, i + 1)
    #qc.barrier()
    #qc.barrier()
    qc.measure_all()
    return qc , inputs , weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
class Hybrid_VQC_nn(nn.Module):
    def __init__(self, input_dim, vqc_depth,num_qubits, output_dim):
        super(Hybrid_VQC_nn, self).__init__()
        self.feature_extractor = nn.Sequential(            # 2 (states)  -> 4 (qubits)
            nn.Linear(input_dim, num_qubits),
            nn.ReLU()
        ).to(device)
        
        self.vqcs=VQCLayer(num_qubits, vqc_depth)             
        #fully connected
        self.fc=nn.Linear(num_qubits, output_dim).to(device)   # 4 (qubits)  ->   3 (actions - q-values)
    def forward(self, x):
        x=self.feature_extractor(x)
     #   print(f"input x: {x} with size {len(x)}")
        x=torch.atan(x)    #input data will be converted to [-π/2,π/2] and passed to encoding layer of VQC
    #  print(f" converted input x: {x} with size {len(x)}")

        q_out=self.vqcs(x)
      #  print(f"q_out shape: {q_out.shape}")

        return self.fc(q_out)

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
  #  print("Training")
    if replay_buffer.size() < BATCH_SIZE:    #το τραινινγκ γίνεται μόνο όταν υπάρχει αρκετό υλικό 
        return
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    states_tensor = torch.FloatTensor(states).requires_grad_()
    actions_tensor = torch.LongTensor(actions)
    rewards_tensor = torch.FloatTensor(rewards)
    next_states_tensor = torch.FloatTensor(next_states)
    dones_tensor = torch.FloatTensor(dones)

    #Q(s,a) -> Q(s,a) + a[r + γmaxQ(s',a) - Q(s,a) ] 

    # Compute Q(s, a)
    q_values = policy(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)   #Το policy (main) netwrok βγάζει όλα τα Q values που υπάρχουν για την δοσμένη στειτ
      # Compute maxQ(s', a') using the target network
    with torch.no_grad():  
        next_q_values = target(next_states_tensor).max(1)[0]  
    target_q_values = rewards_tensor + (1 - dones_tensor) * GAMMA * next_q_values   

    # Compute loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(q_values, target_q_values)
  #  print(f"loss: {loss}")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()  # Ensure gradients are computed correctly
    optimizer.step()
 #   print("training done")

def run():
    
    env = gym.make("MountainCar-v0", render_mode=None) 
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(n_states)
    depth=1
    qubits=2
    policy=Hybrid_VQC_nn(n_states,depth, qubits, n_actions)
    target=Hybrid_VQC_nn(n_states,depth, qubits, n_actions)
    policy.load_state_dict(torch.load('../Hybrid.pth'))   #load pre trained weigths
    target.load_state_dict(policy.state_dict())
    target.eval()
    optimizer = torch.optim.Adam([
        {'params': policy.feature_extractor.parameters(), 'lr': 0.001},
        {'params': policy.vqcs.parameters(), 'lr': 0.01},  # Increase LR for quantum
        {'params': policy.fc.parameters(), 'lr': 0.001}
    ])
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
        while (not terminated and not truncated and rewards> MIN_REWARDS):

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
            reward += distance_reward * 0.5  # Scale appropriately


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
    dummy()
  #  run()