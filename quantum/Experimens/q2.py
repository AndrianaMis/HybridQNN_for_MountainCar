










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
        self.backend = AerSimulator(method="statevector")
        self.qc , self.inputs, self.weights=create_qc(num_qubits, depth)
        self.weights_param = nn.Parameter(torch.randn(num_qubits * depth) * 0.1, requires_grad=False)   #αρχικά θ για το VQC, θα γίνουν optimized με backpropagation
        self.dummy_inputs=[]

    def forward(self, x):
        batch=x.shape[0]
        self.dummy_inputs=x[0]
        outputs=[]   #[batch, qubits]
        for i in range(batch):   #για κάθε sample πρέπει να το πε΄ρασουμε μέσα από το κύκλωμα και να πάρουεμ ouput
            # Convert input and weights to NumPy
            inputs = {self.inputs[j]: x[i, j].item() for j in range(self.num_qubits)}  #update the data 
            #ta input values είναι τα δεδομένα της συγκεκριμένη γραμμής για κάθε qubit
            weights = {self.weights[j]: self.weights_param[j].item() for j in range(self.num_qubits * self.depth)}  #update 
            param_bindings = {**inputs, **weights}  #this dict will be bind to the circuit in order to get the parameters  
#            print("Feeding the circuit by passing through the layer")
       #     if i==0: print(f"θ παραμετροι σειρά 0: {weights.items()}")
            
            bound_circuit = self.qc.assign_parameters(param_bindings)
            # Transpile and run the circuit
            transpiled_circuit = transpile(bound_circuit, self.backend)
            job = self.backend.run(transpiled_circuit)
            result = job.result()

            # Get the statevector
            statevector = result.get_statevector()
            statevector = np.asarray(statevector)  # Convert to NumPy array
#            print(f"statevector: {statevector[0]}|00>  + {statevector[1]}|01> + {statevector[2]}|10>  + {statevector[3]}|11>")
#            print(f"quantum out shape: {np.array(statevector).shape}   expected: {pow(2,self.num_qubits)}")
            # Convert quantum output to PyTorch tensor (returning real part)

#            print(f"statevector real{statevector.real}")
#Post processing of circuit output, in order to return a [batchsize, 3] tensor as quantum output
            #statvector is in shape 2^qubits , , what we have to hae as ouput is 
            real=statevector.real
            vec=[]
            for i in range(pow(2,self.num_qubits)):
                pr=pow(real[i],2)
                vec.append(pr)
#                print(f"values of statevector was {real[i]} and pr is {pr}")
            #vec should be of size 2^2
            Z0=vec[0] + vec[1] - vec[2] -vec [3]
            Z1= vec[0] + vec[2] -vec[1] -vec[3]
            Z0Z1=vec[0] - vec[1] -vec[2] +vec[3]
            qouts=torch.tensor([Z0, Z1, Z0Z1], dtype=torch.float32)            
                           
            outputs.append(qouts)
            #outputs should be of dim [batch_size,] 
  #      print(f"Finished batch collection of Z0,Z1,Z0Z1. Now the outputs should be [batchsize,3] and is: [{len(outputs)},{len(outputs[0])}]")
            

        return torch.stack(outputs)


        #runs the circuit with the params given and computs f(θ+π/2) ή f(θ+π/2) but must f=do for fixed input
    def parameter_shif(self,params, shift=None, shift_index=None):
        if shift is not None and shift_index is not None:
            params = params.clone()
            params[shift_index] += shift

        # Bind the parameters to the quantum circuit
        inputs = {self.inputs[i]: self.dummy_inputs[i].item() for i in range(len(self.inputs))}
        weights = {self.weights[j]: params[j].item() for j in range(len(params))}
        param_bindings={**inputs, **weights}
        bound_circuit = self.qc.assign_parameters(param_bindings)
        bound_circuit.measure_all()
        transpiled_circuit = transpile(bound_circuit, self.backend)
        job = self.backend.run(transpiled_circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()
    
        # Compute expectation value of the Z observable
        expectation_value = self.compute_expectation(counts)
    
        return torch.tensor(expectation_value, dtype=torch.float32)


#βρισκει [f(θ+π/2) - f(θ-π/2)]/2 που είναι 
    def compute_grads(self,params):
        grads=torch.zeros_like(params)
        for i in range(len(params)):
            plus = self.parameter_shif(params, shift=+torch.pi/2, shift_index=i)
            # Evaluate the circuit with a negative shift
            minus = self.parameter_shif(params, shift=-torch.pi/2, shift_index=i)
            # Compute the gradient using the parameter shift rule
            grads[i] = (plus - minus) / 2
        return grads
            
    def compute_expectation(self, counts):
        """Computes expectation value of the Pauli-Z observable from measurement counts."""
        shots = sum(counts.values())  # Total number of shots
        expectation = 0
    
        for bitstring, count in counts.items():
            parity = (-1) ** bitstring.count('1')  # +1 if even parity, -1 if odd parity
            probability = count / shots  # Estimate probability of bitstring
            expectation += parity * probability  # Weighted sum
    
        return expectation



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
   # qc.measure_all()
    qc.save_statevector()
    return qc , inputs , weights


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
        self.feature_extractor = nn.Sequential(            # 2 (states)  -> 4 (qubits)
            nn.Linear(input_dim, num_qubits),
            nn.ReLU()
        )
        
        self.vqcs=VQCLayer(num_qubits, vqc_depth)             
        #fully connected
        self.num_qubits=num_qubits
        self.postprocess = Net(3, output_dim)
    def forward(self, x):
    #    x=self.feature_extractor(x)
     #   print(f"\ninput on main network: {x} with size {len(x)}")
  #      print(f"We gave input to the main network, must be [batch, 2] and is :{ x.shape}")
        x=torch.atan(x)    #input data will be converted to [-π/2,π/2] and passed to encoding layer of VQC
     #   print(f" converted input x: {x} with size {len(x)}")

        q_out=self.vqcs(x)
   #     print(f"\nq_out shape: {q_out.shape}")
        #  print("We have the q out and should do some preprocessing for the netwrok extraction")
        
        q_values = self.postprocess(q_out)
   #     print(f"This is the end: we have wvalues {q_values} of size {q_values.shape}\n")
    
            
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
    actions_tensor = torch.LongTensor(actions)
    rewards_tensor = torch.FloatTensor(rewards)
    next_states_tensor = torch.FloatTensor(next_states)
    dones_tensor = torch.FloatTensor(dones)

    #Q(s,a) -> Q(s,a) + a[r + γmaxQ(s',a) - Q(s,a) ] 

    # Compute Q(s, a)
    #states_tensor should be [batchsize, 2]
 #   print(f"states tensor: {states_tensor.shape}")
    net_s=time.time()
    q_values = policy(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)   #Το policy (main) netwrok βγάζει όλα τα Q values που υπάρχουν για την δοσμένη στειτ
  #  print(f"During training, policcy gaves us the 3 qvalues: {q_values}")
      # Compute maxQ(s', a') using the target network
    with torch.no_grad():  
        next_q_values = target(next_states_tensor).max(1)[0]  
    target_q_values = rewards_tensor + (1 - dones_tensor) * GAMMA * next_q_values  
    net_e=time.time()
    net_t=net_e-net_s
    print(f"Networkds time: {net_t}") 
  #  print(f"tagert q values: {target_q_values} and shape {target_q_values.shape}")

    # Compute loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(q_values, target_q_values)
    global st
    st+=1
    if st % 100==0: print(f"loss: {loss}")


    # Optimize the model
    optimizer.zero_grad()
    start_time=time.time()
    loss.backward()  # Ensure gradients are computed correctly
    end_time=time.time()
    t=end_time-start_time
    print(f"Loss backward time: {t}")
    start_time=time.time()

    grads = policy.vqcs.compute_grads(policy.vqcs.weights_param)   #now updateing the quantum weights
    with torch.no_grad():
        policy.vqcs.weights_param -= 0.01 * grads
    end_time=time.time()
    t=end_time-start_time
    print(f"Grad desc time : {t}")
    o_s=time.time()
    optimizer.step()
    o_e=time.time()
    o_t=o_e-o_s
    print(f"opt time: {o_t}")
 #   print("training done")
    train_end=time.time()
    train_t=train_end-train_start
    print(f"Total train time: {train_t}")
def run():
    
    env = gym.make("MountainCar-v0", render_mode=None) 
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(n_states)
    depth=1
    qubits=2
    policy=Hybrid_VQC_nn(n_states,depth, qubits, n_actions)
    target=Hybrid_VQC_nn(n_states,depth, qubits, n_actions)
   # policy.load_state_dict(torch.load('../Hybrid.pth'))   #load pre trained weigths
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