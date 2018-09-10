#########################################
# KAIST Summer Session 2018             #
# Deep Q-Network for Self-Driving Car   #
#########################################


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# Implementing Deep Neural Networks
class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.fc4 = nn.Linear(10, nb_action)


    def forward(self, state):
        q = torch.tanh(self.fc1(state))
        q = torch.tanh(self.fc2(q))
        q = torch.tanh(self.fc3(q))
        q = self.fc4(q)
        return q

    
# Implementing Experience Replay
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
        
    def push(self, event):    # event: last state, new state, last action, new reward
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return samples


# Implementing Deep Q-Network Agent
class DQN():

    def __init__(self, input_size, nb_action, gamma, epsilon):
                
        # We separate target and learning networks
        self.model = Network(input_size, nb_action)
        self.target_nn = Network(input_size, nb_action)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.nb_action = nb_action
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.005)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.number_of_frames = 0
    
    
    def select_action(self, state):        
        
        # ϵ-Greedy Approach
        random_for_egreedy = random.random()
        
        if random_for_egreedy > self.epsilon:    
     
            with torch.no_grad():  # no gradient calculation (not for learning)
                state = torch.Tensor(state)
                action_from_nn = self.model(state)
                
                # choose the action with maximum Q-value from neural networks
                action = torch.max(action_from_nn[0], 0)[1]
                action = action.item() 
            
        else:
            action = random.randint(0, self.nb_action-1)
        
        return action
    
    
    def optimize(self, batch_state, batch_next_state, batch_reward, batch_action):

        # We separate target and learning networks        
        next_outputs = self.target_nn(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)

        td_loss = F.smooth_l1_loss(outputs, target)
        
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        # Copy target and learning networks         
        if self.number_of_frames % 10 == 0:
            self.target_nn.load_state_dict(self.model.state_dict())
        
        self.number_of_frames += 1
    
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        
        # Learning using random sample from experience memory
        if len(self.memory.memory) > 100:
            samples = self.memory.sample(100)
            batch_state, batch_next_state, batch_action, batch_reward = map(lambda x: torch.cat(x, 0), samples)
            self.optimize(batch_state, batch_next_state, batch_reward, batch_action)
            
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward

        return action
