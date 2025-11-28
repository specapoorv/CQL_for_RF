import numpy as np 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class dqn(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(dqn, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class dqnagent:
    def __init__(self, state_vec, action_vec, alpha, gamma, epsilon):
        self.state = state_vec #should be list []
        self.action = action_vec #should be action []
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        state_dim = 
        action_dim = 
        # #initialising q table
        # num_states = len(self.state)
        # num_actions = len(self.action)
        # self.qtable = np.zeroes((num_states, num_actions)) #array

        # this is replace by neural net

        self.qnet = dqn(state_dim, action_dim)
        self.target_qnet = dqn(state_dim, action_dim)
        self.target_qnet.load_state_dict(self.qnet.state_dict()) # matching both networks w&b which was initialised randomly
        self.replay_buffer = []
        

    def transition(self, state):
        # actions = self.qtable[state] #query the Q table 
        actions = self.qnet.forward(state)
        if random.random() > self.epsilon:
            #take max Q action = exploitation by querying the Q table, high epsilon means more exploration
            best_action = np.argmax(actions)
            return best_action, next_state, reward
        
        else:
           best_action =  np.random.choice(len(actions))
           return best_action, next_state, reward
            

    def update(self, state, action, next_state, reward):

        old_value = self.qtable[state, action]
        next_max = np.max(self.qtable[next_state])
        target = reward + self.gamma * next_max
        self.qtable[state, action] = old_value + self.alpha * (target - old_value)

        
    
if __name__ == "__main__":
    NUM_SWEEPS = 1000
    initial_state = None

    agent = dqnagent(state_vec, action_vec, alpha, gamma, epsilon)
    state = initial_state

    for i in NUM_SWEEPS:
        action, next_state, reward = agent.transition(initial_state)
        agent.replay_buffer.append(state, action, reward, next_state)
        agent.update(initial_state, action, next_state, reward)

        




    

        