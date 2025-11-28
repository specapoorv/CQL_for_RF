import numpy as np 
import random


class qagent:
    def __init__(self, state_vec, action_vec, alpha, gamma, epsilon):
        self.state = state_vec #should be list []
        self.action = action_vec #should be action []
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        #initialising q table
        num_states = len(self.state)
        num_actions = len(self.action)
        self.qtable = np.zeroes((num_states, num_actions)) #array

    def transition(self, state):
        actions = self.qtable[state] #query the Q table 
        if random.random() > self.epsilon:
            #take max Q action = exploitation by querying the Q table, high epsilon means more exploration
            best_action = np.argmax(actions)
            return best_action, next_state, reward
        
        else:
           best_action =  np.random.choice(len(actions))
           return best_action
        
    def env_step(state, action):
        # sends state + action to API
        # returns next_state, reward
        return next_state, reward

            

    def update(self, state, action, next_state, reward):

        old_value = self.qtable[state, action]
        next_max = np.max(self.qtable[next_state])
        target = reward + self.gamma * next_max
        self.qtable[state, action] = old_value + self.alpha * (target - old_value)

        
    
if __name__ == "__main__":
    NUM_SWEEPS = 1000
    initial_state = None

    agent = qagent(state_vec, action_vec, alpha, gamma, epsilon)


    for i in NUM_SWEEPS:
        action, next_state, reward = agent.transition(initial_state)
        agent.update(initial_state, action, next_state, reward)

        




    

        