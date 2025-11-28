import numpy as np
import random


# ======================================================
# ENVIRONMENT API
# Replace this function with your real API call
# ======================================================
def env_step(state, action):
    """
    Sends (state, action) to your API and returns (next_state, reward).
    This is just a placeholder – REPLACE with your real environment.
    """
    # EXAMPLE DUMMY LOGIC (REMOVE THIS):
    next_state = random.randint(0, NUM_STATES - 1)
    reward = random.uniform(-1, 1)
    return next_state, reward


# ======================================================
# Q-Learning Agent
# ======================================================
class QAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table initialized to zeros
        self.qtable = np.zeros((num_states, num_actions))

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() > self.epsilon:
            # exploit
            return np.argmax(self.qtable[state])
        else:
            # explore
            return random.randint(0, self.num_actions - 1)

    def update(self, state, action, next_state, reward):
        """Standard Q-learning update rule."""
        old_value = self.qtable[state, action]
        next_max = np.max(self.qtable[next_state])

        target = reward + self.gamma * next_max
        self.qtable[state, action] = old_value + self.alpha * (target - old_value)


# ======================================================
# TRAINING LOOP
# ======================================================
if __name__ == "__main__":
    NUM_STATES = 10          # change this to your real number of states
    NUM_ACTIONS = 5          # change this to your real number of actions
    NUM_EPISODES = 1000

    agent = QAgent(
        num_states=NUM_STATES,
        num_actions=NUM_ACTIONS,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2
    )

    state = 0  # starting state (whatever makes sense for your environment)

    for episode in range(NUM_EPISODES):

        # 1. agent chooses action
        action = agent.select_action(state)

        # 2. call your API environment
        next_state, reward = env_step(state, action)

        # 3. update Q-table
        agent.update(state, action, next_state, reward)

        # 4. move to next state
        state = next_state

    # done
    print("Training finished.")
    print(agent.qtable)
