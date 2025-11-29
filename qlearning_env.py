import numpy as np
import random
import itertools


# ======================================================
# ACTION SPACE (27 actions)
# ======================================================
action_space = list(itertools.product([-1, 0, 1], repeat=3))
NUM_ACTIONS = len(action_space)   # 27


# ======================================================
# ENVIRONMENT: 3-AP Power Control
# state = [p1, p2, p3]
# ======================================================
class PowerEnv:
    def __init__(self, min_power, max_power):
        self.min_p = min_power
        self.max_p = max_power

    def step(self, state, action_vec):
        """
        state: [p1, p2, p3] (list of ints)
        action_vec: (-1, +1, 0) etc
        """

        next_state = []

        for p, a in zip(state, action_vec):
            new_p = p + a
            new_p = max(self.min_p, min(self.max_p, new_p))  # clamp
            next_state.append(new_p)

        reward = self.compute_reward(next_state)
        return next_state, reward

    def compute_reward(self, state):
        return sum(state)   # placeholder


# ======================================================
# STATE ENCODING
# Convert vector [p1,p2,p3] to unique integer index for Q-table
# ======================================================
def encode_state(state, min_p, max_p):
    """
    Example:
    Powers = [3,4,5], min=0, max=10  
    Encode into integer for Q-table indexing.
    """
    base = max_p - min_p + 1
    return (state[0] * base * base) + (state[1] * base) + state[2]


def decode_state(index, min_p, max_p):
    """The reverse of encode_state()."""
    base = max_p - min_p + 1
    p1 = index // (base * base)
    p2 = (index % (base * base)) // base
    p3 = index % base
    return [p1, p2, p3]


# ======================================================
# Q-Agent using TABLE (not DQN)
# ======================================================
class QAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.qtable = np.zeros((num_states, num_actions))

    def select_action(self, state_idx):
        if random.random() > self.epsilon:
            return np.argmax(self.qtable[state_idx])
        else:
            return random.randint(0, self.num_actions - 1)

    def update(self, s_idx, a, next_s_idx, reward):
        old = self.qtable[s_idx, a]
        next_max = np.max(self.qtable[next_s_idx])
        target = reward + self.gamma * next_max
        self.qtable[s_idx, a] = old + self.alpha * (target - old)


# ======================================================
# TRAINING LOOP
# ======================================================
if __name__ == "__main__":

    # Power constraints
    MIN_POWER = 10
    MAX_POWER = 25

    env = PowerEnv(MIN_POWER, MAX_POWER)

    # total number of states in Q-table
    base = MAX_POWER - MIN_POWER + 1
    NUM_STATES = base * base * base  # (11^3 = 1331)

    agent = QAgent(
        num_states=NUM_STATES,
        num_actions=NUM_ACTIONS,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2
    )

    # initial power levels
    state = [15, 15, 15]

    NUM_EPISODES = 1000

    for episode in range(NUM_EPISODES):

        state_idx = encode_state(state, MIN_POWER, MAX_POWER)

        # pick action index
        action_idx = agent.select_action(state_idx)

        # convert action index → (a1,a2,a3)
        action_vec = action_space[action_idx]

        # take step in the environment
        next_state, reward = env.step(state, action_vec)
        next_state_idx = encode_state(next_state, MIN_POWER, MAX_POWER)

        # update Q-table
        agent.update(state_idx, action_idx, next_state_idx, reward)

        # move to next state
        state = next_state

    print("Training finished.")
    print(agent.qtable)
