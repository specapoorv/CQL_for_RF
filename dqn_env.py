import numpy as np
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# ======================================================
# ACTION SPACE (27 actions)
# ======================================================
action_space = list(itertools.product([-1, 0, 1], repeat=3))
NUM_ACTIONS = len(action_space)   # 27


# ======================================================
# ENVIRONMENT: 3-AP Power Control
# ======================================================
class PowerEnv:
    def __init__(self, min_power, max_power):
        self.min_p = min_power
        self.max_p = max_power

    def step(self, state, action_vec):
        next_state = []

        for p, a in zip(state, action_vec):
            new_p = p + a
            new_p = max(self.min_p, min(self.max_p, new_p))
            next_state.append(new_p)

        reward = self.compute_reward(next_state)
        return next_state, reward

    def compute_reward(self, state):
        return sum(state)  # placeholder
        

# ======================================================
# Q-Network (Deep Neural Network)
# input = [p1,p2,p3], output = Q-values for 27 actions
# ======================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=27):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ======================================================
# Replay Buffer
# ======================================================
class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ======================================================
# DQN Agent
# ======================================================
class DQNAgent:
    def __init__(self, action_dim=27, gamma=0.99, lr=1e-3, epsilon=0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

        self.qnet = QNetwork()
        self.target_qnet = QNetwork()
        self.target_qnet.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        self.replay = ReplayBuffer()

    def select_action(self, state):
        """epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.tensor([state], dtype=torch.float32)
        qvals = self.qnet(state_t)
        return torch.argmax(qvals).item()

    def train_step(self, batch_size=64):
        if len(self.replay) < batch_size:
            return

        states, actions, rewards, next_states = self.replay.sample(batch_size)

        qvals = self.qnet(states)
        qvals = qvals.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_qvals = self.target_qnet(next_states).max(1)[0]
            targets = rewards + self.gamma * next_qvals

        loss = nn.MSELoss()(qvals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ======================================================
# TRAINING LOOP
# ======================================================
if __name__ == "__main__":
    MIN_POWER = 10
    MAX_POWER = 25

    env = PowerEnv(MIN_POWER, MAX_POWER)
    agent = DQNAgent(action_dim=NUM_ACTIONS, epsilon=0.2)

    NUM_EPISODES = 2000

    state = [15, 15, 15]

    for episode in range(NUM_EPISODES):

        action_idx = agent.select_action(state)
        action_vec = action_space[action_idx]

        next_state, reward = env.step(state, action_vec)

        agent.replay.push(state, action_idx, reward, next_state)
        agent.train_step(batch_size=64)

        # slowly update target network
        if episode % 50 == 0:
            agent.target_qnet.load_state_dict(agent.qnet.state_dict())

        state = next_state

    print("Training complete.")
