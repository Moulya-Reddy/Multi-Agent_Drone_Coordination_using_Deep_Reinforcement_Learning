import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQN:
    def __init__(self, n_agents, state_dim, action_dim, device):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Discretize continuous actions into grid
        # 9 actions: 8 directions + stay
        self.discrete_actions = [
            np.array([0, 0]),      # 0: stay
            np.array([1, 0]),      # 1: right
            np.array([-1, 0]),     # 2: left
            np.array([0, 1]),      # 3: up
            np.array([0, -1]),     # 4: down
            np.array([0.7, 0.7]),  # 5: up-right
            np.array([-0.7, 0.7]), # 6: up-left
            np.array([0.7, -0.7]), # 7: down-right
            np.array([-0.7, -0.7]) # 8: down-left
        ]
        self.n_discrete_actions = len(self.discrete_actions)

        # Networks
        self.q_net = QNetwork(state_dim, self.n_discrete_actions).to(device)
        self.target_net = QNetwork(state_dim, self.n_discrete_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)

        self.memory = deque(maxlen=50000)

        self.gamma = 0.95
        self.batch_size = 128
        self.epsilon = 1.0
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.05
        self.target_update_freq = 10
        self.train_step = 0

    def select_actions(self, obs, training=True):
        actions = {}
        indices = {}

        for agent, state in obs.items():
            if training and np.random.rand() < self.epsilon:
                idx = np.random.randint(0, self.n_discrete_actions)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_net(state_t)
                    idx = torch.argmax(q_values).item()

            action = self.discrete_actions[idx].copy()
            actions[agent] = action
            indices[agent] = idx

        return actions, indices

    def store_transition(self, states, indices, rewards, next_states, dones):
        for agent in states:
            self.memory.append((
                states[agent],
                indices[agent],
                rewards[agent],
                next_states[agent],
                dones[agent]
            ))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor([b[0] for b in batch]).to(self.device)
        actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_states = torch.FloatTensor([b[3] for b in batch]).to(self.device)
        dones = torch.FloatTensor([b[4] for b in batch]).to(self.device)

        # Current Q values
        q_values = self.q_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_value = next_q_values.max(1)[0]
            target = rewards + self.gamma * next_q_value * (1 - dones)

        loss = nn.MSELoss()(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']