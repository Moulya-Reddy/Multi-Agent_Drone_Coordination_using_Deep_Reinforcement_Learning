import numpy as np
import torch
from collections import deque
import random


class MultiAgentReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, states, actions, rewards, next_states, dones):
        """
        states: list of n_agents states
        actions: list of n_agents actions
        rewards: list of n_agents rewards
        next_states: list of n_agents next_states
        dones: list of n_agents dones
        """
        self.buffer.append((states, actions, rewards, next_states, dones))
    
    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for s, a, r, ns, d in batch:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)