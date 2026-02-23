import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.net(x)


class MultiAgentReplayBuffer:
    def __init__(self, capacity=100000):
        from collections import deque
        self.buffer = deque(maxlen=capacity)
    
    def push(self, states, actions, rewards, next_states, dones):
        self.buffer.append((states, actions, rewards, next_states, dones))
    
    def sample(self, batch_size, device):
        import random
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
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class MADDPG:
    def __init__(self, n_agents, state_dim, action_dim, device="cpu"):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Create actors for each agent
        self.actors = [Actor(state_dim, action_dim).to(device) for _ in range(n_agents)]
        self.target_actors = [Actor(state_dim, action_dim).to(device) for _ in range(n_agents)]

        # Centralized critic
        total_state_dim = state_dim * n_agents
        total_action_dim = action_dim * n_agents

        self.critic = Critic(total_state_dim, total_action_dim).to(device)
        self.target_critic = Critic(total_state_dim, total_action_dim).to(device)

        # Optimizers
        self.actor_optims = [optim.Adam(a.parameters(), lr=1e-4) for a in self.actors]
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.buffer = MultiAgentReplayBuffer(capacity=100000)
        self.gamma = 0.95
        self.tau = 0.01
        self.batch_size = 256
        
        # Noise for exploration
        self.noise_scale = 0.2
        self.noise_decay = 0.9995
        self.noise_min = 0.05

        self.update_targets(tau=1.0)

    def update_targets(self, tau=None):
        if tau is None:
            tau = self.tau

        # Update target actors
        for i in range(self.n_agents):
            for target_param, param in zip(self.target_actors[i].parameters(), 
                                           self.actors[i].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Update target critic
        for target_param, param in zip(self.target_critic.parameters(), 
                                       self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def select_actions(self, observations, add_noise=True):
        actions = {}

        for i, agent in enumerate(observations):
            state = torch.FloatTensor(observations[agent]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.actors[i](state).cpu().numpy()[0]

            if add_noise:
                noise = np.random.normal(0, self.noise_scale, size=self.action_dim)
                action += noise
                
            action = np.clip(action, -1, 1)
            actions[agent] = action

        return actions

    def store_transition(self, states, actions, rewards, next_states, dones):
        state_arr = []
        action_arr = []
        reward_arr = []
        next_state_arr = []
        done_arr = []

        for agent in states:
            state_arr.append(states[agent])
            action_arr.append(actions[agent])
            reward_arr.append(rewards[agent])
            next_state_arr.append(next_states[agent])
            done_arr.append(dones[agent])

        self.buffer.push(state_arr, action_arr, reward_arr, next_state_arr, done_arr)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size, self.device
        )

        batch_size = states.shape[0]

        # Reshape for centralized processing
        states_flat = states.view(batch_size, -1)
        actions_flat = actions.view(batch_size, -1)
        next_states_flat = next_states.view(batch_size, -1)

        # Use mean reward (cooperative setting)
        rewards_mean = rewards.mean(dim=1, keepdim=True)
        dones_mean = dones.max(dim=1, keepdim=True)[0]

        # ===== Critic Update =====
        # Compute target actions using target actors
        next_actions = []
        for i in range(self.n_agents):
            next_state_i = next_states[:, i, :]
            next_action_i = self.target_actors[i](next_state_i)
            next_actions.append(next_action_i)
        
        next_actions_flat = torch.cat(next_actions, dim=1)

        # Compute target Q value
        with torch.no_grad():
            target_q = self.target_critic(next_states_flat, next_actions_flat)
            y = rewards_mean + self.gamma * target_q * (1 - dones_mean)

        # Current Q value
        current_q = self.critic(states_flat, actions_flat)
        critic_loss = nn.MSELoss()(current_q, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # ===== Actor Update =====
        # Compute current actions using current actors
        current_actions = []
        for i in range(self.n_agents):
            state_i = states[:, i, :]
            action_i = self.actors[i](state_i)
            current_actions.append(action_i)
        
        current_actions_flat = torch.cat(current_actions, dim=1)

        # Actor loss is negative Q value
        actor_loss = -self.critic(states_flat, current_actions_flat).mean()

        # Update all actors
        for optim in self.actor_optims:
            optim.zero_grad()
        
        actor_loss.backward()
        
        for i in range(self.n_agents):
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_optims[i].step()

        # Update target networks
        self.update_targets()
        
        # Decay noise
        if self.noise_scale > self.noise_min:
            self.noise_scale *= self.noise_decay

    def save(self, path):
        torch.save({
            "actors": [a.state_dict() for a in self.actors],
            "critic": self.critic.state_dict(),
            "actor_optims": [opt.state_dict() for opt in self.actor_optims],
            "critic_optim": self.critic_optim.state_dict(),
            "noise_scale": self.noise_scale
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for i, a in enumerate(self.actors):
            a.load_state_dict(ckpt["actors"][i])
        self.critic.load_state_dict(ckpt["critic"])
        for i, opt in enumerate(self.actor_optims):
            opt.load_state_dict(ckpt["actor_optims"][i])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.noise_scale = ckpt["noise_scale"]
        self.update_targets(tau=1.0)