import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        # Log std for action distribution
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value

    def get_action_std(self):
        return self.log_std.exp()


class PPO:
    def __init__(self, n_agents, state_dim, action_dim, device="cpu"):
        self.n_agents = n_agents
        self.device = device
        self.action_dim = action_dim

        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

        self.gamma = 0.95
        self.gae_lambda = 0.95
        self.eps_clip = 0.2
        self.k_epochs = 10
        self.c1 = 0.5  # value loss coef
        self.c2 = 0.01  # entropy coef

        self.memory = []

    def select_actions(self, observations, training=True):
        actions = {}
        log_probs = {}
        values = {}

        for agent, obs in observations.items():
            state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_mean, value = self.model(state)
                action_std = self.model.get_action_std()

            if training:
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
            else:
                action = action_mean
                log_prob = torch.zeros(1)

            action = torch.clamp(action, -1, 1)

            actions[agent] = action.squeeze(0).cpu().numpy()
            log_probs[agent] = log_prob.item()
            values[agent] = value.item()

        return actions, log_probs, values

    def store_transition(self, states, actions, log_probs, rewards, dones, values):
        for agent in states:
            self.memory.append({
                "state": states[agent],
                "action": actions[agent],
                "log_prob": log_probs[agent],
                "reward": rewards[agent],
                "done": dones[agent],
                "value": values[agent]
            })

    def train(self):
        if len(self.memory) == 0:
            return

        # Convert memory to tensors
        states = torch.FloatTensor([m["state"] for m in self.memory]).to(self.device)
        actions = torch.FloatTensor([m["action"] for m in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([m["log_prob"] for m in self.memory]).to(self.device)
        rewards = [m["reward"] for m in self.memory]
        dones = [m["done"] for m in self.memory]
        old_values = torch.FloatTensor([m["value"] for m in self.memory]).to(self.device)

        # Compute returns and advantages using GAE
        returns = []
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = old_values[t + 1].item()
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - old_values[t].item()
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + old_values[t].item())

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.k_epochs):
            action_mean, state_values = self.model(states)
            action_std = self.model.get_action_std()
            
            dist = torch.distributions.Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)

            loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        self.memory = []

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])