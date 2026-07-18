"""Dueling DQN agent: epsilon-greedy action selection plus the learning update."""

import random

import torch
import torch.nn as nn
import torch.optim as optim

from dueling_q_network import DuelingQNetwork
from replay_buffer import ReplayBuffer


class DuelingDQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=128,
        lr=5e-4,
        gamma=0.99,
        buffer_capacity=100_000,
        batch_size=64,
        tau=0.005,
        device=None,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DuelingQNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DuelingQNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._soft_update_target()

        return loss.item()

    def _soft_update_target(self):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
