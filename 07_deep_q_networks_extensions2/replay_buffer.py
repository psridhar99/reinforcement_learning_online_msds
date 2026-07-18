"""Experience replay buffer for off-policy training (see module 5)."""

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device="cpu"):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
