# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 13:09:21 2023

@author: Ali
"""

import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 2e-4               # learning rate 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, cls, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            cls (string): player class
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.cls = cls
        self.seed = seed
        random.seed(seed)

        # Actor
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)

        # Critic
        self.critic_local = Critic(state_size, action_size, 2, seed).to(device)
        self.critic_target = Critic(state_size, action_size, 2, seed).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        self.oun = ornstein_uhlenbeck_noise(np.zeros(action_size, dtype=float), np.zeros(action_size, dtype=float), 0.1, 0.15, 1)

    def step(self, state, player_action, opponent_action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, player_action, opponent_action, reward, next_state, done)
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, player_action, opponent_action, rewards, next_states, dones = experiences
        action_embedded = torch.cat((player_action, opponent_action), dim=1)

        action_target = self.actor_target.forward(next_states)
        action_target_embedded = torch.cat((action_target, opponent_action), dim=1)
        
        new_state_value_target = self.critic_target.forward(next_states, action_target_embedded)
        state_value_local = self.critic_local.forward(states, action_embedded)

        targets = rewards + (gamma * new_state_value_target * (1 - dones))

        # Compute loss
        critic_loss = F.mse_loss(targets, state_value_local)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()



        action_local = self.actor_local.forward(states)
        action_local_embedded = torch.cat((action_local, opponent_action), dim=1)

        # Compute loss
        actor_loss = torch.mean(-1 * self.critic_local.forward(states, action_local_embedded))
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()



        # ------------------- update target network ------------------- #
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)
        
    
    def act(self, state, noise=False):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local.forward(state)
        self.actor_local.train()

        if noise:
            noise = torch.from_numpy(next(self.oun)).float().unsqueeze(0).to(device)
            value = action_values + noise
        else:
            value = action_values
        
        value = torch.clamp(value, -1, 1)
        
        return value.cpu().data.numpy()
    
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    
    def save_model(self):
        actor_state_dict = self.actor_local.state_dict()
        critic_state_dict = self.critic_local.state_dict()
        torch.save(actor_state_dict, 'models/{}/actor_checkpoint.pth'.format(self.cls))
        torch.save(critic_state_dict, 'models/{}/critic_checkpoint.pth'.format(self.cls))
        
    def load_model(self):
        actor_state_dict = torch.load('models/{}/actor_checkpoint.pth'.format(self.cls))
        critic_state_dict = torch.load('models/{}/critic_checkpoint.pth'.format(self.cls))
        self.actor_local.load_state_dict(actor_state_dict)
        self.actor_target.load_state_dict(actor_state_dict)
        self.critic_local.load_state_dict(critic_state_dict)
        self.critic_target.load_state_dict(critic_state_dict)
    
    def reset_models(self):
        self.actor_local.reset_parameters()
        self.actor_target.reset_parameters() 
        self.critic_local.reset_parameters()
        self.critic_target.reset_parameters() 


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "player_action", "opponent_action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, player_action, opponent_action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, player_action, opponent_action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        player_action = torch.from_numpy(np.vstack([e.player_action for e in experiences if e is not None])).float().to(device)
        opponent_action = torch.from_numpy(np.vstack([e.opponent_action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, player_action, opponent_action, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



def ornstein_uhlenbeck_noise(initialValue, mu, sigma=0.2, theta=0.15, dt=1e-2):
    currentValue = initialValue
    while True:
        currentValue += (theta * (mu - currentValue) * dt) + (sigma * np.sqrt(dt) * np.random.normal(size=currentValue.size))
        yield currentValue
