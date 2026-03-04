# src/agent_ddqn.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CustomReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, state, action, reward, next_state, done):
        """ Saves a transition to memory. If the buffer is at capacity, it randomly deletes an old memory before adding the new one. """
        if len(self.memory) >= self.capacity:
            half_capacity = self.capacity // 2
            
            index_to_delete = random.randint(0, half_capacity - 1)
            self.memory.pop(index_to_delete)
            
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Samples a random batch of transitions for training."""
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, state_dim = 6, action_dim = 11):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DDQNAgent:
    def __init__(self, state_dim=6, action_dim=11, lr=2.5e-4, gamma=0.99,
                buffer_capacity=100000, batch_size=64, 
                eps_start=1.0, eps_end=0.01, eps_decay=0.99980, 
                target_update_freq=2000):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0 
        
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # Double DQN Architecture (Policy Net + Target Net)
        # Policy Net: Actively learns and makes decisions
        self.policy_net = QNetwork(state_dim, action_dim)

        # Target Net: A slow-updating copy used strictly to evaluate future rewards safely
        self.target_net = QNetwork(state_dim, action_dim)

        # The Target Net is initialized with the same weights as the Policy Net.
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # We only train the Policy Net. The Target Net is updated periodically to match the Policy Net.
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay Buffer
        self.memory = CustomReplayBuffer(buffer_capacity)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            best_action = q_values.argmax(dim=1).item()

        return best_action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)


    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute Q-values from the Policy Network
        current_q_values = self.policy_net(states).gather(1, actions)

        # Compute the next Q-values and the best actions from the Policy Network
        with torch.no_grad():
            best_next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, best_next_actions)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute the loss between current Q-values and target Q-values
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Zeros the gradients
        self.optimizer.zero_grad()

        # Calculates how much each weight in layers contributed to that mistake
        loss.backward()
        
        # Clip the gradients to prevent them from exploding
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        # Update
        self.optimizer.step()

        # Update the Target Network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())