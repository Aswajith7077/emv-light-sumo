import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_size=9,
        num_actions=2,
        alpha=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        replay_buffer_size=10000,
        batch_size=64,
        target_update_freq=500,
    ):

        self.name = "Deep Q-Network (DQN)"
        self.state_size = state_size
        self.num_actions = num_actions

        self.alpha = alpha  # Learning Rate
        self.gamma = gamma  # Discount Factor
        self.epsilon = epsilon_start  # Initial epsilon
        self.epsilon_end = epsilon_end  # Final epsilon
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        self.batch_size = batch_size  # Batch size
        self.target_update_freq = target_update_freq  # Target network update frequency

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Build networks
        self.model = self._build_model()  # Online Q-network
        self.target_model = self._build_model()  # Target Q-network
        self._sync_target_network()

        self.training_steps = 0

    def _build_model(self):
        return QNetwork(self.state_size, self.num_actions)

    def _sync_target_network(self):
        # Copy weights from online network to target network.
        self.target_model.load_state_dict(self.model.state_dict())

    def _predict(self, model, state_array):

        with torch.no_grad():
            tensor = torch.FloatTensor(state_array)
            return model(tensor).numpy()[0]

    def select_action(self, state):

        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        # Exploitation
        else:
            state_array = np.array(state, dtype=np.float32).reshape(1, -1)
            q_values = self._predict(self.model, state_array)
            return int(np.argmax(q_values))

    def __decompose_batch(self, batch):
        states = np.array([e[0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def learn(self, state, action, reward, next_state, done):

        # Store experience
        self.replay_buffer.append((state, action, reward, next_state, done))

        # Only train after we have enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample mini-batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = self.__decompose_batch(batch)

        self._train_batch(states, actions, rewards, next_states, dones)

        # Decaying of epsilon (Significance of Exploration decreases as increase in trials)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_steps += 1

        # Sync the network periodically, until that period, the network outputs a constant y value.
        if self.training_steps % self.target_update_freq == 0:
            self._sync_target_network()

    def _train_batch(self, states, actions, rewards, next_states, dones):

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        # Current Q-values
        current_q = self.model(states_t).gather(1, actions_t).squeeze()

        # Target Q-values
        with torch.no_grad():
            max_next_q = self.target_model(next_states_t).max(1)[0]
            targets = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = self.model.loss_fn(current_q, targets)
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

    def get_stats(self):
        """Return current agent statistics."""
        return {
            "epsilon": self.epsilon,
            "replay_buffer_size": len(self.replay_buffer),
            "training_steps": self.training_steps,
        }

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"  DQN model saved to {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self._sync_target_network()
        print(f"  DQN model loaded from {path}")
