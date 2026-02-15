import random
import numpy as np
import json

class QLearningAgent:
    def __init__(self, num_actions=2, alpha=0.1, gamma=0.95, epsilon_start=1, epsilon_end=0.05, epsilon_decay=0.9995):
        self.name = 'Q-Learning'
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = {}
        self.training_steps = 0

    def get_q_value(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        return self.q_table[state]
    
    def select_action(self, state):

        # Exploration 
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        # Exploitation
        else:
            q_values = self.get_q_value(state)
            return int(np.argmax(q_values))
        
    def learn(self, state, action, reward, next_state, done):
        q_values = self.get_q_value(state)
        next_q_values = self.get_q_value(next_state)

        old_q = q_values[state]
        best_next_q = np.max(next_q_values) if not done else 0
        td_target = reward + self.gamma * best_next_q
        q_values[action] = old_q + self.alpha * (td_target - old_q)

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_steps += 1
    
    def get_stats(self):
        return {
            'q_table_size':len(self.q_table),
            'epsilon':self.epsilon,
            'training_steps':self.training_steps
        }

    def save(self, path):
        serializable = {}
        for state, qvals in self.q_table.items():
            key = str(state)
            serializable[key] = qvals.tolist()

        data = {
            'q_table': serializable,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'  Q-Learning agent saved to {path}')

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)

        self.q_table = {}
        for key, qvals in data['q_table'].items():
            state = eval(key)
            self.q_table[state] = np.array(qvals)

        self.epsilon = data.get('epsilon', self.epsilon_end)
        self.training_steps = data.get('training_steps', 0)
        print(f'  Q-Learning agent loaded from {path}')
        print(f'  Q-table size: {len(self.q_table)}, epsilon: {self.epsilon:.4f}')