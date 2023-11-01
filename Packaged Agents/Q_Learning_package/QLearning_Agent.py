#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
np.set_printoptions(precision=3)


class QLearningAgent:
    def __init__(self, n_actions=5, alpha=0.2, gamma=0.9, epsilon=0.9, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def _discretize_state(self, state):
        tyre, condition, cur_weather, radius, laps_cleared = state
        return (tyre, round(condition, 2), cur_weather, radius, int(laps_cleared))

    def act(self, state):
        discrete_state = self._discretize_state(state)
        self.epsilon = self.epsilon * self.epsilon_decay 
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.get_q_values(discrete_state))

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def update(self, state, action, reward, next_state):
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        current_q = self.get_q_values(discrete_state)[action]
        next_max_q = np.max(self.get_q_values(discrete_next_state))
        self.q_table[discrete_state][action] = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_max_q)

