import random
import numpy as np

class SarsaAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, random_seed=0):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.Q = {}
        self.rewards = []
        self.state = None
        random.seed(random_seed)

    def get_Q(self, state, action):
        return self.Q.get((state, action), 0.0)
        # return self.Q.get((state, action), None)

    def greedy_action_selection(self, state):
        q_values = [self.get_Q(state, action) for action in self.actions]
        max_Q = max(q_values)
        best_actions = [i for i in range(len(self.actions)) if q_values[i] == max_Q]
        return random.choice(best_actions)
    
    def act(self, state, epsilon=0.1):
        state_d = self._discretize_state(state)
        if np.random.uniform(0,1) < epsilon:
            return np.random.choice(self.actions)
        else:
            return self.greedy_action_selection(state_d)
        
    def _discretize_state(self, state):
        tyre, condition, cur_weather, radius, laps_cleared = state
        # return (tyre, round(condition, 2), cur_weather, radius, int(laps_cleared))
        return (tyre, round(condition, 2), cur_weather, radius, laps_cleared)
    
    def update(self, state, action, reward, next_state, next_state_action):
        curr_state_d = self._discretize_state(state)
        next_state_d = self._discretize_state(next_state)
        Q_value = self.get_Q(curr_state_d, action)
        td_error = reward + self.gamma * self.get_Q(next_state_d, next_state_action) - Q_value
        Q_value += self.alpha * td_error
        self.Q[(curr_state_d,action)] = Q_value