import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import random
import copy
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from array import array



class DDPG():
    def __init__(self, state_dim = 5, action_dim = 1):
        """
        Initializes the DDPG agent.
        Takes three arguments:
               state_dim which is the dimensionality of the state space,
               action_dim which is the dimensionality of the action space, and
               max_action which is the maximum value an action can take.

        Creates a replay buffer, an actor-critic  networks and their corresponding target networks.
        It also initializes the optimizer for both actor and critic networks alog with
        counters to track the number of training iterations.
        """
        self.capacity=10000
        self.batch_size=64
        self.update_iteration=10
        self.tau=0.001 # tau for soft updating
        self.gamma=0.99 # discount factor
        self.directory = './'
        self.hidden1=64 # hidden layer for actor
        self.hidden2=64
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.replay_buffer = Replay_buffer(max_size=self.capacity)

        self.actor = Actor(state_dim, action_dim, self.hidden1).to(self.device)
        self.actor_target = Actor(state_dim, action_dim,  self.hidden1).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-3)

        self.critic = Critic(state_dim, action_dim,  self.hidden2).to(self.device)
        self.critic_target = Critic(state_dim, action_dim,  self.hidden2).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2e-2)
        # learning rate

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def map_state(self, state):
        tyre, condition, cur_weather, radius, laps_cleared = state

        tyre_map = {"Ultrasoft":1,
                     "Soft":2,
                     "Intermediate":3,
                     "Fullwet":4}

        cur_weather_map = {"Dry":1,
                    "20% Wet":2,
                    "40% Wet":3,
                    "60% Wet":4,
                    "80% Wet":5,
                    "100% Wet":6}


        return (np.array([tyre_map[tyre], condition, cur_weather_map[cur_weather], radius, laps_cleared]))

    def act(self, state):
        """
        takes the current state as input and returns an action to take in that state.
        It uses the actor network to map the state to an action.
        """
        torch_state = torch.FloatTensor(self.map_state(state).reshape(1, -1)).to(self.device)
        return round(self.actor(torch_state).cpu().data.numpy().flatten()[0])


    def update(self):
        """
        updates the actor and critic networks using a batch of samples from the replay buffer.
        For each sample in the batch, it computes the target Q value using the target critic network and the target actor network.
        It then computes the current Q value
        using the critic network and the action taken by the actor network.

        It computes the critic loss as the mean squared error between the target Q value and the current Q value, and
        updates the critic network using gradient descent.

        It then computes the actor loss as the negative mean Q value using the critic network and the actor network, and
        updates the actor network using gradient ascent.

        Finally, it updates the target networks using
        soft updates, where a small fraction of the actor and critic network weights are transferred to their target counterparts.
        This process is repeated for a fixed number of iterations.
        """

        for it in range(self.update_iteration):
            # For each Sample in replay buffer batch

            st, n_st, act, rew, is_d = self.replay_buffer.sample(self.batch_size)

            for i in range(self.batch_size):
                
                state = torch.FloatTensor(self.map_state(st[i])).to(self.device)

                action = torch.tensor(act[i]).to(self.device)

                next_state = torch.FloatTensor(self.map_state(n_st[i])).to(self.device)
                done = torch.FloatTensor(is_d[i]).to(self.device)
                reward = torch.FloatTensor(rew[i]).to(self.device)

                # Compute the target Q value
                target_Q = self.critic_target(next_state, self.actor_target(torch.unsqueeze(next_state, 0)))

                target_Q = reward + (done * self.gamma * target_Q).detach()

                # Get current Q estimate
                current_Q = self.critic(state, action)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q, target_Q)

                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                # Compute actor loss as the negative mean Q value using the critic network and the actor network
                actor_loss = -self.critic(state, self.actor(torch.unsqueeze(state, 0))).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


            """
            Update the frozen target models using
            soft updates, where
            tau,a small fraction of the actor and critic network weights are transferred to their target counterparts.
            """
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * self.target_param.data)

            for param, target_param in zip(self.self.actor.parameters(), self.self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        """
        Saves the state dictionaries of the actor and critic networks to files
        """
        torch.save(self.actor.state_dict(), self.directory + 'actor.pth')
        torch.save(self.critic.state_dict(), self.directory + 'critic.pth')

    def load(self):
        """
        Loads the state dictionaries of the actor and critic networks to files
        """
        self.actor.load_state_dict(torch.load(self.directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(self.directory + 'critic.pth'))

class Actor(nn.Module):
    """
    The Actor model takes in a state observation as input and
    outputs an action, which is a continuous value.

    It consists of four fully connected linear layers with ReLU activation functions and
    a final output layer selects one single optimized action for the state
    """
    def __init__(self, n_states, action_dim, hidden1):
        super(Actor, self).__init__()
        self.epsilon = 0.3
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, 5)
        )

    def forward(self, state):
        output = self.net(state)
        
        softmax_output = torch.softmax(output, dim=1)
        if np.random.uniform(0,1) < self.epsilon:
            choice = torch.randint(low=0, high=4,size=())
        else:
            choice = torch.argmax(softmax_output)
        return choice

class Critic(nn.Module):
    """
    The Critic model takes in both a state observation and an action as input and
    outputs a Q-value, which estimates the expected total reward for the current state-action pair.

    It consists of four linear layers with ReLU activation functions,
    State and action inputs are concatenated before being fed into the first linear layer.

    The output layer has a single output, representing the Q-value
    """

    def __init__(self, n_states, action_dim, hidden2):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states + action_dim, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim)
        )

    def forward(self, state, action):
        return self.net(torch.cat((state.reshape(1,5), action.reshape(1,1)),1))



class Replay_buffer():
    '''
    Code referred from :
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    '''
    def __init__(self, max_size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        state: np.array
            batch of state or observations
        action: np.array
            batch of actions executed given a state
        reward: np.array
            rewards received as results of executing action
        next_state: np.array
            next state next state or observations seen after executing action
        done: np.array
            done[i] = 1 if executing ation[i] resulted in
            the end of an episode and 0 otherwise.
        """
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        state, next_state, action, reward, is_done = [], [], [], [], []

        #replay_buffer.push((state, next_state, action, reward, is_done))


        for i in ind:
            st, n_st, act, rew, dn = self.storage[i]
            state.append(st)
            next_state.append(n_st)
            action.append(np.array(act, copy=False))
            reward.append(np.array(rew, copy=False))
            is_done.append(np.array(dn, copy=False))

        return state, next_state, np.array(action), np.array(reward).reshape(-1, 1), np.array(is_done).reshape(-1, 1)
