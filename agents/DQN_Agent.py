import numpy as np
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple,deque
from itertools import count
import random
import math
import sys

np.set_printoptions(precision=3)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 48)
        self.layer2 = nn.Linear(48, 48)
        self.layer3 = nn.Linear(48, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class DQN_Agent:
    def __init__(self,n_actions=5,n_observations=5,policy_net_path='policy_net.pt'):
        self.BATCH_SIZE = 1024
        self.GAMMA = 0.97
        self.EPS_START = 128
        self.EPS_END = 0.1
        self.EPS_DECAY = 1000
        self.TAU = 0.1
        self.LR = 5e-3
        self.currentEPS=0

        self.n_actions=n_actions
        self.n_observations=n_observations

        self.policy_net=DQN(self.n_observations,self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer=optim.AdamW(self.policy_net.parameters(),lr=self.LR,amsgrad=True)
        self.memory=ReplayMemory(1048576)
        self.steps_done=0
        self.episode_G=[]

        self.load_trainedModule(policy_net_path)

        pass

    def load_trainedModule(self,policy_net_path):
        self.policy_net=torch.load(policy_net_path)
        self.steps_done=1000
    

    #this function returns tensor, if you want to get integer action, use function act()
    def select_action(self,state):
        sample=random.random()
        eps_threshold=self.EPS_END+(self.EPS_START-self.EPS_END)*math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.currentEPS=eps_threshold
        self.steps_done+=1
        '''if eps_threshold>10:
            return torch.tensor([[2,]], device=device, dtype=torch.long)'''
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randint(0,4)]], device=device, dtype=torch.long)

    def plot_durations(self,show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_G, dtype=torch.float)

        # Take 100 episode averages and plot them too
        means=None
        if len(durations_t) >= 10:
            means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        
        if show_result:
            plt.title('Result:')
        else:
            plt.clf()
            plt.title('Training... EPS={:.2f} RAMDOMPERFORM={:.2f} CURPERFORM={:.2f}'.format(self.currentEPS,means.numpy()[0] if means!=None else 0,means.numpy()[-1] if means!=None else 0))
        plt.xlabel('Episode')
        plt.ylabel('G')
        plt.plot(durations_t.numpy())
        if means!=None:
            plt.plot(means.numpy())

        plt.pause(0.0001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory)<self.BATCH_SIZE:
            return
        transitions=self.memory.sample(self.BATCH_SIZE)
        batch=Transition(*zip(*transitions))

        non_final_mask=torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)),device=device,dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values=self.policy_net(state_batch).gather(1,action_batch)

        next_state_values=torch.zeros(self.BATCH_SIZE,device=device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        criterion=nn.SmoothL1Loss()
        loss=criterion(state_action_values,expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

 
    def stringStateToIntState(self,state):
        tyreDict={}
        tyreDict["Ultrasoft"]=0
        tyreDict["Soft"]=1
        tyreDict["Intermediate"]=2
        tyreDict["Fullwet"]=3

        weatherDict={}
        weatherDict["Dry"]=0
        weatherDict["20% Wet"]=1
        weatherDict["40% Wet"]=2
        weatherDict["60% Wet"]=3
        weatherDict["80% Wet"]=4
        weatherDict["100% Wet"]=5

        return([tyreDict[state[0]],state[1],weatherDict[state[2]],state[3],state[4]])

    def act(self, state):
        
        actionShuffle=[0,3,4,1,2]

        state_tensor=torch.tensor(self.stringStateToIntState(state),dtype=torch.float32,device=device).unsqueeze(0)
        
        action_tensor=self.select_action(state_tensor)

        return actionShuffle[action_tensor.item()]
