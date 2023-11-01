import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

device = torch.device("cpu")

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs,device=device, dtype=torch.float)
        
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output
    
class PPO_Agent:
    def __init__(self,obs_dim=5,act_dim=5,actor_net_path="actor_net.pt"):
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        self.actor=torch.load(actor_net_path)
        self.cov_var=torch.full(size=(self.act_dim,),device=device,fill_value=0.01)
        self.cov_mat=torch.diag(self.cov_var)
        pass

    def stringStateToIntNormedState(self,state):
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

        data=[tyreDict[state[0]]/3,state[1],weatherDict[state[2]]/5,state[3]/1200,state[4]/162]

        return(data)
    
    def act(self,state):
        obs=self.stringStateToIntNormedState(state)
        mean=self.actor(torch.tensor(obs,device=device,dtype=torch.float32))
        dist=MultivariateNormal(mean,self.cov_mat)
        action=dist.sample()
        action=torch.Tensor.cpu(action.detach()).numpy()
        action=np.argmax(action)

        return(action)

