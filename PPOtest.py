import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import random
from torch.distributions import MultivariateNormal
from torch.optim import Adam

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
    
class PPO:
    def __init__(self,obs_dim,act_dim,env):
        self._init_hyperparameters()

        self.env=env
        self.obs_dim=obs_dim
        self.act_dim=act_dim

        self.actor=FeedForwardNN(self.obs_dim,self.act_dim).to(device)
        self.critic=FeedForwardNN(self.obs_dim,1).to(device)

        self.actor_optim=Adam(self.actor.parameters(),lr=self.lr)
        self.critic_optim=Adam(self.critic.parameters(),lr=self.lr)

        self.cov_var=torch.full(size=(self.act_dim,),device=device,fill_value=0.5)
        self.cov_mat=torch.diag(self.cov_var)

        self.best_avg_so_far=None

        self.max_grad_norm=0.5

    
    def learn(self,total_timesteps):
        t_so_far=0
        while t_so_far<total_timesteps:
            batch_obs,batch_acts,batch_log_probs,batch_rtgs,batch_lens=self.rollout()

            t_so_far+=np.sum(batch_lens)

            V,_=self.evaluate(batch_obs,batch_acts)

            A_k=batch_rtgs-V.detach()

            A_k=(A_k-A_k.mean())/(A_k.std()+1e-10)

            for _ in range(self.n_updates_per_iteration):
                V,curr_log_probs=self.evaluate(batch_obs,batch_acts)
                
                ratios=torch.exp(curr_log_probs-batch_log_probs)

                surr1=ratios*A_k
                surr2=torch.clamp(ratios,1-self.clip,1+self.clip)*A_k

                actor_loss=(-torch.min(surr1,surr2)).mean()
                critic_loss=nn.MSELoss()(V,batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm) 
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

            print(t_so_far)

    def evaluate(self,batch_obs,batch_acts):
        V=self.critic(batch_obs).squeeze()

        mean=self.actor(batch_obs)
        dist=MultivariateNormal(mean,self.cov_mat)
        log_probs=dist.log_prob(batch_acts)

        return V,log_probs
            
    def _init_hyperparameters(self):
        self.timesteps_per_batch=40000
        self.max_timesteps_per_episode=3200
        self.gamma=0.98
        self.n_updates_per_iteration=2
        self.clip=0.02
        self.lr=0.01

        self.set=None

    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        t=0
        while t<self.timesteps_per_batch:
            ep_rews=[]
            obs=self.env.reset()
            done=False

            G=0
            for ep_t in range(self.max_timesteps_per_episode):
                t+=1
                batch_obs.append(obs)
                action,log_prob=self.get_action(obs)

                ngz_p=[i-min(action) for i in action]
                #print("emmmm:{}".format(ngz_p))
                sampled_action=random.choices([0,1,2,3,4],weights=ngz_p,k=1)
                #print("     why:{}".format(sampled_action))
                #sampled_action=[0]

                reward,obs,done,_=self.env.transition(sampled_action[0])
                G+=reward

                ep_rews.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    #print("done,G={:.2f},radius={:.2f},G/radius={:.2f}".format(G,self.env.radius,G/env.radius))
                    break
            
            batch_lens.append(ep_t+1)
            batch_rews.append(ep_rews)
        
        avg=np.mean(batch_rews)
        print("batch over, avg reward={:.2f}".format(avg))
        if self.best_avg_so_far!=None:
            if avg>self.best_avg_so_far:
                self.best_avg_so_far=avg
                print("progress!")
                torch.save(self.actor,"acter_net.py")
        else:
            self.best_avg_so_far=avg

        
        batch_obs=torch.tensor(batch_obs,device=device,dtype=torch.float32)
        batch_acts=torch.tensor(batch_acts,device=device,dtype=torch.float32)
        batch_log_probs=torch.tensor(batch_log_probs,device=device,dtype=torch.float32)

        batch_rtgs=self.compute_rtgs(batch_rews)

        return batch_obs,batch_acts,batch_log_probs,batch_rtgs,batch_lens
    
    def get_action(self,obs):
        mean=self.actor(torch.tensor(obs,device=device,dtype=torch.float32))
        dist=MultivariateNormal(mean,self.cov_mat)
        action=dist.sample()
        log_prob=dist.log_prob(action)
        return torch.Tensor.cpu(action.detach()).numpy(),log_prob.detach()
    
    def compute_rtgs(self,batch_rews):
        batch_rtgs=[]

        for ep_rews in reversed(batch_rews):
            discounted_reward=0
            for reward in reversed(ep_rews):
                discounted_reward=reward+discounted_reward*self.gamma
                batch_rtgs.insert(0,discounted_reward)
        
        batch_rtgs=torch.tensor(batch_rtgs,device=device,dtype=torch.float32)
        return batch_rtgs

class Agent:
    def __init__(self,n_actions,n_observations):
        pass
    
    def act(self, state):
        # Simple-minded agent that always select action 1
        return 1
    
def stringStateToIntState(state):
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



class Car:
    def __init__(self, tyre="Intermediate"):
        self.default_tyre = tyre
        self.possible_tyres = ["Ultrasoft", "Soft", "Intermediate", "Fullwet"]
        self.pitstop_time = 23
        self.reset()
    
    
    def reset(self):
        self.change_tyre(self.default_tyre)
    
    
    def degrade(self, w, r):
        if self.tyre == "Ultrasoft":
            self.condition *= (1 - 0.0050*w - (2500-r)/90000)
        elif self.tyre == "Soft":
            self.condition *= (1 - 0.0051*w - (2500-r)/93000)
        elif self.tyre == "Intermediate":
            self.condition *= (1 - 0.0052*abs(0.5-w) - (2500-r)/95000)
        elif self.tyre == "Fullwet":
            self.condition *= (1 - 0.0053*(1-w) - (2500-r)/97000)
        
        
    def change_tyre(self, new_tyre):
        assert new_tyre in self.possible_tyres
        self.tyre = new_tyre
        self.condition = 1.00
    
    
    def get_velocity(self):
        if self.tyre == "Ultrasoft":
            vel = 80.7*(0.2 + 0.8*self.condition**1.5)
        elif self.tyre == "Soft":
            vel = 80.1*(0.2 + 0.8*self.condition**1.5)
        elif self.tyre == "Intermediate":
            vel = 79.5*(0.2 + 0.8*self.condition**1.5)
        elif self.tyre == "Fullwet":
            vel = 79.0*(0.2 + 0.8*self.condition**1.5)
        return vel

    
class Track:
    def __init__(self, car=Car()):
        # self.radius and self.cur_weather are defined in self.reset()
        self.total_laps = 162
        self.car = car
        self.possible_weather = ["Dry", "20% Wet", "40% Wet", "60% Wet", "80% Wet", "100% Wet"]
        self.wetness = {
            "Dry": 0.00, "20% Wet": 0.20, "40% Wet": 0.40, "60% Wet": 0.60, "80% Wet": 0.80, "100% Wet": 1.00
        }
        self.p_transition = {
            "Dry": {
                "Dry": 0.987, "20% Wet": 0.013, "40% Wet": 0.000, "60% Wet": 0.000, "80% Wet": 0.000, "100% Wet": 0.000
            },
            "20% Wet": {
                "Dry": 0.012, "20% Wet": 0.975, "40% Wet": 0.013, "60% Wet": 0.000, "80% Wet": 0.000, "100% Wet": 0.000
            },
            "40% Wet": {
                "Dry": 0.000, "20% Wet": 0.012, "40% Wet": 0.975, "60% Wet": 0.013, "80% Wet": 0.000, "100% Wet": 0.000
            },
            "60% Wet": {
                "Dry": 0.000, "20% Wet": 0.000, "40% Wet": 0.012, "60% Wet": 0.975, "80% Wet": 0.013, "100% Wet": 0.000
            },
            "80% Wet": {
                "Dry": 0.000, "20% Wet": 0.000, "40% Wet": 0.000, "60% Wet": 0.012, "80% Wet": 0.975, "100% Wet": 0.013
            },
            "100% Wet": {
                "Dry": 0.000, "20% Wet": 0.000, "40% Wet": 0.000, "60% Wet": 0.000, "80% Wet": 0.012, "100% Wet": 0.988
            }
        }
        self.reset()
    
    
    def reset(self):
        self.radius = 600
        self.radius = np.random.randint(600,1201)
        self.cur_weather = np.random.choice(self.possible_weather)
        self.is_done = False
        self.pitstop = False
        self.laps_cleared = 0
        self.car.reset()
        return stringStateToIntState(self._get_state())
    
    
    def _get_state(self):
        return [self.car.tyre, self.car.condition, self.cur_weather, self.radius, self.laps_cleared]
        
    
    def transition(self, action=0):
        """
        Args:
            action (int):
                0. Make a pitstop and fit new ‘Ultrasoft’ tyres
                1. Make a pitstop and fit new ‘Soft’ tyres
                2. Make a pitstop and fit new ‘Intermediate’ tyres
                3. Make a pitstop and fit new ‘Fullwet’ tyres
                4. Continue the next lap without changing tyres
        """
        ## Pitstop time will be added on the first eight of the subsequent lap
        time_taken = 0
        if self.laps_cleared == int(self.laps_cleared):
            if self.pitstop:
                self.car.change_tyre(self.committed_tyre)
                time_taken += self.car.pitstop_time
                self.pitstop = False
        
        ## The environment is coded such that only an action taken at the start of the three-quarters mark of each lap matters
        if self.laps_cleared - int(self.laps_cleared) == 0.75:
            if action < 4:
                self.pitstop = True
                self.committed_tyre = self.car.possible_tyres[action]
            else:
                self.pitstop = False
        
        self.cur_weather = np.random.choice(
            self.possible_weather, p=list(self.p_transition[self.cur_weather].values())
        )
        # we assume that degration happens only after a car has travelled the one-eighth lap
        velocity = self.car.get_velocity()
        time_taken += (2*np.pi*self.radius/8) / velocity
        reward = 0-time_taken
        #reward=0-action
        self.car.degrade(
            w=self.wetness[self.cur_weather], r=self.radius
        )
        self.laps_cleared += 0.125
        
        if self.laps_cleared == self.total_laps:
            self.is_done = True
        
        next_state = stringStateToIntState(self._get_state())
        return reward, next_state, self.is_done, velocity

G_list=[]
model=torch.load("actor_net.pt")

for i in range(50):
    new_car = Car()
    env = Track(new_car)
    agent = Agent(5,5)

    holder = []

    state = env.reset()
    start_state = deepcopy(state)
    done = False
    G = 0
    while not done:
        holder.append(env.cur_weather)
        action = agent.act(state)
        reward, next_state, done, velocity = env.transition(action)
        # added velocity for sanity check
        state = deepcopy(next_state)
        G += reward

    agent=PPO(5,5,None)
    agent.actor=model
    agent.cov_var=torch.full(size=(agent.act_dim,),device=device,fill_value=0.01)
    agent.cov_mat=torch.diag(agent.cov_var)

    start_weather, radius = start_state[2], start_state[3]
    state = env.reset() 
    env.cur_weather = start_weather   # assert common start weather
    env.radius = 1200               # assert common track radius
    done = False
    G = 0
    i = 0
    checkstring=''
    while not done:
        env.cur_weather = holder[i]   # assert weather transition

        action_Tensor,_=agent.get_action(state)
        checkstring+=str(np.argmax(action_Tensor))


        reward, next_state, done, velocity = env.transition(np.argmax(action_Tensor))
        # added velocity for sanity check
        state = deepcopy(next_state)
        G += reward
        i += 1

    print("G: %.2f" % G)
    G_list.append(G)
    #print(checkstring)

print(np.mean(G_list))