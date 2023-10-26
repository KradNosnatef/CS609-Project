from network import FeedForwardNN

class PPO:
    def __init__(self,env):
        self.env=env
        self.obs_dim=env.observation_space.shape[0]
        self.act_dim=env.action_space.shape[0]