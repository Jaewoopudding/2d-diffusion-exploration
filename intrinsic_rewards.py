import torch
import torch.nn as nn
from net import MLP4RND

class IntrinsicRewardBase:
    def __init__(self, diffusion):
        self.diffusion=diffusion
    
    def compute_reward(self, sample, timestep=None, cumulated_count=None):
        return 0

    def __call__(self, sample, timestep=None, cumulated_count=None):
        return self.compute_reward(sample, timestep, cumulated_count)
    

class DifferentiablePseudoCountReward(IntrinsicRewardBase):
    def __init__(self, diffusion, decay_coef=0.1):
        super().__init__(diffusion)
        self.decay_coef = decay_coef
    
    def compute_reward(self, sample, timestep, cumulated_count):
        log_likelihood, _, _ = self.diffusion.get_likelihood(sample, timestep)
        assert log_likelihood.requires_grad
        predictive_gain = log_likelihood - log_likelihood.detach()
        return torch.sqrt(torch.exp(predictive_gain * self.decay_coef / cumulated_count ** -0.5) - 1)
        
        
class NonDifferentiablePseudoCountReward(IntrinsicRewardBase):
    def compute_reward(self, sample, timestep, cumulated_count=None):
        log_likelihood, _, _ = self.diffusion.get_likelihood(sample, timestep)
        return 1 / ((torch.exp(log_likelihood)) ** 0.5 + 0.01)
    
    
class StateEntropyReward(IntrinsicRewardBase):
    def compute_reward(self, sample, timestep, cumulated_count):
        log_likelihood, _, _ = self.diffusion.get_likelihood(sample, timestep)
        return -log_likelihood
    

class RND(IntrinsicRewardBase):
    def __init__(self, device):
        super().__init__(diffusion=None)

        self.random_target_network = MLP4RND().to(device)
        self.predictor_network = MLP4RND().to(device)

        # self.device = next(self.predictor_network.parameters()).device
        # self.random_target_network = self.random_target_network.to(self.device)

    
    def forward(self, next_obs):

        next_obs_copy = next_obs.clone()

        random_obs = self.random_target_network(next_obs)
        predicted_obs = self.predictor_network(next_obs_copy)

        return random_obs, predicted_obs

    def compute_reward(self, next_obs):
        random_obs, predicted_obs = self.forward(next_obs)

        intrinsic_reward = torch.norm(predicted_obs.detach() - random_obs.detach(), dim=-1, p=2)

        return intrinsic_reward

    def compute_loss(self, next_obs):
        random_obs, predicted_obs = self.forward(next_obs)
        rnd_loss = torch.norm(predicted_obs - random_obs.detach(), dim=-1, p=2)
        mean_rnd_loss = torch.mean(rnd_loss)
        return mean_rnd_loss



intrinsic_reward_mapping = {
    "none": IntrinsicRewardBase,
    "differentiable_pseudo_count": DifferentiablePseudoCountReward,
    "differentiable_last_pseudo_count": DifferentiablePseudoCountReward,
    "non_differentiable_pseudo_count": NonDifferentiablePseudoCountReward,
    "state_entropy": StateEntropyReward,
    "rnd" : RND
}