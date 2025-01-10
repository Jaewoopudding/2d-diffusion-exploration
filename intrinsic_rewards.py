import torch

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
        return torch.sqrt(torch.exp((torch.zeros(1).to(self.diffusion.device), predictive_gain) * self.decay_coef / cumulated_count ** -0.5) - 1)
        
        
class NonDifferentiablePseudoCountReward(IntrinsicRewardBase):
    def compute_reward(self, sample, timestep, cumulated_count=None):
        log_likelihood, _, _ = self.diffusion.get_likelihood(sample, timestep)
        return 1 / (torch.exp(log_likelihood)) ** 0.5
    
    
class StateEntropyReward(IntrinsicRewardBase):
    def compute_reward(self, sample, timestep, cumulated_count):
        log_likelihood, _, _ = self.diffusion.get_likelihood(sample, timestep)
        return -log_likelihood
    

intrinsic_reward_mapping = {
    "none": IntrinsicRewardBase,
    "differentiable_pseudo_count": DifferentiablePseudoCountReward,
    "differentiable_last_pseudo_count": DifferentiablePseudoCountReward,
    "non_differentiable_pseudo_count": NonDifferentiablePseudoCountReward,
    "state_entropy": StateEntropyReward,
}