import argparse

import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchdiffeq import odeint_adjoint as odeint
from tqdm import trange
import wandb
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Unet1D, GaussianDiffusion1D

from unnormalized_densities import Toy_dataset
from net import UnclippedDiffusion


class ODEFunc(nn.Module):
    def __init__(self, diffusion):
        super().__init__()
        self.diffusion = diffusion
        self.nfe = 0
    
    def estimate_drift_and_divergence(self, t, x, epsilon):
        with torch.enable_grad():
            x, log_p = x
            x = x.requires_grad_()
            print(t)
            batched_t = t.expand(x.shape[0]).requires_grad_()
            drift = self.diffusion.forward_drift(x, t) - 0.5 * self.diffusion.forward_diffusion(t) ** 2 * (-self.diffusion.model(x[:, None, :], batched_t) / self.diffusion.sqrt_one_minus_alphas_cumprod[int(t)]).squeeze()
            divergence = torch.sum(
                torch.autograd.grad(
                    torch.sum(drift * epsilon.clone()),
                    x,
                    create_graph=True
                )[0] * epsilon, dim=(1)
            )
        return drift, divergence
    
    def forward(self, t, x):
        self.nfe = self.nfe + 1
        epsilon = torch.randn_like(x[0])
        drift, divergence = self.estimate_drift_and_divergence(t, x, epsilon)
        return drift, divergence


class ProbabilityFlowODE(UnclippedDiffusion):
    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        
    def get_prior_likelihood(
        self,
        z: torch.FloatTensor, 
        sigma: float,
    ):
        shape = torch.tensor(z.shape)
        N = torch.prod(shape[1:])
        return -N / 2. * np.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1)) / (2 * sigma**2)
    
    def forward_drift(self, x, t):
        return -0.5 * self.betas[int(t)] * x
    
    def forward_diffusion(self, t):
        return self.betas[int(t)].sqrt()
            
    
    def probability_ode_sample(self, batch_size, method='euler', atol=1e-2, rtol=1e-2):
        odefunc = ODEFunc(self)
        noise = torch.randn(batch_size, self.seq_length)
        result = odeint(
            odefunc,
            (noise.to(self.device).requires_grad_(), torch.zeros(noise.shape[0]).to(self.device)),
            torch.arange(0, self.num_timesteps).flip(0).to(torch.float32).to(self.device),
            method=method,
            atol=atol, 
            rtol=rtol
        )
        return result[0][-1], odefunc.nfe
    
    def get_likelihood(self, samples, method='euler', atol=1e-2, rtol=1e-2):
        odefunc = ODEFunc(self)
        result = odeint(
            odefunc,
            (samples.to(self.device).requires_grad_(), torch.zeros(samples.shape[0]).to(self.device)),
            torch.arange(0, self.num_timesteps).flip(0).to(torch.float32).to(self.device),
            method=method,
            atol=atol, 
            rtol=rtol
        )
        
        trajectory, delta_ll_traj = result[0], result[1]
        prior, delta_ll= trajectory[-1], delta_ll_traj[-1]
        prior_likelihood = self.get_prior_likelihood(prior, sigma=1.).squeeze()
        log_likelihood = delta_ll + prior_likelihood
        return log_likelihood, prior, odefunc.nfe

