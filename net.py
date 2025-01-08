import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Unet1D, GaussianDiffusion1D


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = (torch.log(torch.Tensor([10000.0])) / (half_size - 1)).to(x.device)
        emb = torch.exp(-emb * torch.arange(half_size).to(x.device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class LinearEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class LearnableEmbedding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: torch.Tensor):
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self):
        return self.size


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class ZeroEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1) * 0

    def __len__(self):
        return 1


class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()

        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        elif type == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        elif type == "learnable":
            self.layer = LearnableEmbedding(size)
        elif type == "zero":
            self.layer = ZeroEmbedding()
        elif type == "identity":
            self.layer = IdentityEmbedding()
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 256, hidden_layers: int = 5, emb_size: int = 256,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()
        self.channels = 1
        self.self_condition = False
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t, x_self_cond = None):
        x = x.squeeze()
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x).unsqueeze(1)
        return x
    
    
class UnclippedDiffusion(GaussianDiffusion1D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = False):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = False):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start
    
    def prepare_ddim_timesteps(self, sampling_timesteps):

        times = torch.linspace(-1, self.num_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        return times, time_pairs

    def sample(self, batch_size=16, sampling_timesteps = None):
        self.sampling_timesteps = sampling_timesteps if sampling_timesteps is not None else self.num_timesteps
        assert self.sampling_timesteps <= self.num_timesteps, 'sampling_timesteps must be smaller than num_timesteps'
        
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if self.sampling_timesteps==self.num_timesteps else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length))

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = False, ddim_sampling_eta = 1.0):
        self.ddim_sampling_eta = ddim_sampling_eta
        batch, device, total_timesteps, sampling_timesteps, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.objective

        times, time_pairs = self.prepare_ddim_timesteps(sampling_timesteps)

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)


            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = self.unnormalize(img)
        return img
    

    def ddim_step_with_logprob(self, pred_noise, x_start, timestep, latent, next_latent, ddim_sampling_eta = 1.0):

        
        times, time_pairs = self.prepare_ddim_timesteps(self.sampling_timesteps)

        ts = timestep.tolist()

        batch_timepairs = [time_pairs[self.sampling_timesteps - (i+1)] for i in ts]
        batch_timepairs = torch.tensor(batch_timepairs)
        

        time, time_next = batch_timepairs[:, 0], batch_timepairs[:, 1]
        
        assert time.detach().cpu().sum() == timestep.detach().cpu().sum() 

        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]

        sigma = ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(pred_noise)

        next_sample_mean = alpha_next.sqrt().unsqueeze(1).unsqueeze(2) * x_start + c.unsqueeze(1).unsqueeze(2) * pred_noise

        if next_latent is None:
            next_latent = next_sample_mean + sigma.unsqueeze(1).unsqueeze(2) * noise 
        

        logprob = (
            -((next_latent.detach() - next_sample_mean)**2) / (2 * (sigma**2).unsqueeze(1).unsqueeze(2))
            - torch.log(sigma.unsqueeze(1).unsqueeze(2))
            - torch.log(torch.sqrt(2 * torch.as_tensor(np.pi)))
        )


        logprob = logprob.mean(dim=(1,2))


        return next_latent, logprob



    def ddim_sample_with_logprob(self, batch_size=16, clip_denoised = False, sampling_timesteps = 100, ddim_sampling_eta = 1.0):
        shape = (batch_size, self.channels, self.seq_length)
        eta = ddim_sampling_eta

        device, total_timesteps, objective = self.betas.device, self.num_timesteps, self.objective

        times, time_pairs = self.prepare_ddim_timesteps(sampling_timesteps)

        img = torch.randn(shape, device = device)

        x_start = None
        

        latents = [img]
        logprobs = []

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)
            
            next_sample_mean = alpha_next.sqrt() * x_start + c * pred_noise

            img = next_sample_mean + sigma * noise

            logprob = (
                -((img.detach() - next_sample_mean)**2) / (2 * (sigma**2))
                - torch.log(sigma)
                - torch.log(torch.sqrt(2 * torch.as_tensor(np.pi)))
            )


            logprob = logprob.mean(dim=(1,2))

            latents.append(img)
            logprobs.append(logprob)

            # std_dev_t = sigma
            # pred_sample_direction = c * pred_noise
            # pred_original_sample = x_start
            # alpha_prod_t_prev = alpha_next
            # prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
            # prev_sample = sigma * noise + prev_sample_mean
        img = self.unnormalize(img)    
        
        return latents, logprobs, img
    
    def get_kld(self, pred_noise, prior_pred_noise, x_start, prior_x_start, timestep, ddim_sampling_eta = 1.0):
        times, time_pairs = self.prepare_ddim_timesteps(self.sampling_timesteps)
        ts = timestep.tolist()
        batch_timepairs = [time_pairs[self.sampling_timesteps - (i+1)] for i in ts]
        batch_timepairs = torch.tensor(batch_timepairs)
        time, time_next = batch_timepairs[:, 0], batch_timepairs[:, 1]
        
        assert time.detach().cpu().sum() == timestep.detach().cpu().sum() 

        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]
        sigma = ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(pred_noise)
        next_sample_mean       = alpha_next.sqrt().unsqueeze(1).unsqueeze(2) * x_start       + c.unsqueeze(1).unsqueeze(2) * pred_noise
        prior_next_sample_mean = alpha_next.sqrt().unsqueeze(1).unsqueeze(2) * prior_x_start + c.unsqueeze(1).unsqueeze(2) * prior_pred_noise
        kld = ((next_sample_mean - prior_next_sample_mean.detach()).squeeze() ** 2 / (2 * sigma[time] ** 2).unsqueeze(1)).mean()
        return kld