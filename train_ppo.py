import argparse
import os
import copy

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from unnormalized_densities import Toy_dataset
from torch.utils.data import DataLoader
from tqdm import trange
import tqdm
import tempfile
from PIL import Image
from functools import partial
from collections import defaultdict

from net import MLP, UnclippedDiffusion
from reward_fn import GMM


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

# Command-line arguments
parser = argparse.ArgumentParser(description="Train a Gaussian Diffusion Model")
parser.add_argument('--device', type=str, default='cuda:6', help='Device to use for training')


## training
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs for training')
parser.add_argument('--distribution', type=str, default='elliptic_paraboloid', help='Distribution for training diffusion model')
parser.add_argument('--train_batch_size', type=int, default=1000, help='Batch size for training')
parser.add_argument('--num_inner_epochs', type=int, default=1, help='Number of inner epochs for training')
parser.add_argument('--num_batches_per_epoch', type=int, default=1000, help='Number of batches per epoch')
parser.add_argument('--adv_clip_max', type=float, default=5.0, help='Maximum value for clipping advantages')
parser.add_argument('--clip_range', type=float, default=1e-4, help='Clipsping range for rewards')
# parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Number of gradient accumulation steps')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')

parser.add_argument('--reward_fn', type=str, default='gmm', help='Reward model to use for training')

## sampling
parser.add_argument('--sample_batch_size', type=int, default=1, help='Batch size for sampling')
parser.add_argument('--sampling_timesteps', type=int, default=100, help='Number of timesteps for sampling')
parser.add_argument('--kl_divergence_coef', type=float, default=0.0, help='Coefficient for KL divergence regularizer. Set 0 for unconstrained tuning.')


## validation
parser.add_argument('--datanum', type=int, default=50000, help='Number of data points in the dataset')



args = parser.parse_args()


image_dir = f"images/{args.distribution}"
model_dir = f"models/{args.distribution}"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# Device setup
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = args.device

wandb.init(
    entity='gda-for-orl',
    project='toy-explore',
    name=f"{args.distribution}_{args.reward_fn}",
    config=vars(args)
)

model = MLP()

diffusion = UnclippedDiffusion(
    model=model,
    seq_length=2,
    timesteps=100,
    auto_normalize=False,
)

model = model.to(device)
diffusion = diffusion.to(device)


if args.kl_divergence_coef:
    prior_model = copy.deepcopy(model)
    prior_diffusion = copy.deepcopy(diffusion)
    

state_dict = torch.load(f"models/{args.distribution}/save_model.pt", map_location=device)
diffusion.load_state_dict(state_dict)

optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)


rewardfn = GMM(n_components=3, covariance_type='full', random_state=42, rewardfn_path = f"rewardfns/{args.reward_fn}.pkl")

global_step = 0
# Training loop
for epoch in trange(args.num_epochs):


    latents, logprobs, images = diffusion.ddim_sample_with_logprob(
        batch_size=args.num_batches_per_epoch,
        sampling_timesteps=args.sampling_timesteps,
    )

    latents = torch.stack(latents, dim = 1) # (batch_size, num_steps, 1, 2)
    logprobs = torch.stack(logprobs, dim = 1).unsqueeze(-1).detach() # (batch_size, num_steps-1 , 1)
    timesteps, timepairs = diffusion.prepare_ddim_timesteps(args.sampling_timesteps) 
    timesteps = torch.tensor(timesteps[:-1], device=device).repeat(args.num_batches_per_epoch, 1) # (batch_size, num_steps)
    
    ### put rewards
    rewards = rewardfn(images) # (batch_size,)
    

    samples = {
        "latents": latents[:, :-1],
        "next_latents": latents[:, 1:],
        "logprobs": logprobs,
        "rewards": rewards,
        "timesteps": timesteps,
    }

    wandb.log(
        {
            "rewards_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
        },
        step = epoch
    )
        
    # breakpoint()
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    samples["advantages"] = advantages

    # breakpoint()

    for inner_epoch in range(args.num_inner_epochs):

        
        perm = torch.randperm(args.num_batches_per_epoch, device=device)
        samples = {k: v[perm] for k, v in samples.items()}


        perms = torch.stack(
            [
                torch.randperm(args.sampling_timesteps - 1, device=device)
                for _ in range(args.num_batches_per_epoch)
            ]
        )

        for key in ["latents", "next_latents", "logprobs", "timesteps"]:
            samples[key] = samples[key][
                torch.arange(args.num_batches_per_epoch, device=device)[:, None],
                perms,
            ]

        samples_batched = {
            k: v.reshape(-1, args.train_batch_size, *v.shape[1:])
            for k, v in samples.items()
        }

        samples_batched = [
            dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
        ]

        diffusion.model.train()
        optimizer.zero_grad()

        info = defaultdict(list)
        
        loss_accum = 0
        for i, sample in tqdm(
            list(enumerate(samples_batched)), 
            desc=f"Epoch {epoch}| Inner Epoch {inner_epoch + 1}: training",
            position = 0,
        ):
            for j in tqdm(
                range(args.sampling_timesteps - 1),
            ):
                self_cond = None
                clip_denoised = False
                
                noise_preds, x_start = diffusion.model_predictions(
                    sample["latents"][:, j].detach(),
                    sample["timesteps"][:, j],
                    self_cond,
                    clip_x_start=clip_denoised,
                )

                _, logprob = diffusion.ddim_step_with_logprob(
                    noise_preds, x_start, sample["timesteps"][:, j], 
                    latent = sample["latents"][:,j].detach(), next_latent = sample["next_latents"][:,j].detach(), 
                    ddim_sampling_eta = 1.0
                )


                advantages = torch.clamp(
                    sample["advantages"],
                    -args.adv_clip_max,
                    args.adv_clip_max,
                )
                # print("advantages", advantages)
                

                ratio = torch.exp(logprob - sample["logprobs"][:, j].squeeze(-1))
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio, 1 - args.clip_range, 1 + args.clip_range
                )
                loss = torch.mean(torch.max(unclipped_loss, clipped_loss))
                
                if args.kl_divergence_coef:
                    prior_noise_pred, prior_x_start = prior_diffusion.model_predictions(
                        sample["latents"][:, j].detach(),
                        sample["timesteps"][:, j],
                        self_cond,
                        clip_x_start=clip_denoised,
                    ) 
                    
                    kl_divergence = diffusion.get_kl_divergence(
                        noise_preds, 
                        prior_noise_pred,
                        x_start, 
                        prior_x_start,
                        sample["timesteps"][:, j], 
                    )
                    loss = loss - args.kl_divergence_coef * kl_divergence
                loss_accum += loss


                info["ratio"].append(ratio.mean().item())   
                info["loss"].append(loss.item())
                info["approx_kl"].append(
                    0.5
                    * torch.mean((logprob - sample["logprobs"][:, j].squeeze(-1)) ** 2).item()
                )
                info["clipfrac"].append(
                    torch.mean(
                        (
                            torch.abs(ratio - 1.0) > args.clip_range # 1e-4
                        ).float()
                    ).item()
                )

            # loss_accum = loss_accum / args.gradient_accumulation_steps
            # loss_accum.backward()

            # if (i+1) % args.gradient_accumulation_steps == 0:

        loss_accum = loss_accum / args.sampling_timesteps

        loss_accum.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), args.max_grad_norm)
        optimizer.step()
        

        # print(info)
        info = {k: np.mean(v) for k, v in info.items()}
        wandb.log(
            info,
            step = epoch
        )
        info = defaultdict(list)

    
    # Sample and log images after each epoch
    if ((epoch +1) % 10) == 0: 
        diffusion.model.eval()
        latents, logprobs, img = diffusion.ddim_sample_with_logprob(batch_size=int(args.datanum/10), sampling_timesteps=args.sampling_timesteps)
        sampled_images = img.squeeze(1).cpu().detach().numpy()  # Shape: (1000, 2)
        within_bounds = np.all((sampled_images >= -5) & (sampled_images <= 5), axis=1)
        proportion_within_bounds = np.mean(within_bounds)
        plt.figure(figsize=(8, 8))
        plt.scatter(sampled_images[:, 0], sampled_images[:, 1], alpha=0.2, s=1)
        plt.title(f"Sampled Images at Epoch {epoch + 1} | {proportion_within_bounds * 100:.2f}%")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.xlim([-5,5])
        plt.ylim([-5,5])
        plt.grid(True)

        # Save and log plot to wandb
        plot_file = f"images/{args.distribution}/epoch_{epoch + 1}_samples.png"
        plt.savefig(plot_file, dpi=300)
        wandb.log(
            {
                "samples_epoch": wandb.Image(plot_file),
                "Proportion_Within_Bounds": proportion_within_bounds,
            },
            step = epoch
        )
        plt.close()


# Save the trained model
torch.save(diffusion.state_dict(),f'models/{args.distribution}/dppo_save_model.pt')
print(f"Trained model saved to models/{args.distribution}/dppo_save_model.pt")

# Final sampling and plotting
sampled_images = diffusion.sample(batch_size=10000)
print(f"Sampled Images Shape: {sampled_images.shape}")

sampled_images = sampled_images.squeeze(1).cpu().detach().numpy()  # Shape: (1000, 2)
plt.figure(figsize=(8, 8))
plt.scatter(sampled_images[:, 0], sampled_images[:, 1], alpha=0.5, s=10)
plt.title("Sampled Images (2D Scatter Plot)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)

output_image_route = f'images/{args.distribution}/dppo_sampled_images.png'

# Save final plot to file
plt.savefig(output_image_route, dpi=300)
print(f"Sampled image plot saved to {output_image_route}")
wandb.log(
    {"final_samples": wandb.Image(output_image_route)},
)
plt.show()



    



