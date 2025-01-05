import argparse
import os

import torch
import wandb
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Unet1D, GaussianDiffusion1D
from unnormalized_densities import Toy_dataset
from torch.utils.data import DataLoader
from tqdm import trange

# Command-line arguments
parser = argparse.ArgumentParser(description="Train a Gaussian Diffusion Model")
parser.add_argument('--batchsize', type=int, default=1024, help='Batch size for training')
parser.add_argument('--datanum', type=int, default=200000, help='Number of data points in the dataset')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs for training')
parser.add_argument('--distribution', type=str, default='8gaussians', help='Distribution for training diffusion model')
args = parser.parse_args()


image_dir = f"images/{args.distribution}"
model_dir = f"models/{args.distribution}"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(
    entity='gda-for-orl',
    project='toy-explore',
)

# Model setup
unet = Unet1D(
    dim=64,
    init_dim=2,
    channels=1,
    dim_mults=(1, 2),
)

diffusion = GaussianDiffusion1D(
    unet,
    seq_length=2,
    timesteps=100,
    auto_normalize=False
)

unet = unet.to(device)
diffusion = diffusion.to(device)
# Dataset and DataLoader
custom = Toy_dataset(args.distribution, minimum=0, datanum=args.datanum, device=device)
custom.visualize_and_save(args.datanum)
dataloader = DataLoader(custom, batch_size=args.batchsize, shuffle=True)

# Optimizer
optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

# Training loop
for epoch in trange(args.num_epochs):
    epoch_loss = 0
    batch_count = 0
    for batch in dataloader:
        training_data = batch[:, None, :]  # Extract the training data
        optimizer.zero_grad()
        loss = diffusion(training_data)  # Compute the loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update the model parameters

        # Accumulate loss
        epoch_loss += loss.item()
        batch_count += 1

    # Average loss for the epoch
    avg_loss = epoch_loss / batch_count
    wandb.log({"loss": avg_loss}, step=epoch)

    # Sample and log images after each epoch
    if (epoch % 10) == 0: 
        sampled_images = diffusion.sample(batch_size=int(args.datanum/10)).squeeze(1).cpu().detach().numpy()  # Shape: (1000, 2)
        plt.figure(figsize=(8, 8))
        plt.scatter(sampled_images[:, 0], sampled_images[:, 1], alpha=0.2, s=1)
        plt.title(f"Sampled Images at Epoch {epoch + 1}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)

        # Save and log plot to wandb
        plot_file = f"images/{args.distribution}/epoch_{epoch + 1}_samples.png"
        plt.savefig(plot_file, dpi=300)
        wandb.log(
            {"samples_epoch": wandb.Image(plot_file)},
            step=epoch
        )
        plt.close()

# Save the trained model
torch.save(diffusion.state_dict(),f'models/{args.distribution}/save_model.pt')
print(f"Trained model saved to models/{args.distribution}/save_model.pt")

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

output_image_route = f'images/{args.distribution}/sampled_images.png'

# Save final plot to file
plt.savefig(output_image_route, dpi=300)
print(f"Sampled image plot saved to {output_image_route}")
wandb.log(
    {"final_samples": wandb.Image(output_image_route)}
)
plt.show()