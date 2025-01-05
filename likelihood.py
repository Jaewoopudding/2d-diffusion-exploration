import argparse
import torch
import wandb
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Unet1D, GaussianDiffusion1D
from unnormalized_densities import Toy_dataset
from torch.utils.data import DataLoader
from tqdm import trange
from probability_flow_ode import ProbabilityFlowODE

# Command-line arguments
parser = argparse.ArgumentParser(description="Train a Gaussian Diffusion Model")
parser.add_argument('--batchsize', type=int, default=1024, help='Batch size for training')
parser.add_argument('--datanum', type=int, default=50000, help='Number of data points in the dataset')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs for training')
parser.add_argument('--output_file', type=str, default='images/sampled_images.png', help='File name for saving sampled image plot')
parser.add_argument('--model_save_path', type=str, default='models/trained_model.pth', help='Path to save the trained model')
args = parser.parse_args()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model setup
unet = Unet1D(
    dim=32,
    init_dim=2,
    channels=1,
    dim_mults=(1, 2),
)

diffusion = ProbabilityFlowODE(
    model=unet,
    device=device,
    seq_length=2,
    timesteps=100,
    auto_normalize=False
)

unet = unet.to(device)
diffusion = diffusion.to(device)

state_dict = torch.load("/home/jaewoo/research/2d-exploration-diffusion/custom_model.pth")
diffusion.load_state_dict(state_dict)




x = torch.linspace(-1, 1, 200, device=device)
y = torch.linspace(-1, 1, 200, device=device)

# Create a 2D grid using torch.meshgrid
xx, yy = torch.meshgrid(x, y, indexing="ij")  # 'ij' ensures matrix indexing (like numpy)

# Combine grid points into a tensor of shape (N, 2)
grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # Shape: (200*200, 2)


log_likelihood, prior, nfe = diffusion.get_likelihood(grid_points)

likelihood_values = log_likelihood.view(200, 200).cpu()

# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
    xx.cpu().detach().numpy(),
    yy.cpu().detach().numpy(),
    likelihood_values.cpu().detach().numpy(),
    cmap='viridis',
    edgecolor='none',
    alpha=0.8
)

# Customize the plot
ax.set_title("Log-Likelihood Surface")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Log-Likelihood')

# Save the figure
plt.savefig("likelihood_surface.png", dpi=300)
plt.show()

# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
    xx.cpu().numpy(),
    yy.cpu().numpy(),
    likelihood_values.cpu().detach().numpy(),
    cmap='viridis',
    edgecolor='none',
    alpha=0.8
)

# Customize the plot
ax.set_title("Log-Likelihood Surface")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Log-Likelihood')

# Save the figure
plt.savefig("likelihood_surface_3d.png", dpi=300)
plt.show()

# Plot the 2D heatmap
plt.figure(figsize=(10, 8))
plt.imshow(
    likelihood_values.cpu().detach().numpy(),
    extent=[-1, 1, -1, 1],  # Set axis limits
    origin="lower",         # Ensure the lower-left corner is (-1, -1)
    cmap="viridis",
    aspect="auto"
)
plt.colorbar(label="Log-Likelihood")
plt.title("2D Heatmap of Log-Likelihood")
plt.xlabel("x")
plt.ylabel("y")

# Save the figure
plt.savefig("2d_heatmap_likelihood.png", dpi=300)
plt.show()