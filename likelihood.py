import argparse
import torch
import wandb
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Unet1D, GaussianDiffusion1D
from unnormalized_densities import Toy_dataset
from torch.utils.data import DataLoader
from tqdm import trange
from probability_flow_ode import ProbabilityFlowODE


from net import MLP, UnclippedDiffusion

# Command-line arguments
parser = argparse.ArgumentParser(description="Train a Gaussian Diffusion Model")
parser.add_argument('--batchsize', type=int, default=1024, help='Batch size for training')
parser.add_argument('--datanum', type=int, default=50000, help='Number of data points in the dataset')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs for training')
parser.add_argument('--output_file', type=str, default='images/sampled_images.png', help='File name for saving sampled image plot')
parser.add_argument('--model_save_path', type=str, default='models/trained_model.pth', help='Path to save the trained model')
parser.add_argument('--distribution', type=str, default='elliptic_paraboloid', help='Distribution for training diffusion model')
args = parser.parse_args()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model setup
model = MLP()

diffusion = ProbabilityFlowODE(
    model=model,
    device=device,
    seq_length=2,
    timesteps=100,
    auto_normalize=False
)

model = model.to(device)
diffusion = diffusion.to(device)

state_dict = torch.load(f"models/{args.distribution}/save_model.pt")
diffusion.load_state_dict(state_dict)




x = torch.linspace(-4, 4, 200, device=device)
y = torch.linspace(-4, 4, 200, device=device)

# Create a 2D grid using torch.meshgrid
xx, yy = torch.meshgrid(x, y, indexing="ij")  # 'ij' ensures matrix indexing (like numpy)

# Combine grid points into a tensor of shape (N, 2)
grid_points = torch.stack([yy.flatten(), xx.flatten()], dim=-1)  # Shape: (200*200, 2)


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
plt.savefig(f"images/{args.distribution}/2d_likelihood_surface.png", dpi=300)
plt.show()

# Plot the 2D heatmap
plt.figure(figsize=(10, 8))
plt.imshow(
    likelihood_values.cpu().detach().numpy(),
    extent=[-5, 5, -5, 5],  # Set axis limits
    origin="lower",         # Ensure the lower-left corner is (-1, -1)
    cmap="viridis",
    aspect="auto"
)
plt.colorbar(label="Log-Likelihood")
plt.title("2D Heatmap of Log-Likelihood")
plt.xlabel("x")
plt.ylabel("y")

# Save the figure
plt.savefig(f"images/{args.distribution}/2d_heatmap_likelihood.png", dpi=300)
plt.show()

samples = diffusion.probability_ode_sample(10000)[0].detach().cpu().numpy()
plt.figure(figsize=(8, 8))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.2, s=1)
plt.title(f"Sampled Images via probability flow ODE")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.grid(True)
plot_file = f"images/{args.distribution}/2d_probODE.png"
plt.savefig(plot_file, dpi=300)