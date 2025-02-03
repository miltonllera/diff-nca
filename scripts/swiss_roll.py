import argparse
import math
import matplotlib.pyplot as plt
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


#----------------------------------------------- Dataset -----------------------------------------

class SpiralDiffusionDataset(Dataset):
    def __init__(
        self,
        num_samples: int =1000,
        num_turns: float = 5,
        a: float = 0.1,
        noise_scale: float = 0.0,
        timesteps: int = 100,
        noise_schedule: Literal["linear" , "cosine"] = "linear",
        predict_added_noise: bool = False,
    ):
        """
        PyTorch Dataset for generating 2D spirals with diffusion noise.

        Parameters:
        - num_samples (int): Number of points in the spiral.
        - num_turns (int): Number of turns in the spiral.
        - a (float): Scaling factor for the spiral.
        - timesteps (int): Number of diffusion timesteps.
        - noise_schedule (str): Type of noise schedule ("linear" or "cosine").
        """
        self.num_samples = num_samples
        self.num_turns = num_turns
        self.a = a
        self.timesteps = timesteps
        self.predict_added_noise = predict_added_noise

        # Generate the clean spiral dataset
        theta = np.linspace(0, num_turns * 2 * np.pi, num_samples)
        r = a * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        self.clean_data = np.stack([x, y], axis=1).astype(np.float32)

        # Convert to torch tensor
        self.clean_tensor = torch.tensor(self.clean_data, dtype=torch.float32)

        if noise_scale > 0:
            self.clean_tensor = self.clean_tensor + \
                noise_scale * torch.rand_like(self.clean_tensor)

        # Define noise schedule
        if noise_schedule == "linear":
            self.betas = torch.linspace(1e-4, 0.02, timesteps)
        elif noise_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError("Invalid noise schedule. Choose 'linear' or 'cosine'.")

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _cosine_beta_schedule(self, timesteps):
        """
        Implements a cosine noise schedule similar to Improved DDPMs.
        """
        s = 0.008
        f = lambda t: np.cos((t / timesteps + s) / (1 + s) * (np.pi / 2)) ** 2

        a_bars = f(np.linspace(0, timesteps+1, timesteps+1, endpoint=False))
        betas = np.clip(1 - a_bars[1:] / a_bars[:-1], 0.0001, 0.9999)
        return torch.tensor(betas, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a single noisy sample along with its timestep and noise.
        """
        x_0 = self.clean_tensor[idx]  # Clean spiral point

        # Sample a random timestep
        t = torch.randint(self.timesteps, (1,)).item()
        alpha_bar_t = self.alpha_bars[t]

        # Generate Gaussian noise
        noise = torch.randn_like(x_0)

        # Apply forward diffusion step
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        target = noise

        if self.predict_added_noise:
            target = torch.sqrt(1 - alpha_bar_t) * target

        return x_t, t, target  # Noisy sample, timestep, ground truth noise


#------------------------------------------------ Model ------------------------------------------

class MLPDenoiser(nn.Module):
    def __init__(self, timesteps, hidden_dim=128):
        super().__init__()
        self.timesteps = timesteps
        self.model = nn.Sequential(
            nn.Linear(2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x_t, t):
        """
        Forward pass of the denoiser.

        Inputs:
        - x_t: Noisy input of shape (batch_size, 2)
        - t: Timestep (batch_size, 1)

        Output:
        - Predicted noise ε (same shape as x_t)
        """
        t_embed = t.float() / self.timesteps  # Normalize timestep
        t_embed = t_embed.unsqueeze(1)  # Ensure shape (batch_size, 1)
        x_input = torch.cat([x_t, t_embed], dim=1)  # Concatenate (x, y, t)
        return self.model(x_input)


#---------------------------------------------- Training -----------------------------------------


def train_denoiser(model, dataloader, epochs=100, lr=1e-3, device="cpu"):
    model.to(device)

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5 ** (1/200_000))
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for x_t, t, noise in dataloader:
            x_t, t, noise = x_t.to(device), t.to(device), noise.to(device)

            # Predict the noise
            noise_pred = model(x_t, t)

            # Compute loss
            loss = loss_fn(noise_pred, noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(parameters, 1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    print("Training complete!")


def sample_spiral(model, dataset, device="cpu"):
    model.eval()  # Set model to evaluation mode
    model.to(device)

    num_samples = len(dataset)
    timesteps = dataset.timesteps
    alpha_bars = torch.concatenate([dataset.alpha_bars, torch.tensor([1.0])])

    # Define noise schedule (same as training)

    # Start from pure Gaussian noise
    x_t = torch.randn((num_samples, 2), device=device)

    for tm1 in reversed(range(timesteps)):
        t_tensor = torch.full((num_samples,), tm1, device=device, dtype=torch.float32)

        # Predict the noise using the trained model
        epsilon_pred = model(x_t, t_tensor)

        if dataset.predict_added_noise:
            x_t = x_t + epsilon_pred
        else:
            alpha_bar_tm1, alpha_bar_t = alpha_bars[tm1: tm1 + 2]
            x_t = torch.sqrt(alpha_bar_tm1) * (x_t - torch.sqrt(1 - alpha_bar_t) * \
                epsilon_pred) / torch.sqrt(alpha_bar_t)

        # Add small noise (except at last step t=0)
        if tm1 > 0:
            noise = epsilon_pred * torch.sqrt(1.0 - alpha_bar_tm1)
            x_t += noise

    return x_t.cpu().detach().numpy()  # Convert back to NumPy for plotting


#-------------------------------------------- Plotting -------------------------------------------

def plot_schedules(dataset):
    t = np.arange(dataset.timesteps)
    betas = dataset.betas
    alpha_bars =  dataset.alpha_bars

    _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,6))
    ax1.plot(t, betas)
    ax1.set_title('beta')
    ax2.plot(t, alpha_bars)
    ax2.set_title('alpha bar')


def plot_diffusion(dataset: SpiralDiffusionDataset, steps: int = 100):
    x, y = dataset.clean_tensor.T.numpy()
    num_points = len(x)

    total_noise = dataset.alpha_bars[steps // 2]
    mean_scale = math.sqrt(total_noise)
    var_scale = math.sqrt(1 - total_noise)

    # Add normally distributed noise
    x_noisy = mean_scale * x + np.random.normal(0, var_scale, num_points)
    y_noisy = mean_scale * y + np.random.normal(0, var_scale, num_points)

    # Plot the noisy spiral
    plt.figure(figsize=(6, 6))
    plt.scatter(x_noisy, y_noisy, c=np.linspace(0, 1, num_points), cmap='viridis', s=5, alpha=0.7)
    plt.scatter(x, y, c='red', s=5, alpha=0.7)
    plt.axis("equal")  # Keep aspect ratio square
    plt.title("2D Spiral Dataset")


def plot_spiral(x_T, y_T, x_0, y_0):
    num_points = len(x_T)
    # Plot the noisy spiral
    plt.figure(figsize=(6, 6))
    plt.scatter(x_0, y_0, c=np.linspace(0, 1, num_points), cmap='viridis', s=5, alpha=0.7)
    plt.title("2D Spiral Dataset")



#------------------------------------------------ Main -------------------------------------------


def main(
    num_samples: int = 3000,
    num_turns: float = 1.5,
    radius_scale: float = 0.2,
    timesteps: int = 500,
    beta_schedule: Literal['cosine', 'linear'] = 'linear',
    hidden_dim: int = 128,
    epochs: int = 3000,
    lr: float = 1e-3,
    plot_data: bool = True
):
    dataset = SpiralDiffusionDataset(
        num_samples,
        num_turns,
        radius_scale,
        0.3,
        timesteps,
        beta_schedule
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    if plot_data:
        plot_diffusion(dataset, timesteps)
        plt.show()
        plot_schedules(dataset)
        plt.show()

    model = MLPDenoiser(timesteps, hidden_dim)
    train_denoiser(model, dataloader, epochs=epochs, lr=lr, device='cpu')

    sample = sample_spiral(model, dataset)
    x_t = torch.randn((len(dataset), 2), device='cpu')

    if plot_data:
        plot_spiral(*(x_t.T), *(sample.T))
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--num_turns", type=float, default=1.3)
    parser.add_argument("--radius_scale", type=float, default=0.2)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--plot_data", type=bool, default=True)

    args = parser.parse_args()

    main(**vars(args))
