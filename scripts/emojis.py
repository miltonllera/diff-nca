import io
import requests
import argparse
from enum import Enum
from PIL import Image
from functools import partial, wraps

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ignite.metrics as metrics
from numpy.random import Generator
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from ignite.engine import Engine, Events
from ignite.handlers import ProgressBar
from ema_pytorch import EMA
from typing import Literal

#---------------------------------------------- Dataset ------------------------------------------

def one_hot(values, max):
    b = np.zeros((len(values), max))
    b[np.arange(len(values)), values] = 1.0
    return b


def download_emoji(emoji, max_size):
    code = hex(ord(emoji))[2:].lower()
    url = f"https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u{code}.png?raw=true"
    return load_image(url, max_size)


def load_image(url, max_size=40):
    r = requests.get(url)
    img = Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    img = np.asarray(img, dtype=np.float32) / 255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img


def make_circle_masks(n, h, w, r=None):
    x = np.linspace(-1.0, 1.0, w)[None, None, :]
    y = np.linspace(-1.0, 1.0, h)[None, :, None]
    center = np.random.uniform(-0.5, 0.5, size=[2, n, 1, 1])
    if r is None:
        r = np.random.uniform(0.1, 0.4, size=[n, 1, 1])
    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = x * x + y * y < 1.0
    return mask.astype(float)



class Emojis(Enum):
    BANG = "💥"
    BUTTERFLY = "🦋"
    EYE = "👁"
    FISH = "🐠"
    LADYBUG = "🐞"
    PRETZEL = "🥨"
    SALAMANDER = "🦎"
    SMILEY = "😀"
    TREE = "🎄"
    WEB = "🕸"


class EmojiDataset(IterableDataset):
    def __init__(
        self,
        emoji: str | list[str] = "all",
        target_size: int = 40,
        pad: int = 16,
        batch_size: int = 64,
        rng: Generator | None = None
    ) -> None:
        super().__init__()

        if rng is None:
            rng = np.random.default_rng()

        pad_fn = partial(np.pad, pad_width=((pad, pad), (pad, pad), (0, 0)), mode="constant")
        def init_emojis(emoji):
            emoji = download_emoji(emoji.value, target_size)
            return pad_fn(emoji)

        if isinstance(emoji, str) and emoji == 'all':
            emoji_list = Emojis
        elif isinstance(emoji, str):
            emoji_list = [e for e in Emojis if e.name.lower() == emoji.lower()]
        else:
            emoji = [e.lower for e in emoji]  # type: ignore
            emoji_list = [e for e in Emojis if e.name.lower() in emoji]

        emojis = tuple(map(init_emojis, emoji_list))
        emoji_names = (e.name for e in emoji_list)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 6 * (x - 0.5)),
        ])

        self.emojis = emojis
        self.emoji_names = emoji_names
        self.target_size = target_size
        self.batch_size = batch_size
        self.rng = rng
        self.transform = transform

    def __iter__(self):
        while True:
            yield self.get_emoji()

    def get_emoji(self, i=None):
        if i is None:
            idxs = self.rng.choice(len(self.emojis), (self.batch_size,), replace=True)
        else:
            idxs = np.asarray([i])

        emoji_class = one_hot(idxs, len(Emojis))
        images = [self.emojis[i] for i in idxs]

        return torch.stack([self.transform(i) for i in images]), emoji_class


#----------------------------------------------- Model -------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for time conditioning"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=time.device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DoubleConv(nn.Module):
    """Residual block with two convolutional layers"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x, t):
        t_emb = self.time_mlp(t)[:, :, None, None]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x + t_emb)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.act(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        time_emb_dim: int = 128,
        features: list[int] = [64, 128, 256, 512],
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoding layers
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature, time_emb_dim))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, time_emb_dim)

        # Decoding layers
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature, time_emb_dim))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        skip_connections = []
        for enc_layer in self.encoder:
            x = enc_layer(x, t_emb)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, t_emb)

        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_connection = skip_connections[i // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True
                )

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i + 1](x, t_emb)

        return self.final_conv(x)


class NoiseScheduler:
    """
    Defines a noise schedule for diffusion models.
    Supports linear and cosine schedules.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule_type: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        s: float = 0.008,
        predict_unscaled_noise: bool = True
    ):
        """
        Args:
            timesteps (int): Total number of diffusion steps.
            schedule_type (str): "linear" or "cosine".
            beta_start (float): Initial noise level (only for linear schedule).
            beta_end (float): Final noise level (only for linear schedule).
            s (float): Cosine schedule parameter.
        """
        self.timesteps = timesteps
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_schedule(timesteps, s)
        else:
            raise ValueError("schedule_type must be 'linear' or 'cosine'")

        # Compute alpha values
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # Cumulative product of alphas
        self.predict_unscaled_noise = predict_unscaled_noise

    def _cosine_schedule(self, timesteps, s=0.008):
        """Computes a cosine noise schedule."""
        t = torch.linspace(0, timesteps + 1, timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((t / timesteps + s) / (1 + s)) * (np.pi / 2)) ** 2
        betas = torch.clip(1 - f_t[1:] / f_t[:-1], 0, 0.999)
        return betas

    def sample_time(self, batch_size):
        return torch.randint(1, self.timesteps, size=(batch_size,))

    def corrupt(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        Adds noise to an image according to the schedule.

        Args:
            x_0 (torch.Tensor): Clean image (batch, C, H, W).
            t (torch.Tensor): Diffusion timestep (batch,).

        Returns:
            x_t (torch.Tensor): Noisy image.
            noise (torch.Tensor): The noise added.
        """
        batch_size = x_0.shape[0]
        t = t.to(torch.long)
        n_dim = x_0.ndim

        alpha_bar_t = self.alpha_bars[t].view(batch_size, *([1] * (n_dim -1 ))).to(x_0.device)
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

        if not self.predict_unscaled_noise:
            noise = torch.sqrt(1 - alpha_bar_t) * noise

        return x_t, noise


class DiffusionModel(nn.Module):
    """
    Combines the U-Net and NoiseScheduler into a full diffusion model.
    """
    def __init__(self,
        in_channels: int = 3,
        out_channels: int = 3,
        time_emb_dim: int = 128,
        features: list[int] = [64, 128, 256, 512],
        timesteps: int = 1000,
        schedule_type: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        s: float = 0.008,
        predict_unscaled_noise: bool = True,
    ):
        super().__init__()

        self.unet = UNet(in_channels, out_channels, time_emb_dim, features)
        self.noise_scheduler = NoiseScheduler(
            timesteps, schedule_type, beta_start, beta_end, s, predict_unscaled_noise
        )

    def forward(self, x):
        """
        Runs the diffusion model forward pass.

        Args:
            x_0 (torch.Tensor): Clean image.
            t (torch.Tensor): Diffusion step.

        Returns:
            Predicted noise.
        """
        x_0, y = x
        t = self.noise_scheduler.sample_time(len(x_0))
        x_t, noise = self.noise_scheduler.corrupt(x_0, t)
        predicted_noise = self.unet(x_t, t.to(x_t.device))
        return predicted_noise, noise

    def sample(self, shape):
        """
        Generates images by reversing the diffusion process.

        Args:
            shape (tuple): Shape of the image batch to generate.
            num_steps (int): Number of reverse diffusion steps.

        Returns:
            Generated image tensor.
        """
        x_t = torch.randn(shape).to(next(self.parameters()).device)

        for t in reversed(range(self.noise_scheduler.timesteps)):
            t_tensor = torch.full((shape[0],), t, dtype=torch.float32).to(x_t.device)
            predicted_noise = self.unet(x_t, t_tensor)

            alpha_bar = self.noise_scheduler.alpha_bars[t].view(1, 1, 1, 1).to(x_t.device)
            beta = self.noise_scheduler.betas[t].view(1, 1, 1, 1).to(x_t.device)

            x_t = (
                (1 / torch.sqrt(1 - beta)) *
                (x_t - beta / torch.sqrt(1 - alpha_bar) * predicted_noise)
            )
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t += torch.sqrt(beta) * noise

        return x_t


def get_process_functions(model, criterion, optimizer, scheduler=None, ema=None, device='cpu'):
    model = model.to(device)
    parameters = model.parameters()

    if ema is not None:
        ema.to(device)

    def prepare_batch(fn):
        @wraps(fn)
        def _wrapper(engine, batch):
            x_0, y = batch
            return fn(engine, (x_0.to(device=device), y.to(device=device)))
        return _wrapper

    @prepare_batch
    def train_step(engine, inputs):
        model.train()
        predicted_noise, noise = model(inputs)

        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(parameters, 1.0)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update()

        return loss.item()

    @torch.no_grad()
    @prepare_batch
    def val_step(engine, inputs):
        val_model = model.eval() if ema is None else ema.ema_model.eval()
        return val_model(inputs)

    return train_step, val_step


#---------------------------------------------- Plots --------------------------------------------

def strip(ax):
    ax.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False
    )


def plot_emojis(dataset: EmojiDataset):
    emojis = dataset.emojis
    emojis_names = dataset.emoji_names

    nrows = int(np.ceil(len(emojis) / 5))
    n_cols = 5

    fig, axes = plt.subplots(nrows, n_cols, figsize=(12, 6))
    axes = axes.ravel()

    for em, em_name, ax in zip(emojis, emojis_names, axes):
        ax.imshow(em)
        ax.set_title(em_name)
        strip(ax)

    plt.show()
    return fig


def plot_examples(model, n_samples):
    model.eval()
    samples = model.sample((n_samples, 4, 64, 64)).permute(0, 2, 3, 1).cpu()
    samples = (samples.clip(-3, 3) + 3) / 6

    nrows = int(np.ceil(n_samples / 5))
    n_cols = 5

    fig, axes = plt.subplots(nrows, n_cols, figsize=(12, 6))
    axes = axes.ravel()

    for em, ax in zip(samples, axes):
        ax.imshow(em)
        strip(ax)

    # plt.show()
    return fig


#---------------------------------------------- Main ---------------------------------------------


def main(
    emojis: str,
    noise_steps: int = 1000,
    noise_schedule: Literal['cosine', 'linear'] = 'linear',
    epochs: int = 3000,
    lr: float = 1e-3,
):
    # init dataset
    rng = np.random.default_rng(None)
    dataset = EmojiDataset(emojis, target_size=40, pad=12, batch_size=64, rng=rng)
    train_loader = val_loader = DataLoader(dataset, batch_size=None)

    plot_emojis(dataset)

    # init model
    ddpm = DiffusionModel(
        in_channels=4,
        out_channels=4,
        timesteps=noise_steps,
        schedule_type=noise_schedule
    )
    ema = EMA(ddpm, beta=0.9, update_every=10)

    # training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss = lambda x, y: nn.functional.mse_loss(x, y, reduction='sum') / len(y)
    optimizer = optim.Adam(ddpm.parameters(), lr=lr)

    train_step, val_step = get_process_functions(ddpm, loss, optimizer, None, ema, device)

    trainer = Engine(train_step)
    ProgressBar().attach(trainer, output_transform=lambda x: {'loss': x})

    validator = Engine(val_step)
    metrics.Loss(loss).attach(validator, 'mse')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        metrics = validator.run(val_loader, max_epochs=1, epoch_length=10).metrics
        print(
            f"Training Results - Epoch[{trainer.state.epoch}] - " \
            f"Avg loss: {metrics['mse']:.2f}"
        )

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def example_plots(trainer: Engine):
        with torch.no_grad():
            fig = plot_examples(ddpm, 10)
            fig.savefig(f"plots/examples_iter:{trainer.state.iteration}.jpeg")
            plt.close(fig)

    trainer.run(train_loader, epochs, 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--emojis", type=str, default="all", nargs='+')
    parser.add_argument("--noise_steps", type=int, default=500)
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    main(**vars(args))
