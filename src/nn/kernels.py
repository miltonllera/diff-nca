import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelFilter(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        kernel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]) / 8
        kernel_y = kernel_x.T.clone()
        kernel_x = torch.tile(kernel_x, (n_channels, 1, 1, 1))
        kernel_y = torch.tile(kernel_y, (n_channels, 1, 1, 1))

        self.n_channels = n_channels
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, inputs: torch.Tensor):
        # use groups so that each input is convolved with it's own kernel
        dx = F.conv2d(inputs, self.kernel_x, padding=1, groups=self.n_channels)
        dy = F.conv2d(inputs, self.kernel_y, padding=1, groups=self.n_channels)
        return torch.cat([dx, dy], dim=1)


class DiffusionKernel(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        kernel = torch.tensor([
            [0.05, 0.20, 0.05],
            [0.20, -1.0, 0.20],
            [0.05, 0.20, 0.05],
        ])
        kernel = torch.tile(kernel, (n_channels, 1, 1, 1))

        self.n_channels = n_channels
        self.rates = nn.Parameter(torch.zeros(n_channels, 1, 1))
        self.register_buffer("kernel", kernel)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.rates, -0.1, 1.0)

    def forward(self, inputs: torch.Tensor):
        # use groups so that each input is convolved with it's own kernel
        diff = F.conv2d(inputs, self.kernel, padding=1, groups=self.n_channels)
        return self.rates * diff
