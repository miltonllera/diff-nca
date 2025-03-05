import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Callable


class CellularAutomata(nn.Module):
    def __init__(
        self,
        perception_fn: Callable,
        update_fn: Callable,
        input_net: Callable = nn.Identity(),
    ) -> None:
        super().__init__()

        self.perception_fn = perception_fn
        self.update_fn = update_fn
        self.input_net = input_net

    def reset_params(self):
        for module in self.children():
            if hasattr(module, 'reset_params'):
                module.reset_params()  # type: ignore

    def forward(
        self,
        state: torch.Tensor,
        n_steps: int | tuple[int, int],
        inputs: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ):
        B = len(state)

        if isinstance(n_steps, tuple):
            min_steps, max_steps = n_steps
        else:
            min_steps, max_steps = None, n_steps

        condition = self.input_net(inputs) if inputs is not None else None
        batch_steps = _broadcast(sample_steps(B, min_steps, max_steps), state)

        i, not_done = 0, 0 < batch_steps
        trajectory = [state] if return_trajectory else None

        while torch.any(not_done) and (i := i + 1) < max_steps:
            percpetion = self.perception_fn(state)
            updated = self.update_fn(state, percpetion, condition)
            state = torch.where(not_done, updated, state)

            if return_trajectory:
                trajectory.append(state)  # type: ignore

            # update iteration
            not_done = i <= batch_steps

        if return_trajectory:
            trajectory = torch.stack(trajectory, dim=0)  # type: ignore

        return state, trajectory


def sample_steps(batch_size, min_steps, max_steps):
    if min_steps is None:
        return torch.tensor([max_steps]).expand(batch_size)
    return torch.randint(min_steps, max_steps, (batch_size,))


def _broadcast(mask, tensor):
    mask_shape = [len(tensor)] + [1] * (tensor.ndim - 1)
    return mask.view(*mask_shape).to(device=tensor.device)


class ResidualUpdate(nn.Sequential):
    def forward(self, state, perception, condition=None):
        update = super().forward(perception)
        return state + update


class CustomPadding(nn.Module):
    def __init__(self, padding, dims, value):
        super().__init__()
        self.value = value
        self.padding = padding
        self.dims = dims

    def forward(self, inputs):
        pad = [self.padding] * (self.dims * 2)
        return torch.nn.functional.pad(inputs, pad, mode='constant', value=self.value)


class NCA(nn.Module):
    def __init__(
        self,
        grid_size: tuple[int, int],
        vis_channels: int,
        hidden_channels: int,
        num_steps: int,
    ):
        super().__init__()
        channels = vis_channels + hidden_channels
        perception_fn = nn.Sequential(
            # CustomPadding(padding=1, dims=2, value=-1),
            # nn.Conv2d(channels, channels, kernel_size=3, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=3, groups=channels, padding=1),
            nn.ReLU(),
        )
        update_fn = ResidualUpdate(
            nn.Conv2d(channels, 2 * channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            # nn.Dropout(0.2),
        )

        self.vis_channels = vis_channels
        self.hidden_channels = hidden_channels
        self.grid_size = grid_size
        self.ca = CellularAutomata(perception_fn, update_fn)
        self.num_steps = num_steps

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def total_channels(self):
        return self.vis_channels + self.hidden_channels

    def forward(self, x):
        B = len(x)

        state = torch.zeros((B, self.total_channels, *self.grid_size))
        state[:, :self.vis_channels] = 0.5 + 0.1 * torch.randn(
            (B, self.vis_channels, *self.grid_size)
        ).clamp(0, 1)
        # state[:, :self.vis_channels] = torch.rand(
        #     (B, self.vis_channels, *self.grid_size)
        # )

        state = self.ca(state, self.num_steps)[0]

        return torch.sigmoid(state[:, :self.vis_channels])


def main():
    size = 32
    nca = NCA((size, size), 1, 4, size * 2).train()

    target = torch.zeros((1, 1, size, size))
    target[:, :, size//2 - 4:size//2 + 4, size // 2 - 4: size // 2 + 4] = 1.0

    optimizer = optim.Adam(nca.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='sum')

    for _  in range(10000):
        optimizer.zero_grad()

        output = nca(target)
        loss = criterion(output, target) / len(target)
        loss.backward()

        optimizer.step()

        print(f"Loss: {loss.item()}")

    with torch.no_grad():
        nca.eval()
        final_example = nca(target)

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    ax1.imshow(target[0].permute(1, 2, 0).numpy(), cmap='gray')
    ax2.imshow(final_example[0].permute(1, 2, 0).numpy(), cmap='gray')

    plt.show()


if __name__ == "__main__":
    main()

