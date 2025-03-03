from typing import Callable
import torch
import torch.nn as nn


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
                module.reset_params()

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
            update = self.update_fn(state, percpetion, condition)
            state = state + update * not_done

            if return_trajectory:
                trajectory.append(state)  # type: ignore

            # update iteration
            not_done = i <= batch_steps

        if return_trajectory:
            trajectory = torch.stack(trajectory, dim=0)  # type: ignore

        return state, trajectory


def sample_steps(batch_size, min_steps, max_steps):
    if min_steps is None:
        return max_steps
    return torch.randint(min_steps, max_steps, (batch_size,))


def _broadcast(mask, tensor):
    mask_shape = [len(tensor)] + [1] * (tensor.ndim - 1)
    return mask.view(*mask_shape).to(device=tensor.device)
