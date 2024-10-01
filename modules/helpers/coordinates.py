import torch
from torch import Tensor


def generate_coordinates_grid(height: int, width: int) -> Tensor:
    return torch.cartesian_prod(
        torch.linspace(0.0, 1.0, height),
        torch.linspace(0.0, 1.0, width),
    ).unflatten(0, (height, width))
