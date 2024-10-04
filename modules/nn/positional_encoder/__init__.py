import torch
from torch import Tensor, nn

from modules.logging import init_logger

LOGGER = init_logger(__name__)


class PositionalEncoder(nn.Module):
    def __init__(self, num_frequencies: int, scale: float = 2.0):
        super().__init__()

        periods = [torch.pi * (scale**i) for i in range(num_frequencies)]

        self.register_buffer("periods", torch.Tensor(periods), persistent=False)

    def output_features_for(self, input_features: int):
        return input_features + input_features * 2 * self.periods.size(-1)

    def forward(self, x: Tensor) -> Tensor:
        angles = self.periods * x.unsqueeze(-1)
        angles = angles.flatten(-2, -1)
        return torch.cat([x, torch.sin(angles), torch.cos(angles)], -1)
