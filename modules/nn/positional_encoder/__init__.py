from dataclasses import dataclass
from typing import Callable
import torch
from torch import Tensor, nn

from modules.logging import init_logger

LOGGER = init_logger(__name__)


@dataclass
class PositionalEncoderConfig:
    num_frequencies: int
    scale: float = 2.0


class PositionalEncoder(nn.Module):
    def __init__(self, config: PositionalEncoderConfig):
        super().__init__()

        periods = [torch.pi * (config.scale**i) for i in range(config.num_frequencies)]

        self.register_buffer("periods", torch.Tensor(periods), persistent=False)

    def output_features_for(self, input_features: int):
        return input_features + input_features * 2 * self.periods.size(-1)

    def forward(self, x: Tensor) -> Tensor:
        angles = self.periods * x.unsqueeze(-1)
        angles = angles.flatten(-2, -1)
        return torch.cat([x, torch.sin(angles), torch.cos(angles)], -1)
