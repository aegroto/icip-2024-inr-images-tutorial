import torch
from torch import Tensor, nn

from modules.logging import init_logger

LOGGER = init_logger(__name__)


class Sine(nn.Module):
    def __init__(self, period: float):
        super().__init__()

        self.period = period

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x * self.period)
