from dataclasses import dataclass
from typing import Callable
from torch import Tensor, nn

from modules.logging import init_logger
from modules.nn.siren.activation import Sine

LOGGER = init_logger(__name__)


@dataclass
class SirenConfig:
    input_features: int = None
    hidden_features: int = None
    hidden_layers: int = None
    output_features: int = None
    period: float = 30.0
    a: float = 6.0

class Siren(nn.Module):
    def __init__(self, config: SirenConfig):
        super().__init__()

        layers = list()
        layers.append(nn.Linear(config.input_features, config.hidden_features))
        layers.append(Sine(config.period))

        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(config.hidden_features, config.hidden_features))
            layers.append(Sine(config.period))

        layers.append(nn.Linear(config.hidden_features, config.output_features))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
