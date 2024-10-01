from dataclasses import dataclass
from typing import Callable
import torch
from torch import Tensor, nn

from modules.logging import init_logger

LOGGER = init_logger(__name__)


@dataclass
class MultiLayerPerceptronConfig:
    input_features: int
    hidden_features: int
    hidden_layers: int
    output_features: int
    activation_builder: Callable = None


class MultiLayerPerceptron(nn.Module):
    def __init__(self, config: MultiLayerPerceptronConfig):
        super().__init__()

        activation_builder = config.activation_builder or (lambda: nn.Identity())

        layers = list()
        layers.append(nn.Linear(config.input_features, config.hidden_features))
        layers.append(activation_builder())

        for i in range(config.hidden_layers):
            layers.append(nn.Linear(config.hidden_features, config.hidden_features))
            layers.append(activation_builder())

        layers.append(nn.Linear(config.hidden_features, config.output_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
