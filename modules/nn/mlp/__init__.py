from dataclasses import dataclass
from typing import Callable
from torch import Tensor, nn

from modules.logging import init_logger
from modules.nn.linear import QuantizableLinear

LOGGER = init_logger(__name__)


@dataclass
class MultiLayerPerceptronConfig:
    input_features: int = None
    hidden_features: int = None
    hidden_layers: int = None
    output_features: int = None
    activation_builder: Callable = None


class MultiLayerPerceptron(nn.Module):
    def __init__(self, config: MultiLayerPerceptronConfig):
        super().__init__()

        activation_builder = config.activation_builder or (lambda: nn.Identity())

        layers = list()
        layers.append(QuantizableLinear(config.input_features, config.hidden_features))
        layers.append(activation_builder())

        for i in range(config.hidden_layers):
            layers.append(
                QuantizableLinear(config.hidden_features, config.hidden_features)
            )
            layers.append(activation_builder())

        layers.append(QuantizableLinear(config.hidden_features, config.output_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
