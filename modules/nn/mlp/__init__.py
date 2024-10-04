from typing import Callable
from torch import Tensor, nn

from modules.logging import init_logger
from modules.nn.quantizable_linear import QuantizableLinear

LOGGER = init_logger(__name__)


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        hidden_layers: int,
        output_features: int,
        activation_builder: Callable = None,
    ):
        super().__init__()

        activation_builder = activation_builder or (lambda: nn.Identity())

        layers = list()
        layers.append(QuantizableLinear(input_features, hidden_features))
        layers.append(activation_builder())

        for _ in range(hidden_layers):
            layers.append(QuantizableLinear(hidden_features, hidden_features))
            layers.append(activation_builder())

        layers.append(QuantizableLinear(hidden_features, output_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
