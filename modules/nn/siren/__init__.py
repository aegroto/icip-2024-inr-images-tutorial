from dataclasses import dataclass
from torch import Tensor, nn

from modules.logging import init_logger
from modules.nn.quantizable_linear import QuantizableLinear
from modules.nn.siren.activation import Sine
from modules.nn.siren.initialization import (
    initialize_first_siren_layer,
    initialize_siren_layer,
)

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

        first_layer = QuantizableLinear(config.input_features, config.hidden_features)
        initialize_first_siren_layer(first_layer)
        layers.append(first_layer)
        layers.append(Sine(config.period))

        for _ in range(config.hidden_layers):
            hidden_layer = QuantizableLinear(
                config.hidden_features, config.hidden_features
            )
            initialize_siren_layer(hidden_layer, config.period, config.a)
            layers.append(hidden_layer)
            layers.append(Sine(config.period))

        last_layer = QuantizableLinear(config.hidden_features, config.output_features)
        initialize_siren_layer(last_layer, config.period, config.a)
        layers.append(last_layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
