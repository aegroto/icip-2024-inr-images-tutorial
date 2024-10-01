import torch
from torch import Tensor, Size

from modules.logging import init_logger
from modules.nn.image_representation.base import ImplicitImageRepresentation
from modules.nn.mlp import MultiLayerPerceptron, MultiLayerPerceptronConfig
from modules.nn.positional_encoder import PositionalEncoder, PositionalEncoderConfig

LOGGER = init_logger(__name__)


class CoordinatesBasedRepresentation(ImplicitImageRepresentation):
    def __init__(
        self,
        encoder_config: PositionalEncoderConfig,
        network_config: MultiLayerPerceptronConfig,
    ):
        super().__init__()

        self.encoder = PositionalEncoder(encoder_config)

        network_config.input_features = self.encoder.output_features_for(network_config.input_features)
        self.network = MultiLayerPerceptron(network_config)

    def generate_input(self, output_shape: Size) -> Tensor:
        (height, width) = (output_shape[0], output_shape[1])

        return torch.cartesian_prod(
            torch.linspace(0.0, 1.0, height),
            torch.linspace(0.0, 1.0, width),
        ).unflatten(0, (height, width))

    def forward(self, coordinates: Tensor) -> Tensor:
        encoded_coordinates = self.encoder(coordinates)
        reconstructed = self.network(encoded_coordinates)
        return reconstructed
