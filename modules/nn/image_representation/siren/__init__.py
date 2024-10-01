from torch import Tensor, Size

from modules.helpers.coordinates import generate_coordinates_grid
from modules.logging import init_logger
from modules.nn.image_representation.base import ImplicitImageRepresentation
from modules.nn.positional_encoder import PositionalEncoder, PositionalEncoderConfig
from modules.nn.siren import Siren, SirenConfig

LOGGER = init_logger(__name__)


class SirenRepresentation(ImplicitImageRepresentation):
    def __init__(
        self,
        encoder_config: PositionalEncoderConfig,
        network_config: SirenConfig,
    ):
        super().__init__()

        self.encoder = PositionalEncoder(encoder_config)

        network_config.input_features = self.encoder.output_features_for(
            network_config.input_features
        )
        self.network = Siren(network_config)

    def generate_input(self, output_shape: Size) -> Tensor:
        (height, width) = (output_shape[0], output_shape[1])
        return generate_coordinates_grid(height, width)

    def forward(self, coordinates: Tensor) -> Tensor:
        encoded_coordinates = self.encoder(coordinates)
        reconstructed = self.network(encoded_coordinates)
        return reconstructed
