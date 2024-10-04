from torch import Tensor, Size, nn

from modules.helpers.coordinates import generate_coordinates_grid
from modules.logging import init_logger
from modules.nn.image_representation.base import ImplicitImageRepresentation

LOGGER = init_logger(__name__)


class CoordinatesBasedRepresentation(ImplicitImageRepresentation):
    def __init__(self, encoder: nn.Module, network: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.network = network

    def generate_input(self, output_shape: Size) -> Tensor:
        (height, width) = (output_shape[0], output_shape[1])
        return generate_coordinates_grid(height, width)

    def forward(self, coordinates: Tensor) -> Tensor:
        encoded_coordinates = self.encoder(coordinates)
        reconstructed = self.network(encoded_coordinates)
        return reconstructed
