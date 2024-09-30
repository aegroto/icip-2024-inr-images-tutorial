import torch
from torch import Tensor, Size, nn

from modules.logging import init_logger
from modules.nn.base import ImplicitImageRepresentation
from modules.training.batch import TrainingBatch

LOGGER = init_logger(__name__)


class CoordinatesBasedRepresentation(ImplicitImageRepresentation):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def generate_input(self, output_shape: Size) -> Tensor:
        (height, width) = (output_shape[0], output_shape[1])

        return torch.cartesian_prod(
            torch.linspace(0.0, 1.0, height),
            torch.linspace(0.0, 1.0, width),
        ).unflatten(0, (height, width))

    def forward(self, coordinates: Tensor) -> Tensor:
        reconstructed = self.network(coordinates)
        return reconstructed
