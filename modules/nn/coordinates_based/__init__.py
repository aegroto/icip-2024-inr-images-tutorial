import torch
from torch import Tensor, nn

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
            nn.Linear(64, 3),
        )

    def forward(self, coordinates: Tensor) -> Tensor:
        reconstructed = self.network(coordinates)
        return reconstructed
