import torch
from torch import Tensor, Size

from modules.logging import init_logger

LOGGER = init_logger(__name__)


class ImplicitImageRepresentation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def generate_input(self, output_shape: Size) -> Tensor:
        raise NotImplementedError
