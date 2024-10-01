import torch
from torch import Tensor, Size

from modules.logging import init_logger

LOGGER = init_logger(__name__)


class ImplicitImageRepresentation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def generate_input(self, output_shape: Size) -> Tensor:
        LOGGER.warning("Call to default generate_input_for implementation")
        return None

    def preprocess(self, unprocessed_image: Tensor) -> Tensor:
        LOGGER.warning("Call to default preprocess implementation")
        return unprocessed_image

    def postprocess(self, network_output: Tensor) -> Tensor:
        LOGGER.warning("Call to default postprocess implementation")
        return network_output
