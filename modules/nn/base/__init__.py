import torch
from torch import Tensor

from modules.logging import init_logger

LOGGER = init_logger(__name__)

class ImplicitImageRepresentation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def preprocess(unprocessed_image: Tensor) -> Tensor:
        LOGGER.warning("Call to default preprocess implementation")
        return unprocessed_image

    def postprocess(network_output: Tensor) -> Tensor:
        LOGGER.warning("Call to default postprocess implementation")
        return network_output
