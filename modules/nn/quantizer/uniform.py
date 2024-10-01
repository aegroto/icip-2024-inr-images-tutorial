from torch import Tensor

from modules.logging import init_logger
from modules.nn.quantizer import Quantizer

LOGGER = init_logger(__name__)


class UniformQuantizer(Quantizer):
    def quantize(self, x: Tensor) -> Tensor:
        LOGGER.warning("Call to default quantize implementation")
        return None

    def dequantize(self, x: Tensor) -> Tensor:
        LOGGER.warning("Call to default dequantize implementation")
        return None
