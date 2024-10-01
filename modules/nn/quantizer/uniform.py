from torch import Tensor

from modules.logging import init_logger
from modules.nn.quantizer import Quantizer

LOGGER = init_logger(__name__)


class UniformQuantizer(Quantizer):
    def __init__(self, bits):
        super().__init__()
        self.__bits = bits

    def quantize(self, x: Tensor) -> Tensor:
        LOGGER.warning("Uniform quantization not implemented yet")
        return x

    def dequantize(self, x: Tensor) -> Tensor:
        LOGGER.warning("Uniform dequantization not implemented yet")
        return x
