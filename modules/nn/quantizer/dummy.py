from torch import Tensor

from modules.logging import init_logger
from modules.nn.quantizer import Quantizer

LOGGER = init_logger(__name__)


class DummyQuantizer(Quantizer):
    def quantize(self, x: Tensor) -> Tensor:
        return x

    def dequantize(self, x: Tensor) -> Tensor:
        return x
