from torch import Tensor

from modules.logging import init_logger
from modules.nn.quantizer import Quantizer

LOGGER = init_logger(__name__)


class UniformQuantizer(Quantizer):
    def __init__(self, bits):
        super().__init__()
        self.bits = bits

        self.zero = 0.0

    def calibrate(self, x: Tensor):
        self.bound = x.abs().max().item()

    def _max_symbol(self) -> int:
        return 2**self.bits

    def quantize(self, x: Tensor) -> Tensor:
        y = x.sub(self.zero)
        y = y.clamp(-self.bound, self.bound)
        y = y.div(self.bound)
        y = y.mul(self._max_symbol())
        y = y.round()

        return y

    def dequantize(self, x: Tensor) -> Tensor:
        y = x.div(self._max_symbol())
        y = y.mul(self.bound)
        y = y.add(self.zero)

        return y
