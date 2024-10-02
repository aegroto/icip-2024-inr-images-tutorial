import struct
from torch import Tensor

from modules.logging import init_logger
from modules.nn.quantizer import Quantizer
from modules.packing import IPackable

LOGGER = init_logger(__name__)


class UniformQuantizer(Quantizer, IPackable):
    def __init__(self, bits):
        super().__init__()
        self.bits = bits

        self.zero = 0.0

    def pack(self) -> bytes:
        data = bytes()
        data += struct.pack("!h", self.bits)
        data += struct.pack("!f", self.bound)
        return data

    def calibrate(self, x: Tensor):
        self.bound = x.abs().max().item()

    def _max_symbol(self) -> int:
        return 2 ** (self.bits - 1)

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
