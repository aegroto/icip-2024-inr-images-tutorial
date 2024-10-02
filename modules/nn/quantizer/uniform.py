import struct
import torch
from torch import Tensor

from modules.logging import init_logger
from modules.nn.quantizer import Quantizer
from modules.packing import IPackable

LOGGER = init_logger(__name__)


class UniformQuantizer(Quantizer, IPackable):
    def __init__(self, bits):
        super().__init__()

        self.register_buffer("bits", Tensor([bits]).to(torch.int32), persistent=True)
        self.register_buffer("zero", Tensor([0.0]), persistent=True)
        self.register_buffer("bound", Tensor([1.0]), persistent=True)

    def pack(self) -> bytes:
        LOGGER.debug(f"Packing quantization values::  bits: {self.bits} bound: {self.bound}")

        data = bytes()
        data += struct.pack("!h", self.bits.item())
        data += struct.pack("!f", self.bound.item())
        return data

    def unpack(self, stream: bytes) -> int:
        bits = struct.unpack("!h", stream[0:2])[0]
        bound = struct.unpack("!f", stream[2:6])[0]

        self.bits = Tensor([bits])
        self.bound = Tensor([bound])

        LOGGER.debug(f"Unpacked quantization values::  bits: {self.bits} bound: {self.bound}")

        return 6

    def calibrate(self, x: Tensor):
        self.bound = Tensor([x.abs().max().item()]).to(x.device)

        self.bits = self.bits.to(x.device)
        self.zero = self.zero.to(x.device)

        LOGGER.debug(f"New calibrated bound: {self.bound}")

    def _max_symbol(self) -> int:
        return 2 ** (self.bits.to(torch.float32) - 1)

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
