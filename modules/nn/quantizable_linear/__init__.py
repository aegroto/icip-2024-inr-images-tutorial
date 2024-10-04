import struct
from typing import Callable
import numpy
import torch
import math
from torch import Tensor, nn

from modules.entropy_coding import (
    build_laplace_entropy_model,
    build_range_decoder,
    build_range_encoder,
    entropy_decode,
    entropy_encode,
)
from modules.logging import init_logger
from modules.nn.quantizer import IQuantizable, Quantizer
from modules.nn.quantizer.dummy import DummyQuantizer
from modules.packing import IPackable

LOGGER = init_logger(__name__)


class QuantizableLinear(nn.Module, IQuantizable, IPackable):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_quantizer = DummyQuantizer()
        self.bias_quantizer = DummyQuantizer()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.empty(out_features))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def __get_estimated_quantized_params(self):
        equantized_weight = self.weight_quantizer(self.weight)
        equantized_bias = self.bias_quantizer(self.bias)

        return (equantized_weight, equantized_bias)

    def init_quantizers(self, quantizer_builder: Callable):
        self.weight_quantizer = quantizer_builder(self)
        self.bias_quantizer = quantizer_builder(self)

    def recalibrate_quantizers(self):
        self.weight_quantizer.calibrate(self.weight)
        self.bias_quantizer.calibrate(self.bias)

    def apply_quantization(self):
        (equantized_weight, equantized_bias) = self.__get_estimated_quantized_params()

        self.weight.set_(equantized_weight)
        self.bias.set_(equantized_bias)

    def __pack_tensor(self, tensor: Tensor, quantizer: Quantizer) -> bytes:
        LOGGER.debug(" --- Packing")

        data = bytes()
        data += quantizer.pack()

        LOGGER.debug(f"Tensor mean: {tensor.mean()}")

        quantized = quantizer.quantize(tensor)

        LOGGER.debug(f"Quantized tensor mean: {quantized.mean()}")

        serialized_tensor = entropy_encode(
            quantized, build_range_encoder, build_laplace_entropy_model
        )
        data += serialized_tensor

        return data

    def __unpack_tensor(
        self, tensor: Tensor, quantizer: Quantizer, stream: bytes
    ) -> int:
        LOGGER.debug(" --- Unpacking")
        read_bytes = quantizer.unpack(stream)

        (quantized, decoding_read_bytes) = entropy_decode(
            stream[read_bytes:], build_range_decoder, build_laplace_entropy_model
        )
        read_bytes += decoding_read_bytes
        LOGGER.debug(f"Unpacked quantized tensor mean: {quantized.mean()}")

        quantizer.to(tensor.device)
        quantized = quantized.to(tensor.device).reshape(tensor.shape)

        dequantized = quantizer.dequantize(quantized)
        tensor.set_(dequantized)

        LOGGER.debug(f"Unpacked tensor mean: {tensor.mean()}")

        return read_bytes

    def pack(self) -> bytes:
        data = bytes()
        data += self.__pack_tensor(self.weight, self.weight_quantizer)
        data += self.__pack_tensor(self.bias, self.bias_quantizer)
        return data

    def unpack(self, stream: bytes) -> int:
        read_bytes = self.__unpack_tensor(self.weight, self.weight_quantizer, stream)
        read_bytes += self.__unpack_tensor(
            self.bias, self.bias_quantizer, stream[read_bytes:]
        )
        return read_bytes

    def forward(self, x: Tensor) -> Tensor:
        (equantized_weight, equantized_bias) = self.__get_estimated_quantized_params()

        return nn.functional.linear(x, equantized_weight, equantized_bias)
