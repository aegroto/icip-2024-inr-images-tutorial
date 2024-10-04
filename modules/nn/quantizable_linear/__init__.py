from typing import Callable
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
from modules.packing.bytestream import ByteStream

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

    def __pack_tensor(self, tensor: Tensor, quantizer: Quantizer) -> ByteStream:
        LOGGER.debug(" --- Packing")

        stream = ByteStream()
        stream.append(quantizer.pack())

        LOGGER.debug(f"Tensor mean: {tensor.mean()}")

        quantized = quantizer.quantize(tensor)

        LOGGER.debug(f"Quantized tensor mean: {quantized.mean()}")

        encoded_stream = entropy_encode(
            quantized, build_range_encoder, build_laplace_entropy_model
        )
        stream.append(encoded_stream)

        return stream

    def __unpack_tensor(self, tensor: Tensor, quantizer: Quantizer, stream: ByteStream):
        LOGGER.debug(" --- Unpacking")
        quantizer.unpack(stream)

        quantized = entropy_decode(
            stream, build_range_decoder, build_laplace_entropy_model
        )
        LOGGER.debug(f"Unpacked quantized tensor mean: {quantized.mean()}")

        quantizer.to(tensor.device)
        quantized = quantized.to(tensor.device).reshape(tensor.shape)

        dequantized = quantizer.dequantize(quantized)
        tensor.set_(dequantized)

        LOGGER.debug(f"Unpacked tensor mean: {tensor.mean()}")

    def pack(self) -> ByteStream:
        stream = ByteStream()
        stream.append(self.__pack_tensor(self.weight, self.weight_quantizer))
        stream.append(self.__pack_tensor(self.bias, self.bias_quantizer))
        return stream

    def unpack(self, stream: ByteStream):
        self.__unpack_tensor(self.weight, self.weight_quantizer, stream)
        self.__unpack_tensor(self.bias, self.bias_quantizer, stream)

    def forward(self, x: Tensor) -> Tensor:
        (equantized_weight, equantized_bias) = self.__get_estimated_quantized_params()

        return nn.functional.linear(x, equantized_weight, equantized_bias)
