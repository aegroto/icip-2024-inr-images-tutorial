import struct
from typing import Callable
import numpy
import torch
import math
from torch import Tensor, nn

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

        with torch.no_grad():
            LOGGER.debug(f"Unquantized weight mean: {self.weight.mean()}")
            test_param = equantized_weight
            LOGGER.debug(f"Estimated quantized weight mean: {test_param.mean()}")
            test_param = self.weight_quantizer(test_param)
            LOGGER.debug(f"2nd Estimated quantized weight mean: {test_param.mean()}")
            test_param = self.weight_quantizer(test_param)
            LOGGER.debug(f"3rd Estimated quantized weight mean: {test_param.mean()}")

            LOGGER.debug(f"Unquantized bias mean: {self.bias.mean()}")
            test_param = equantized_bias
            LOGGER.debug(f"Estimated quantized bias mean: {test_param.mean()}")
            test_param = self.bias_quantizer(test_param)
            LOGGER.debug(f"2nd Estimated quantized bias mean: {test_param.mean()}")
            test_param = self.bias_quantizer(test_param)
            LOGGER.debug(f"3rd Estimated quantized bias mean: {test_param.mean()}")

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
        data = bytes()
        data += quantizer.pack()

        LOGGER.debug(f"Tensor to be packed mean: {tensor.mean()}")

        quantized = quantizer.quantize(tensor)

        LOGGER.debug(f"Quantized tensor to be packed mean: {quantized.mean()}")

        dequantized_test = quantizer.dequantize(quantized)
        LOGGER.debug(f"Dequantized test tensor mean: {dequantized_test.mean()}")

        serialized_tensor = (
            quantized.cpu().to(torch.int8).numpy().astype(numpy.int8).tobytes()
        )
        data += struct.pack("!I", len(serialized_tensor))
        data += serialized_tensor

        LOGGER.debug(f"Serialized tensor length: {len(serialized_tensor)}")

        return data

    def __unpack_tensor(
        self, tensor: Tensor, quantizer: Quantizer, stream: bytes
    ) -> int:
        read_bytes = quantizer.unpack(stream)

        serialized_tensor_len = struct.unpack(
            "!I", stream[read_bytes : read_bytes + 4]
        )[0]
        read_bytes += 4
        serialized_tensor_bytes = stream[
            read_bytes : read_bytes + serialized_tensor_len
        ]
        read_bytes += serialized_tensor_len
        array = numpy.frombuffer(serialized_tensor_bytes, numpy.int8).copy()

        LOGGER.debug(f"Unpacked array size: {len(array)}")

        quantized_tensor = (
            torch.from_numpy(array)
            .to(torch.float32)
            .to(tensor.device)
            .reshape(tensor.shape)
        )
        LOGGER.debug(f"Unpacked quantized tensor mean: {quantized_tensor.mean()}")

        quantizer.to(tensor.device)
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        tensor.set_(dequantized_tensor)

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
