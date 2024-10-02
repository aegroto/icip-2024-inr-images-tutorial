import struct
import numpy
import torch
import copy
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

    def __get_quantized_params(self):
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_bias = self.weight_quantizer(self.bias)

        return (quantized_weight, quantized_bias)

    def set_quantizer(self, quantizer: Quantizer):
        self.weight_quantizer = copy.deepcopy(quantizer)
        self.bias_quantizer = copy.deepcopy(quantizer)

        self.weight_quantizer.calibrate(self.weight)
        self.bias_quantizer.calibrate(self.bias)

    def apply_quantization(self):
        (quantized_weight, quantized_bias) = self.__get_quantized_params()

        self.weight.set_(quantized_weight)
        self.bias.set_(quantized_bias)

    def __pack_tensor(self, tensor: Tensor, quantizer: Quantizer) -> bytes:
        data = bytes()
        data += quantizer.pack()

        # quantized = quantizer.quantize(tensor)
        # serialized_tensor = (
        #     quantized.cpu().to(torch.int8).numpy().astype(numpy.int8).tobytes()
        # )
        # data += serialized_tensor

        return data

    def __unpack_tensor(self, tensor: Tensor, quantizer: Quantizer, stream: bytes) -> int:
        read_bytes = quantizer.unpack(stream)

        return read_bytes

    def pack(self) -> bytes:
        data = bytes()
        data += self.__pack_tensor(self.weight, self.weight_quantizer)
        data += self.__pack_tensor(self.bias, self.bias_quantizer)
        return data

    def unpack(self, stream: bytes) -> int:
        read_bytes = self.__unpack_tensor(self.weight, self.weight_quantizer, stream)
        read_bytes += self.__unpack_tensor(self.bias, self.bias_quantizer, stream[read_bytes:])
        return read_bytes

    def forward(self, x: Tensor) -> Tensor:
        (quantized_weight, quantized_bias) = self.__get_quantized_params()

        return nn.functional.linear(x, quantized_weight, quantized_bias)
