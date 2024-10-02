import torch
import copy
import math
from torch import Tensor, nn

from modules.nn.quantizer import IQuantizable, Quantizer
from modules.nn.quantizer.dummy import DummyQuantizer


class QuantizableLinear(nn.Module, IQuantizable):
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

    def forward(self, x: Tensor) -> Tensor:
        (quantized_weight, quantized_bias) = self.__get_quantized_params()

        return nn.functional.linear(x, quantized_weight, quantized_bias)
