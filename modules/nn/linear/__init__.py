import torch
import math
from torch import Tensor, nn

class QuantizableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.empty(out_features))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        bias = self.bias

        return nn.functional.linear(x, weight, bias)
