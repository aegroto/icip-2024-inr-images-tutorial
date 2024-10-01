from torch import Tensor, nn

from modules.logging import init_logger

LOGGER = init_logger(__name__)


class Quantizer(nn.Module):
    def quantize(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def dequantize(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, unquantized: Tensor) -> Tensor:
        dequantized = self.dequantize(self.quantize(unquantized))
        error = unquantized - dequantized

        estimated_quantized = unquantized + error

        return estimated_quantized


class IQuantizable:
    def set_quantizer(self, quantizer: Quantizer):
        raise NotImplementedError
