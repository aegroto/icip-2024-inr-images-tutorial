from typing import Callable
from torch import Tensor, nn

from modules.logging import init_logger

LOGGER = init_logger(__name__)


class Quantizer(nn.Module):
    def calibrate(self, x: Tensor):
        raise NotImplementedError

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

def inject_quantizer(module: nn.Module, quantizer_builder: Callable):
    LOGGER.debug(f"Injecting quantizer in module {module}")

    if isinstance(module, IQuantizable):
        module.set_quantizer(quantizer_builder(module))
