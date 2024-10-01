from torch import Tensor, nn

from modules.logging import init_logger

LOGGER = init_logger(__name__)

class Quantizer(nn.Module):
    def quantize(self, x: Tensor) -> Tensor:
        LOGGER.warning("Call to default quantize implementation")
        return None

    def dequantize(self, x: Tensor) -> Tensor:
        LOGGER.warning("Call to default dequantize implementation")
        return None

    def forward(self, unquantized: Tensor) -> Tensor:
        dequantized = self.dequantize(self.quantize(unquantized))
        error = unquantized - dequantized

        estimated_quantized = unquantized + error

        return estimated_quantized
