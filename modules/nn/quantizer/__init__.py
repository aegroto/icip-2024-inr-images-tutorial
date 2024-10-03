from typing import Callable
from torch import Tensor, nn
import torch

from modules.logging import init_logger
from modules.nn.image_representation.base import ImplicitImageRepresentation

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
        error = dequantized - unquantized

        estimated_quantized = unquantized + error.detach()

        return estimated_quantized


class IQuantizable:
    def init_quantizers(self, quantizer_builder: Callable):
        raise NotImplementedError

    def apply_quantization(self):
        raise NotImplementedError

    def recalibrate_quantizers(self):
        raise NotImplementedError


def inject_quantizer(module: nn.Module, quantizer_builder: Callable):
    LOGGER.debug(f"Injecting quantizer in module {module}")

    if isinstance(module, IQuantizable):
        module.init_quantizers(quantizer_builder)


def apply_quantization(module: nn.Module):
    with torch.no_grad():
        if isinstance(module, IQuantizable):
            module.apply_quantization()


def recalibrate_quantizers(model: ImplicitImageRepresentation):
    def __callback(module: nn.Module):
        with torch.no_grad():
            if isinstance(module, IQuantizable):
                module.recalibrate_quantizers()

    model.apply(__callback)


def initialize_quantizers(
    model: ImplicitImageRepresentation, quantizer_builder: Callable
):
    if quantizer_builder is None:
        return

    model.apply(lambda module: inject_quantizer(module, quantizer_builder))
