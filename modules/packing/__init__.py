from typing import List
import torch

from modules.logging import init_logger
from modules.nn.image_representation.base import ImplicitImageRepresentation
from modules.nn.quantizer import IQuantizable

LOGGER = init_logger(__name__)


class IPackable:
    def pack(self) -> bytes:
        raise NotImplementedError


def __fetch_packable_modules(list: List[torch.nn.Module], module: torch.nn.Module):
    if isinstance(module, IPackable):
        list.append(module)
    else:
        for submodule in module.children():
            __fetch_packable_modules(list, submodule)


def pack_model(model: ImplicitImageRepresentation) -> bytes:
    LOGGER.debug("Packing model")

    stream = bytes()

    packable_modules: List[IPackable] = list()
    __fetch_packable_modules(packable_modules, model)

    for module in packable_modules:
        packed_module = module.pack()
        LOGGER.debug(f"Packed module size: {len(packed_module)}")
        stream += packed_module

    LOGGER.debug(f"Total stream size: {len(stream)}")

    return stream
