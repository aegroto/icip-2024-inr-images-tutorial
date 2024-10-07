from typing import Tuple
import torchvision
import PIL
import torch
from skimage import io

from modules.logging import init_logger

LOGGER = init_logger(__name__)


class ImageData:
    def __init__(self, path, device):
        pil_image = PIL.Image.open(path)

        self.path = path
        self.tensor = torchvision.transforms.functional.to_tensor(pil_image).to(device)
        self.height = self.tensor.shape[1]
        self.width = self.tensor.shape[2]

    def resolution(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def num_pixels(self) -> int:
        return self.width * self.height


def dump_reconstructed_tensor(reconstructed_tensor: torch.Tensor, path: str):
    reconstructed_image = (
        reconstructed_tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    LOGGER.debug(f"Dumped image shape: {reconstructed_image.shape}")

    io.imsave(path, reconstructed_image)
