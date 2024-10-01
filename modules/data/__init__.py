import torchvision
import PIL
import torch
from skimage import io

from modules.logging import init_logger

LOGGER = init_logger(__name__)


def load_image_tensor(path) -> torch.Tensor:
    pil_image = PIL.Image.open(path)
    image = torchvision.transforms.functional.to_tensor(pil_image)

    LOGGER.debug(f"Loaded image shape: {image.shape}")

    return image


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


def read_image_resolution(path):
    image_tensor = load_image_tensor(path)

    (height, width) = (image_tensor.shape[1], image_tensor.shape[2])

    return (height, width)
