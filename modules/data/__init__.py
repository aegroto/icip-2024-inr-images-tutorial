import torchvision
import PIL

from modules.logging import init_logger

LOGGER = init_logger(__name__)


def load_image_tensor(path):
    pil_image = PIL.Image.open(path)
    image = torchvision.transforms.functional.to_tensor(pil_image)

    LOGGER.debug(f"Loaded image shape: {image.shape}")

    return image
