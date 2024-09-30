import argparse
import torchvision

from modules.logging import init_logger, setup_logging

from PIL import Image

LOGGER = init_logger(__name__)

def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    return parser.parse_args()

def main():
    setup_logging()
    LOGGER.info("Running")

    args = __load_args()

    LOGGER.debug(f"Command-line args: {args}")

    pil_image = Image.open(args.file_path)
    image = torchvision.transforms.functional.to_tensor(pil_image)

    LOGGER.debug(f"Loaded image shape: {image.shape}")

if __name__ == "__main__":
    main()
