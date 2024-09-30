import argparse
import torchvision

from modules.data import load_image_tensor
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

    image = load_image_tensor(args.file_path)

if __name__ == "__main__":
    main()
