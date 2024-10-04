import argparse
from modules.device import load_device
from modules.helpers.config import load_config
from modules.logging import init_logger, setup_logging

LOGGER = init_logger(__name__)


def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--config", type=str, required=False, default="default")
    return parser.parse_args()

def main():
    setup_logging()
    args = __load_args()
    LOGGER.debug(f"Command-line args: {args}")

    _config = load_config(args.config)
    _device = load_device()

def encode(config, device):
    pass

if __name__ == "__main__":
    main()
