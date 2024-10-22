import argparse
from bin.fit import fit
from bin.pack import pack
from modules.data import ImageData
from modules.device import load_device
from modules.helpers.config import load_config
from modules.helpers.reproducibility import load_seed_from_env
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

    config = load_config(args.config)
    device = load_device()

    image_data = ImageData(args.image_path, device)

    encode(config, image_data, args.output_path, device)


def encode(config, image_data, output_path, device):
    load_seed_from_env()
    fitted_state_dict = fit(config, image_data, device)
    pack(config, fitted_state_dict, output_path, device)


if __name__ == "__main__":
    main()
