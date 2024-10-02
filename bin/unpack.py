import copy
import argparse
from modules.device import load_device
import torch

from modules.helpers.config import load_config
from modules.logging import init_logger, setup_logging
from modules.nn.quantizer import inject_quantizer
from modules.packing import pack_model, unpack_model

LOGGER = init_logger(__name__)


def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("packed_path", type=str)
    parser.add_argument("state_dump_path", type=str)
    parser.add_argument("--config", type=str, required=False, default="default")
    return parser.parse_args()


def main():
    setup_logging()
    args = __load_args()
    LOGGER.debug(f"Command-line args: {args}")

    device = load_device()
    config = load_config(args.config)

    packed_stream = open(args.packed_path, "rb").read()

    unpack(config, packed_stream, args.state_dump_path, device)


def unpack(config, packed_stream, output_path, device):
    model = copy.deepcopy(config.model).to(device)

    model.apply(lambda module: inject_quantizer(module, config.quantizer_builder))

    LOGGER.debug(f"Model architecture: {model}")

    unpack_model(model, packed_stream)

if __name__ == "__main__":
    main()
