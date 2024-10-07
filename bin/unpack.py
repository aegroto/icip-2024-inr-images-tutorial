import argparse
from modules.device import load_device
import torch

from modules.helpers.config import load_config
from modules.logging import init_logger, setup_logging
from modules.nn.quantizer import initialize_quantizers
from modules.packing import unpack_model
from modules.packing.bytestream import ByteStream

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

    packed_stream = ByteStream(open(args.packed_path, "rb").read())

    unpacked_state_dict = unpack(config, packed_stream, device)

    torch.save(unpacked_state_dict, args.state_dump_path)


def unpack(config, packed_stream, device):
    model = config.model_builder()
    initialize_quantizers(model, config.quantizer_builder)
    model.to(device)

    LOGGER.debug(f"Model architecture: {model}")

    with torch.no_grad():
        unpack_model(model, packed_stream)

    return model.state_dict()


if __name__ == "__main__":
    main()
