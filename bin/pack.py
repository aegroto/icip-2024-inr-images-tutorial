import argparse
import os
from modules.device import load_device
import torch

from modules.helpers.config import load_config
from modules.logging import init_logger, setup_logging
from modules.nn.quantizer import initialize_quantizers
from modules.packing import pack_model

LOGGER = init_logger(__name__)


def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--config", type=str, required=False, default="default")
    return parser.parse_args()


def main():
    setup_logging()
    args = __load_args()
    LOGGER.debug(f"Command-line args: {args}")

    device = load_device()
    config = load_config(args.config)

    state_dict = torch.load(args.state_path, weights_only=True)

    pack(config, state_dict, args.output_path, device)


def pack(config, state_dict, output_path, device):
    model = config.model_builder()
    initialize_quantizers(model, config.quantizer_builder)
    model.load_state_dict(state_dict)
    model.to(device)

    LOGGER.debug(f"Model architecture: {model}")

    stream = pack_model(model)

    LOGGER.debug(f"Packed model stream length: {len(stream)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as file:
        file.write(stream.get_bytes())


if __name__ == "__main__":
    main()
