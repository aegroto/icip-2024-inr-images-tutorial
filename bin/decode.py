import argparse
from bin.infer import infer, parse_resolution
from bin.unpack import unpack
from modules.device import load_device
from modules.helpers.config import load_config
from modules.logging import init_logger, setup_logging
from modules.packing.bytestream import ByteStream

LOGGER = init_logger(__name__)


def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("encoded_path", type=str)
    parser.add_argument("resolution", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--config", type=str, required=False, default="default")
    return parser.parse_args()


def main():
    setup_logging()
    args = __load_args()
    LOGGER.debug(f"Command-line args: {args}")

    config = load_config(args.config)
    device = load_device()

    encoded_stream = ByteStream(open(args.encoded_path, "rb").read())

    decode(
        config,
        encoded_stream,
        parse_resolution(args.resolution),
        args.output_path,
        device,
    )


def decode(config, encoded_stream, resolution, output_path, device):
    unpacked_state_dict = unpack(config, encoded_stream, device)
    infer(config, unpacked_state_dict, resolution, output_path, device)


if __name__ == "__main__":
    main()
