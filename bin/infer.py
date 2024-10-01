import argparse
import copy
import torch


from modules.data import dump_reconstructed_tensor
from modules.helpers.config import load_config
from modules.logging import init_logger, setup_logging

LOGGER = init_logger(__name__)


def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_path", type=str)
    parser.add_argument("resolution", type=str)
    parser.add_argument("output_dump_path", type=str)
    parser.add_argument("--config", type=str, required=False, default="default")
    return parser.parse_args()


def __parse_resolution(resolution_str: str) -> tuple[int, int]:
    (width_str, height_str) = tuple(resolution_str.split("x"))

    return (int(height_str), int(width_str))


def main():
    setup_logging()
    args = __load_args()
    LOGGER.debug(f"Command-line args: {args}")

    config = load_config(args.config)

    resolution = __parse_resolution(args.resolution)
    state_dict = torch.load(args.state_path, weights_only=True)

    infer(config, state_dict, resolution, args.output_dump_path)


def infer(config, state_dict, resolution, dump_path, device):
    model = copy.deepcopy(config.model).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    (target_height, target_width) = resolution
    target_shape = torch.Size([target_height, target_width])
    input = model.generate_input(target_shape).to(device)

    with torch.no_grad():
        reconstructed_tensor = model(input)

    dump_reconstructed_tensor(reconstructed_tensor, dump_path)


if __name__ == "__main__":
    main()
