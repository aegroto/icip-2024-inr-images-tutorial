import argparse
import importlib
import torch

from skimage import io

from modules.data import dump_reconstructed_tensor, load_image_tensor
from modules.device import load_device
from modules.logging import init_logger, setup_logging
from modules.nn.image_representation.coordinates_based import CoordinatesBasedRepresentation
from modules.nn.mlp import MultiLayerPerceptronConfig
from modules.nn.positional_encoder import PositionalEncoderConfig
from modules.training import Trainer, TrainerConfiguration

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

    device = load_device()
    config = importlib.import_module(f"config.{args.config}")

    state_dict = torch.load(args.state_path, weights_only=True)

    model = config.model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    (target_height, target_width) = __parse_resolution(args.resolution)
    target_shape = torch.Size([target_height, target_width])
    input = model.generate_input(target_shape).to(device)

    with torch.no_grad():
        reconstructed_tensor = model(input)

    dump_reconstructed_tensor(reconstructed_tensor, args.output_dump_path)


if __name__ == "__main__":
    main()
