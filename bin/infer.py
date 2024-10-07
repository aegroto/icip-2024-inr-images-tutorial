import argparse
import torch


from modules.data import dump_reconstructed_tensor
from modules.device import load_device
from modules.helpers.config import load_config
from modules.logging import init_logger, setup_logging
from modules.nn.quantizer import initialize_quantizers

LOGGER = init_logger(__name__)


def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_path", type=str)
    parser.add_argument("resolution", type=str)
    parser.add_argument("output_dump_path", type=str)
    parser.add_argument("--config", type=str, required=False, default="default")
    return parser.parse_args()


def parse_resolution(resolution_str: str) -> tuple[int, int]:
    (width_str, height_str) = tuple(resolution_str.split("x"))

    return (int(height_str), int(width_str))


def main():
    setup_logging()
    args = __load_args()
    LOGGER.debug(f"Command-line args: {args}")

    device = load_device()
    config = load_config(args.config)

    resolution = parse_resolution(args.resolution)
    state_dict = torch.load(args.state_path, weights_only=True)

    infer(config, state_dict, resolution, args.output_dump_path, device)


def infer(config, state_dict, resolution, dump_path, device):
    model = config.model_builder()
    initialize_quantizers(model, config.quantizer_builder)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    LOGGER.debug(f"Model architecture: {model}")

    (target_height, target_width) = resolution
    target_shape = torch.Size([target_height, target_width])
    input = model.generate_input(target_shape).to(device)

    with torch.no_grad():
        reconstructed_tensor = model(input)

    dump_reconstructed_tensor(reconstructed_tensor, dump_path)


if __name__ == "__main__":
    main()
