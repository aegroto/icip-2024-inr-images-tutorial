import argparse
from modules.device import load_device
import torch

from modules.data import load_image_tensor
from modules.logging import init_logger, setup_logging
from modules.nn.image_representation.coordinates_based import (
    CoordinatesBasedRepresentation,
)
from modules.nn.mlp import MultiLayerPerceptronConfig
from modules.nn.positional_encoder import PositionalEncoderConfig
from modules.training import Trainer, TrainerConfiguration

LOGGER = init_logger(__name__)


def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("state_dump_path", type=str)
    return parser.parse_args()


def main():
    setup_logging()

    LOGGER.info("Running")

    args = __load_args()

    LOGGER.debug(f"Command-line args: {args}")

    device = load_device()

    image = load_image_tensor(args.file_path).to(device)
    model = CoordinatesBasedRepresentation(
        encoder_config=PositionalEncoderConfig(num_frequencies=16, scale=1.4),
        network_config=MultiLayerPerceptronConfig(
            input_features=2,
            hidden_features=512,
            hidden_layers=2,
            output_features=3,
            activation_builder=lambda: torch.nn.GELU(),
        ),
    ).to(device)

    LOGGER.debug(f"Model architecture: {model}")

    trainer = Trainer(TrainerConfiguration(iterations=1000, optimizer_parameters={
        "lr": 1.0e-2
    }), model, image, device)

    trainer.train()

    model.eval()
    torch.save(model.state_dict(), args.state_dump_path)


if __name__ == "__main__":
    main()
