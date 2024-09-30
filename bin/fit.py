import argparse

from modules.data import load_image_tensor
from modules.logging import init_logger, setup_logging
from modules.nn.coordinates_based import CoordinatesBasedRepresentation
from modules.training import Trainer, TrainerConfiguration

LOGGER = init_logger(__name__)


def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    return parser.parse_args()


def main():
    setup_logging()
    LOGGER.info("Running")

    args = __load_args()

    LOGGER.debug(f"Command-line args: {args}")

    image = load_image_tensor(args.file_path)
    model = CoordinatesBasedRepresentation()

    trainer = Trainer(TrainerConfiguration(
        iterations=10
    ), model, image)

    trainer.train()

if __name__ == "__main__":
    main()
