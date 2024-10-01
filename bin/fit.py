import importlib
import argparse
from modules.device import load_device
import torch

from modules.data import load_image_tensor
from modules.logging import init_logger, setup_logging
from modules.training import Trainer

LOGGER = init_logger(__name__)


def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("state_dump_path", type=str)
    parser.add_argument("--config", type=str, required=False, default="default")
    return parser.parse_args()


def main():
    setup_logging()

    LOGGER.info("Running")

    args = __load_args()

    LOGGER.debug(f"Command-line args: {args}")

    device = load_device()
    config = importlib.import_module(f"config.{args.config}")

    image = load_image_tensor(args.file_path).to(device)
    model = config.model.to(device)

    LOGGER.debug(f"Model architecture: {model}")

    trainer = Trainer(config.trainer_configuration, model, image, device)

    try:
        trainer.train()
    except KeyboardInterrupt:
        LOGGER.warning("Training interrupted, dumping the current model")

    model, best_loss = trainer.best_result()
    LOGGER.info(f"Best loss value: {best_loss}")

    model.eval()
    torch.save(model.state_dict(), args.state_dump_path)


if __name__ == "__main__":
    main()
