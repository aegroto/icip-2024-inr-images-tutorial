import copy
import argparse
from modules.device import load_device
import torch

from modules.data import load_image_tensor
from modules.helpers.config import load_config
from modules.logging import init_logger, setup_logging
from modules.nn.quantizer import (
    apply_quantization,
    initialize_quantizers,
    recalibrate_quantizers,
)
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
    args = __load_args()
    LOGGER.debug(f"Command-line args: {args}")

    device = load_device()
    config = load_config(args.config)

    fit(config, args.file_path, device, args.state_dump_path)


def fit(config, image_file_path, device, state_dump_path=None, initial_state_dict=None):
    image = load_image_tensor(image_file_path).to(device)
    model = config.model_builder()

    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)

    initialize_quantizers(model, config.quantizer_builder)
    model.to(device)

    if config.recalibrate_quantizers:
        LOGGER.debug("Recalibrating quantizers...")
        recalibrate_quantizers(model)

    LOGGER.debug(f"Model architecture: {model}")

    trainer = Trainer(config.trainer_configuration, model, image, device)

    try:
        trainer.train()
    except KeyboardInterrupt:
        LOGGER.warning("Training interrupted, dumping the current model")

    model, best_loss = trainer.best_result()
    LOGGER.info(f"Best loss value: {best_loss}")

    model.apply(apply_quantization)

    fitted_state_dict = copy.deepcopy(model.state_dict())

    if state_dump_path:
        torch.save(fitted_state_dict, state_dump_path)

    return fitted_state_dict


if __name__ == "__main__":
    main()
