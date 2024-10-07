import argparse
import copy
import torch
import os
from bin.export_stats import export_stats
from bin.infer import infer
from modules.data import ImageData
from modules.device import load_device
from modules.helpers.config import load_config
from modules.logging import init_logger, setup_logging
from modules.nn.quantizer import (
    apply_quantization,
    initialize_quantizers,
    recalibrate_quantizers,
)

LOGGER = init_logger(__name__)


def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_file_path", type=str)
    parser.add_argument("results_folder", type=str)
    parser.add_argument("--config", type=str, required=False, default="default")
    parser.add_argument(
        "--initial_state_dict_path", type=str, required=False, default=None
    )
    return parser.parse_args()


def __fit_in_phase(config, image, device, initial_state_dict=None):
    model = config.model_builder()

    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)

    initialize_quantizers(model, config.quantizer_builder)
    model.to(device)

    if config.recalibrate_quantizers:
        LOGGER.debug("Recalibrating quantizers...")
        recalibrate_quantizers(model)

    LOGGER.debug(f"Model architecture: {model}")

    trainer = config.trainer_builder(model, image, device)

    try:
        trainer.train()
    except KeyboardInterrupt:
        LOGGER.warning("Training interrupted, dumping the current model")

    model, best_loss = trainer.best_result()
    LOGGER.info(f"Best loss value: {best_loss}")

    model.apply(apply_quantization)

    fitted_state_dict = copy.deepcopy(model.state_dict())

    return fitted_state_dict


def __run_phase(
    config, image_data: ImageData, device, initial_state_dict=None, dump_folder=None
):
    trained_state_dict = __fit_in_phase(
        config,
        image_data.tensor,
        device,
        initial_state_dict=initial_state_dict,
    )

    if dump_folder is not None:
        state_dump_path = f"{dump_folder}/state.pth"
        inferred_image_path = f"{dump_folder}/inferred.png"
        stats_dump_path = f"{dump_folder}/stats.json"

        os.makedirs(dump_folder, exist_ok=True)

        torch.save(trained_state_dict, state_dump_path)

        infer(
            config,
            trained_state_dict,
            image_data.resolution(),
            inferred_image_path,
            device,
        )
        export_stats(
            image_data.path,
            inferred_image_path,
            state_dump_path,
            stats_dump_path,
            device,
        )

    return trained_state_dict


def main():
    setup_logging()
    args = __load_args()
    LOGGER.debug(f"Command-line args: {args}")

    config = load_config(args.config)
    device = load_device()

    if args.initial_state_dict_path is not None:
        initial_state_dict = torch.load(args.initial_state_dict_path, weights_only=True)
    else:
        initial_state_dict = None

    image_data = ImageData(args.image_file_path, device)

    fit(config, image_data, device, initial_state_dict, args.results_folder)


def fit(config, image_data, device, initial_state_dict=None, results_folder=None):
    current_state_dict = initial_state_dict

    for phase_name, phase_config in config.phases.items():
        if results_folder:
            dump_folder = f"{results_folder}/{phase_name}"
        else:
            dump_folder = None

        current_state_dict = __run_phase(
            phase_config, image_data, device, current_state_dict, dump_folder
        )

    return current_state_dict


if __name__ == "__main__":
    main()
