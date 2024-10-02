import argparse
import torch
import os
from bin.export_stats import export_stats
from bin.fit import fit
from bin.infer import infer
from modules.data import read_image_resolution
from modules.device import load_device
from modules.helpers.config import load_config
from modules.logging import init_logger, setup_logging
from modules.nn.quantizer import inject_quantizer

LOGGER = init_logger(__name__)


def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("uncompressed_image_path", type=str)
    parser.add_argument("results_folder", type=str)
    parser.add_argument("--config", type=str, required=False, default="default")
    parser.add_argument("--initial_state_dict_path", type=str, required=False, default=None)
    return parser.parse_args()


def __run_phase(config, args, phase_name, device, initial_state_dict=None):
    base_folder = f"{args.results_folder}/{phase_name}/"

    state_dump_path = f"{base_folder}/state.pth"
    inferred_image_path = f"{base_folder}/inferred.png"
    stats_dump_path = f"{base_folder}/stats.json"

    os.makedirs(base_folder, exist_ok=True)

    image_resolution = read_image_resolution(args.uncompressed_image_path)

    trained_state_dict = fit(
        config,
        args.uncompressed_image_path,
        device,
        state_dump_path,
        initial_state_dict,
    )
    infer(
        config,
        trained_state_dict,
        image_resolution,
        inferred_image_path,
        device,
    )
    export_stats(
        args.uncompressed_image_path,
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
        current_state_dict = torch.load(args.initial_state_dict_path)
    else:
        current_state_dict = None
    
    for (phase_name, phase_config) in config.phases.items():
        current_state_dict = __run_phase(
            phase_config, args, phase_name, device, current_state_dict
        )


if __name__ == "__main__":
    main()