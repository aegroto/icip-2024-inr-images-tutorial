import argparse
import os
from bin.export_stats import export_stats
from bin.fit import fit
from bin.infer import infer
from modules.data import read_image_resolution
from modules.device import load_device
from modules.helpers.config import load_config
from modules.logging import init_logger, setup_logging

LOGGER = init_logger(__name__)


def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("uncompressed_image_path", type=str)
    parser.add_argument("results_folder", type=str)
    parser.add_argument("--config", type=str, required=False, default="default")
    return parser.parse_args()


def main():
    setup_logging()
    args = __load_args()
    LOGGER.debug(f"Command-line args: {args}")

    config = load_config(args.config)
    device = load_device()
    image_resolution = read_image_resolution(args.uncompressed_image_path)

    folder = args.results_folder
    fitted_state_dump_path = f"{folder}/fitted/state.pth"
    fitted_inferred_image_path = f"{folder}/fitted/inferred.png"
    stats_dump_path = f"{folder}/fitted/stats.json"

    os.makedirs(f"{folder}/fitted/", exist_ok=True)

    fitted_state_dict = fit(
        config, args.uncompressed_image_path, device, fitted_state_dump_path
    )
    infer(
        config, fitted_state_dict, image_resolution, fitted_inferred_image_path, device
    )
    export_stats(
        args.uncompressed_image_path,
        fitted_inferred_image_path,
        fitted_state_dump_path,
        stats_dump_path,
        device,
    )


if __name__ == "__main__":
    main()
