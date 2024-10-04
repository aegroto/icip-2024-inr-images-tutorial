import argparse
import json
import os
import torch


from modules.data import ImageData
from modules.device import load_device
from modules.logging import init_logger, setup_logging


LOGGER = init_logger(__name__)


def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("original_image_path", type=str)
    parser.add_argument("reconstructed_image_path", type=str)
    parser.add_argument("compressed_file_path", type=str)
    parser.add_argument("stats_dump_path", type=str)
    return parser.parse_args()


def __calculate_psnr(
    original: torch.Tensor, reconstructed: torch.Tensor, value_range: float = 1.0
) -> float:
    mse = (original - reconstructed).square().mean()
    psnr = 20.0 * torch.log10(value_range / mse.sqrt())
    return psnr.item()


def main():
    setup_logging()

    args = __load_args()

    LOGGER.debug(f"Command-line args: {args}")

    device = load_device()
    export_stats(
        args.original_image_path,
        args.reconstructed_image_path,
        args.compressed_file_path,
        args.stats_dump_path,
        device,
    )


def export_stats(
    original_image_path,
    reconstructed_image_path,
    compressed_file_path,
    stats_dump_path,
    device,
):
    original_image = ImageData(original_image_path, device)
    reconstructed_image = ImageData(reconstructed_image_path, device)
    compressed_file_size = os.stat(compressed_file_path).st_size

    num_pixels = original_image.tensor.numel() / 3

    stats = dict()
    stats["psnr"] = __calculate_psnr(original_image.tensor, reconstructed_image.tensor)
    stats["bpp"] = (compressed_file_size * 8) / num_pixels

    LOGGER.info(json.dumps(stats, indent=4))

    json.dump(stats, open(stats_dump_path, "w"), indent=4)


if __name__ == "__main__":
    main()
