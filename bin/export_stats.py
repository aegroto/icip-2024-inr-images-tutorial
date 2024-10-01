import argparse
import json
import os
import torch


from modules.data import load_image_tensor
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

    original_image = load_image_tensor(args.original_image_path).to(device)
    reconstructed_image = load_image_tensor(args.reconstructed_image_path).to(device)
    compressed_file_size = os.stat(args.compressed_file_path).st_size

    num_pixels = original_image.numel() / 3

    stats = dict()
    stats["psnr"] = __calculate_psnr(original_image, reconstructed_image)
    stats["bpp"] = (compressed_file_size * 8) / num_pixels

    LOGGER.info(json.dumps(stats, indent=4))

    json.dump(stats, open(args.stats_dump_path, "w"), indent=4)


if __name__ == "__main__":
    main()
