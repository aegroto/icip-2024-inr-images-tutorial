from modules.logging import init_logger, setup_logging
import torch

LOGGER = init_logger(__name__)


def main():
    setup_logging()

    LOGGER.info(f"Torch version: {torch.__version__}")
    LOGGER.info(f"Cuda availability: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
