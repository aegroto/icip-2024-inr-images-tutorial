import argparse
from modules.logging import init_logger, setup_logging

LOGGER = init_logger(__name__)


def __load_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    setup_logging()
    args = __load_args()
    LOGGER.debug(f"Command-line args: {args}")

    LOGGER.warning("Not implemented yet")

if __name__ == "__main__":
    main()
