import logging
import argparse

from modules.logging import setup_logging

LOGGER = logging.getLogger(__name__)

def __load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    return parser.parse_args()


def main():
    setup_logging()
    LOGGER.info("Running")

    args = __load_args()

    LOGGER.debug(f"Command-line args: {args}")

if __name__ == "__main__":
    main()
