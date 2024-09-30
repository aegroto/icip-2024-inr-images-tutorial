import logging
import argparse

from modules.logging import setup_logging

LOGGER = logging.getLogger(__name__)

def main():
    setup_logging()
    LOGGER.info("Running")

if __name__ == "__main__":
    main()
