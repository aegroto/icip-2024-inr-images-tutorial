import torch
import os
import random

from modules.logging import init_logger

LOGGER = init_logger(__name__)


def load_seed_from_env():
    try:
        seed = os.environ["RANDOM_SEED"]
        torch.manual_seed(seed)
        random.seed(seed)
    except KeyError:
        LOGGER.warning("Random seed not set, results may not be reproducible")
