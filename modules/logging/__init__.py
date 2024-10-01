import logging
import os
import re


def __env_logging_config():
    try:
        return os.environ["LOG_LEVEL"]
    except KeyError:
        return "__root__=INFO"


def get_log_level(target_module_name=None):
    if target_module_name is None:
        target_module_name = "__root__"

    config = __env_logging_config()

    for module_config in config.split(","):
        (module_expr, log_level) = tuple(module_config.split("="))

        if re.match(module_expr, target_module_name):
            return log_level

    return None


def init_logger(module_name):
    logger = logging.getLogger(module_name)
    level = get_log_level(module_name)
    if level is not None:
        logger.setLevel(level)

    return logger


def setup_logging():
    logging.basicConfig(
        level=get_log_level(),
        format="[%(levelname)s] %(asctime)s - %(name)s: %(message)s",
    )
