import logging
import os

def __env_logging_config():
    try:
        return os.environ['LOG_LEVEL']
    except:
        return '__root__=INFO'

def get_log_level(target_module_name=None):
    if target_module_name is None:
        target_module_name == "__root__"

    config = __env_logging_config()

    for module_config in config.split(","):
        (module_name, log_level) = tuple(module_config.split("="))

        if module_name == target_module_name:
            return log_level

def init_logger(module_name):
    logger = logging.getLogger(module_name)
    logger.setLevel(get_log_level(module_name))
    return logger

def setup_logging():
    logging.basicConfig(
        level=get_log_level(), 
        format="[%(levelname)s] %(asctime)s - %(name)s: %(message)s"
    )

