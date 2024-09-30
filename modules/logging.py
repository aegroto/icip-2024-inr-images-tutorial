import logging
import os

def __env_logging_level():
    try:
        return os.environ['LOG_LEVEL']
    except:
        return 'INFO'

def setup_logging():
    logging.basicConfig(
        level=__env_logging_level(), 
        format="[%(levelname)s] %(asctime)s - %(name)s: %(message)s"
    )

