import time
import logging
from .agn_logger import logger

def timing(function):
    def wrap(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        duration = (end_time - start_time) / 60.0
        duration_sec = (end_time - start_time) % 60.0
        f_name = function.__name__
        logger.info(f"{f_name} took {int(duration)}min {int(duration_sec)}s")

        return result

    return wrap