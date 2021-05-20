import logging
import multiprocessing

from ..agn_logger import logger

logger.info("IN MULTIPOOl")


def log_pool_name(function):
    def wrap(*args, **kwargs):
        poolname = multiprocessing.current_process().name
        logger.info(f"{poolname}: running {function.__name__}")
        return function(*args, **kwargs)

    return wrap


def run_function_with_multiprocess(
    num_multiprocesses, target_function, kwargs
):
    assert (
        len(kwargs) == num_multiprocesses,
        f"num kwargs = {len(kwargs)}, proc { num_multiprocesses}",
    )
    processes = []
    target_function = log_pool_name(target_function)
    for i in range(num_multiprocesses):
        p = multiprocessing.Process(
            target=target_function, name=f"Process{i}", kwargs=kwargs[i]
        )
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
