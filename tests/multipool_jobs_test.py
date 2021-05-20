import unittest

from batch_processing.mutipool_jobs import run_function_with_multiprocess

import logging

logging.getLogger("tests").setLevel(logging.INFO)


class MultiprocessJobTest(unittest.TestCase):
    def test_runner(self):
        num_processes = 5
        run_function_with_multiprocess(
            num_multiprocesses=num_processes,
            target_function=foo,
            kwargs=[dict(i=num) for num in range(num_processes)],
        )
        self.fail()


def foo(i):
    logging.info(f"foo{i}")


if __name__ == "__main__":
    # unittest.main()
    num_processes = 5
    run_function_with_multiprocess(
        num_multiprocesses=num_processes,
        target_function=foo,
        kwargs=[dict(i=num) for num in range(num_processes)],
    )
