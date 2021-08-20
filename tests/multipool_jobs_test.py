import unittest

from agn_utils.batch_processing.mutipool_jobs import run_function_with_multiprocess

import logging

logging.getLogger("tests").setLevel(logging.INFO)


class MultiprocessJobTest(unittest.TestCase):
    def test_runner(self):
        num_processes = 20
        run_function_with_multiprocess(
            num_multiprocesses=num_processes,
            target_function=foo,
            kwargs=[dict(i=num) for num in range(num_processes)],
        )


def foo(i):
    print(f"foo{i}")


if __name__ == "__main__":
    unittest.main()
