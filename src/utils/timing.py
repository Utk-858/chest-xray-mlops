import time
from contextlib import contextmanager
from typing import Generator

@contextmanager
def time_block() -> Generator[dict, None, None]:
    """
    Context manager to measure the execution duration of a block of code.
    Yields a stats dictionary with 'elapsed_ms' that is updated on block exit.
    """
    stats = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield stats
    finally:
        end = time.perf_counter()
        stats["elapsed_ms"] = (end - start) * 1000.0
