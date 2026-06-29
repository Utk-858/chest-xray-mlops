import time
import pytest

from src.utils.timing import time_block

def test_time_block_duration():
    """Verify that time_block records non-zero elapsed durations in milliseconds."""
    with time_block() as stats:
        assert stats["elapsed_ms"] == 0.0
        time.sleep(0.05)  # 50 ms delay

    # Stats should record elapsed time with normal scheduling tolerance
    assert stats["elapsed_ms"] >= 40.0
    assert stats["elapsed_ms"] <= 150.0
