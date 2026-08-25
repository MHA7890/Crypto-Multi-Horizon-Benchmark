"""
forecasting.utils.timing — Execution time measurement utilities.
"""

from __future__ import annotations

import time
from types import TracebackType


class Timer:
    """Context manager and stopwatch for measuring execution time in seconds."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
