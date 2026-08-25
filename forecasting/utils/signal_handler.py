"""
forecasting.utils.signal_handler — Graceful Interruption Handler.

Intercepts KeyboardInterrupt (Ctrl+C) and termination signals to save checkpoint,
flush logs, release memory, and display a summary before exiting safely.
"""

from __future__ import annotations

import logging
import signal
import sys
from typing import Any, Callable, Optional

from forecasting.utils.device import clear_gpu_memory

logger = logging.getLogger(__name__)


class GracefulInterruptHandler:
    """
    Context manager and signal handler for graceful shutdown.
    """

    def __init__(
        self,
        checkpoint_saver: Optional[Callable[[], None]] = None,
        checkpoint_mgr: Any = None,
        device_mgr: Any = None,
        on_exit_callback: Optional[Callable[[], None]] = None,
    ):
        self.checkpoint_saver = checkpoint_saver
        if self.checkpoint_saver is None and checkpoint_mgr is not None and hasattr(checkpoint_mgr, "save"):
            self.checkpoint_saver = checkpoint_mgr.save
        self.checkpoint_mgr = checkpoint_mgr
        self.device_mgr = device_mgr
        self.on_exit_callback = on_exit_callback
        self.interrupted = False
        self._old_sigint = None
        self._old_sigterm = None

    def __enter__(self) -> GracefulInterruptHandler:
        self._old_sigint = signal.signal(signal.SIGINT, self._handle_signal)
        self._old_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._old_sigint:
            signal.signal(signal.SIGINT, self._old_sigint)
        if self._old_sigterm:
            signal.signal(signal.SIGTERM, self._old_sigterm)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        if self.interrupted:
            logger.warning("Forced exit requested. Terminating immediately.")
            sys.exit(1)

        self.interrupted = True
        logger.warning("=" * 60)
        logger.warning("INTERRUPTION DETECTED (Signal %d). PERFORMING GRACEFUL SHUTDOWN...", signum)
        logger.warning("=" * 60)

        # 1. Save checkpoint
        if self.checkpoint_saver:
            try:
                self.checkpoint_saver()
                logger.info("Progress checkpoint successfully saved.")
            except Exception as e:
                logger.error("Error saving checkpoint during shutdown: %s", e)

        # 2. Release resources
        try:
            clear_gpu_memory()
            logger.info("GPU and system memory released.")
        except Exception as e:
            logger.error("Error clearing memory during shutdown: %s", e)

        # 3. Custom exit callback
        if self.on_exit_callback:
            try:
                self.on_exit_callback()
            except Exception as e:
                logger.error("Error in exit callback: %s", e)

        # 4. Flush log handlers
        for handler in logging.getLogger().handlers:
            handler.flush()

        logger.warning("Graceful shutdown complete. Resume the experiment using --resume.")
        sys.exit(0)
