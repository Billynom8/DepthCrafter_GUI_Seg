"""Progress hook for the denoising loop.

The pipeline calls report_denoise_step() after each denoising step. The GUI
installs a listener with set_progress_listener() before it runs a job and
removes it afterwards. Calls come from the worker thread that runs the pipeline.
"""

import logging
from typing import Callable, Optional

_logger = logging.getLogger(__name__)

_listener: Optional[Callable[[int, int], None]] = None


def set_progress_listener(listener: Optional[Callable[[int, int], None]]) -> None:
    """Install a listener(done_steps, total_steps), or None to remove it."""
    global _listener
    _listener = listener


def report_denoise_step(done_steps: int, total_steps: int) -> None:
    listener = _listener
    if listener is None:
        return
    try:
        listener(done_steps, total_steps)
    except Exception as exc:  # a progress display error must never stop the render
        _logger.debug(f"Progress listener failed: {exc}")
