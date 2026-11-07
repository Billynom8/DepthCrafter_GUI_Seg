"""Device, dtype and timing helpers so the app runs on CUDA, Apple MPS or CPU.

Set DEPTHCRAFTER_DEVICE to "cuda", "mps" or "cpu" to force a device. The default
is automatic: CUDA if available, then Apple MPS, then CPU.
"""

import os
import time

import torch

_VALID_DEVICES = ("cuda", "mps", "cpu")


def pick_device() -> str:
    """Return the device to run on: "cuda", "mps" or "cpu"."""
    forced = os.environ.get("DEPTHCRAFTER_DEVICE", "").strip().lower()
    if forced in _VALID_DEVICES:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    """float16 on GPU backends. CPU uses float32, because fp16 is slow or unsupported for many CPU ops."""
    return torch.float16 if device in ("cuda", "mps") else torch.float32


def cuda_empty_cache() -> None:
    """Release cached CUDA memory. Does nothing when CUDA is not available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_synchronize() -> None:
    """Wait for queued CUDA work. Does nothing when CUDA is not available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _WallClockEvent:
    """Stand-in for torch.cuda.Event on machines without CUDA."""

    def __init__(self):
        self._time = None

    def record(self):
        self._time = time.perf_counter()

    def elapsed_time(self, end):
        """Elapsed time to `end` in milliseconds, matching torch.cuda.Event.elapsed_time."""
        return (end._time - self._time) * 1000.0


def make_timing_event():
    """Return a timing event with record() and elapsed_time(other) in milliseconds.

    CUDA events on NVIDIA GPUs. Wall-clock timestamps elsewhere, which do not wait for
    the device, so treat those numbers as approximate.
    """
    if torch.cuda.is_available():
        return torch.cuda.Event(enable_timing=True)
    return _WallClockEvent()
