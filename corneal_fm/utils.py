"""Utility functions for Corneal-FM: timer, logging, device detection, parameter counting."""

from __future__ import annotations

import collections.abc
import functools
import logging
import time
from typing import Any, TypeVar, cast

import torch

F = TypeVar("F", bound=collections.abc.Callable[..., Any])

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with consistent formatting.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    level : int
        Logging level, by default ``logging.INFO``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def get_device() -> Any:
    """Auto-detect the best available compute device.

    Returns
    -------
    torch.device
        CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")  # type: ignore[no-any-unimported]
    return torch.device("cpu")  # type: ignore[no-any-unimported]


def check_gpu_memory() -> str:
    """Return a human-readable GPU memory usage string.

    Uses ``torch.cuda.mem_get_info()`` when available (PyTorch ≥1.11),
    falling back to ``torch.cuda.memory_stats()``.

    Returns
    -------
    str
        Formatted memory string (e.g. ``"Allocated: 2.34 GiB / 23.70 GiB"``)
        or ``"No GPU detected"``.
    """
    if not torch.cuda.is_available():
        return "No GPU detected"

    device_count = torch.cuda.device_count()
    lines: list[str] = []
    for i in range(device_count):
        try:
            free, total = torch.cuda.mem_get_info(i)
            allocated = total - free
            lines.append(
                f"GPU {i}: Allocated {_format_bytes(allocated)} / "
                f"{_format_bytes(total)} "
                f"({allocated / total * 100:.1f}%)"
            )
        except (RuntimeError, AttributeError):
            stats = torch.cuda.memory_stats(i) or {}  # type: ignore[arg-type]
            allocated_bytes = stats.get("allocated_bytes.all.current", 0)
            total_mem = torch.cuda.get_device_properties(i).total_memory  # type: ignore[arg-type]
            lines.append(
                f"GPU {i}: Allocated {_format_bytes(allocated_bytes)} / "
                f"{_format_bytes(total_mem)} "
                f"({allocated_bytes / total_mem * 100:.1f}%)"
            )
    return "\n".join(lines) if lines else "No GPU detected"


def _format_bytes(size: float) -> str:
    """Format byte count as a human-readable string with binary prefixes."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(size) < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.

    Returns
    -------
    int
        Number of parameters with ``requires_grad == True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------


def timer(func: F) -> F:
    """Decorator that logs the execution duration of the wrapped function.

    The elapsed time is recorded via ``logging.getLogger`` under the
    function's module name.

    Usage
    -----
    >>> @timer
    ... def train_epoch(...):
    ...     ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2f s", func.__qualname__, elapsed)
        return result

    return cast(F, wrapper)
