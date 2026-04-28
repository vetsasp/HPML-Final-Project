"""
Utility functions for the RAG pipeline.
Provides timing, logging, and memory tracking helpers.
"""

import functools
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable

import psutil
import torch


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("rag_pipeline")
    logger.setLevel(level)

    # Avoid adding handlers if they already exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.

    Args:
        func: Function to decorate

    Returns:
        Wrapped function with timing
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        # Log timing if logger is available
        logger = logging.getLogger("rag_pipeline")
        if logger.handlers:  # Only log if logger is configured
            logger.debug(f"{func.__name__} took {elapsed:.4f} seconds")

        # Return both result and timing info
        return result, elapsed

    return wrapper


@contextmanager
def timer(name: str = "Operation"):
    """
    Context manager for timing code blocks.

    Args:
        name: Name of the operation being timed

    Yields:
        None
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        elapsed = end - start
        logger = logging.getLogger("rag_pipeline")
        if logger.handlers:
            logger.debug(f"{name} took {elapsed:.4f} seconds")


def get_memory_usage() -> dict:
    """
    Get current memory usage statistics.

    Returns:
        Dictionary with memory usage info
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    stats = {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        "percent": process.memory_percent(),
    }

    # Add GPU memory if available
    if torch.cuda.is_available():
        stats.update(
            {
                "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            }
        )

    return stats


def log_memory_usage(label: str = ""):
    """
    Log current memory usage.

    Args:
        label: Optional label for the memory usage log
    """
    stats = get_memory_usage()
    logger = logging.getLogger("rag_pipeline")
    if logger.handlers:
        mem_str = f"Memory[RSS: {stats['rss_mb']:.1f}MB, VMS: {stats['vms_mb']:.1f}MB, %: {stats['percent']:.1f}%"
        if "gpu_allocated_mb" in stats:
            mem_str += f", GPU: {stats['gpu_allocated_mb']:.1f}MB allocated"
        mem_str += ")"

        if label:
            logger.debug(f"{label}: {mem_str}")
        else:
            logger.debug(mem_str)


def reset_peak_memory_stats():
    """Reset peak memory statistics (for GPU and CPU if supported)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # Note: psutil doesn't have direct equivalent for CPU peak reset


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 1e-3:
        return f"{seconds*1e6:.2f}µs"
    elif seconds < 1:
        return f"{seconds*1e3:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


class AverageMeter:
    """Computes and stores average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


if __name__ == "__main__":
    # Simple test when run directly
    logger = setup_logging(logging.DEBUG)
    logger.info("Testing utils module")

    # Test timing decorator
    @timing_decorator
    def test_func():
        time.sleep(0.1)
        return "done"

    result, elapsed = test_func()
    logger.info(f"Function result: {result}, time: {elapsed:.4f}s")

    # Test context manager
    with timer("Test operation"):
        time.sleep(0.05)

    # Test memory logging
    log_memory_usage("After test")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10)
    meter.update(20)
    logger.info(f"Average meter: {meter.avg}")

    logger.info("Utils test completed")
