"""
Utility functions for the RAG pipeline.
Provides timing, logging, and memory tracking helpers.
"""

import functools
import importlib
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable


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
    import psutil

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    stats = {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        "percent": process.memory_percent(),
    }

    # Add GPU memory if available
    import torch

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
    import torch

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
        return f"{seconds * 1e6:.2f}µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f}ms"
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


def check_dependencies() -> int:
    """Check whether the main runtime dependencies are importable."""
    dependencies = [
        ("dotenv", "python-dotenv"),
        ("numpy", "numpy"),
        ("psutil", "psutil"),
        ("torch", "torch"),
        ("sentence_transformers", "sentence-transformers"),
        ("faiss", "faiss-cpu/faiss-gpu"),
        ("vllm", "vllm"),
    ]

    print("Dependency Check")
    print("=" * 60)

    missing = []
    for module_name, package_name in dependencies:
        try:
            module = importlib.import_module(module_name)
            details = ""
            if module_name == "torch":
                cuda_available = getattr(module.cuda, "is_available", lambda: False)()
                details = f" (version={module.__version__}, cuda={cuda_available})"
            elif hasattr(module, "__version__"):
                details = f" (version={module.__version__})"
            print(f"OK       {package_name}{details}")
        except Exception as exc:
            missing.append(package_name)
            print(f"MISSING  {package_name} ({exc})")

    print("=" * 60)
    if missing:
        print("Missing dependencies detected:")
        for package_name in missing:
            print(f"- {package_name}")
        return 1

    print("All checked dependencies imported successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(check_dependencies())
