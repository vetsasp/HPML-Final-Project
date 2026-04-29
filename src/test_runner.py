"""
Benchmark runner for RAG pipeline optimization testing.
Runs pipeline with different flag combinations and outputs metrics table.
"""

import logging
import os
import sys
import time
import json
from dataclasses import asdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from . import config, utils
    from .embedder import Embedder
    from .generator import Generator, format_rag_prompt
    from .pipeline import Pipeline, RAGResult
    from .retriever import Retriever
except ImportError:
    import config
    import utils
    from embedder import Embedder
    from generator import Generator, format_rag_prompt
    from pipeline import Pipeline, RAGResult
    from retriever import Retriever


logger = logging.getLogger("rag_benchmark")
logger.setLevel(logging.INFO)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    flags: str
    embed_time_ms: float
    retrieve_time_ms: float
    generate_time_ms: float
    total_time_ms: float
    gpu_memory_gb: float
    speedup_pct: float = 0.0


def save_results_json(
    results: List[BenchmarkResult], output_path: str = "benchmark_results.json"
):
    """Save benchmark results in the plotting script's expected JSON format."""
    payload = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "model": config.config.model.llm_model_name,
        "results": [],
    }

    for result in results:
        row = asdict(result)
        row["ttft_ms"] = 0.0
        row["tpot_ms"] = 0.0
        payload["results"].append(row)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info(f"Saved benchmark results to {output_path}")


def get_test_queries() -> List[str]:
    """Get benchmark queries based on corpus topics."""
    return [
        "How do neural networks learn from data?",
        "What distinguishes deep learning from machine learning?",
        "Explain the transformer architecture role in modern NLP.",
        "How does retrieval-augmented generation improve answers?",
        "What is the purpose of attention mechanisms?",
    ]


def run_benchmark(
    queries: List[str],
    enable_kv_reuse: bool = False,
    enable_tiered_kv: bool = False,
    quantization: Optional[str] = None,
    enable_overlap: bool = False,
    warmup: int = 2,
) -> BenchmarkResult:
    """Run benchmark with specified configuration."""
    flags = []
    if enable_kv_reuse:
        flags.append("kv")
    if enable_tiered_kv:
        flags.append("tiered")
    if quantization:
        flags.append(f"q={quantization}")
    if enable_overlap:
        flags.append("overlap")
    flags_str = "+".join(flags) if flags else "baseline"

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Benchmarking: {flags_str}")
    logger.info(f"{'=' * 60}")

    # Initialize pipeline
    pipeline = Pipeline(
        enable_kv_reuse=enable_kv_reuse,
        batch_size=1,
        quantization=quantization,
        enable_overlap=enable_overlap,
        enable_tiered_kv=enable_tiered_kv,
    )

    # Warmup runs
    logger.info(f"Warming up ({warmup} runs)...")
    for i in range(warmup):
        _ = pipeline.query(queries[i % len(queries)])

    # Synchronize CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark runs
    logger.info(f"Running benchmark ({len(queries)} queries)...")
    run_start = time.perf_counter()
    if enable_overlap:
        results = pipeline.query_batch(queries)
    else:
        results = [pipeline.query(query) for query in queries]
    wall_time_ms = (time.perf_counter() - run_start) * 1000

    embed_times = [result.timings.get("embedding", 0) * 1000 for result in results]
    retrieve_times = [result.timings.get("retrieval", 0) * 1000 for result in results]
    generate_times = [result.timings.get("generation", 0) * 1000 for result in results]
    total_times = [result.timings.get("total", 0) * 1000 for result in results]

    # Get GPU memory
    gpu_mem = 0.0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024**3)

    # Calculate averages
    avg_embed = sum(embed_times) / len(embed_times)
    avg_retrieve = sum(retrieve_times) / len(retrieve_times)
    avg_generate = sum(generate_times) / len(generate_times)
    avg_total = wall_time_ms / len(results)

    logger.info(
        f"Results: embed={avg_embed:.2f}ms, retrieve={avg_retrieve:.2f}ms, gen={avg_generate:.2f}ms, total={avg_total:.2f}ms"
    )
    logger.info(f"GPU memory: {gpu_mem:.2f}GB")

    return BenchmarkResult(
        flags=flags_str,
        embed_time_ms=avg_embed,
        retrieve_time_ms=avg_retrieve,
        generate_time_ms=avg_generate,
        total_time_ms=avg_total,
        gpu_memory_gb=gpu_mem,
    )


def print_table(results: List[BenchmarkResult]):
    """Print results in table format."""
    if results:
        baseline_total = results[0].total_time_ms
        for result in results:
            if baseline_total > 0:
                result.speedup_pct = (
                    (baseline_total - result.total_time_ms) / baseline_total
                ) * 100

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    print(
        f"{'Flags':<20} | {'Embed (ms)':<12} | {'Retrieve (ms)':<14} | {'Generate (ms)':<14} | {'Total (ms)':<12} | {'Speedup %':<10} | {'GPU Mem (GB)':<12}"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r.flags:<20} | "
            f"{r.embed_time_ms:<12.2f} | "
            f"{r.retrieve_time_ms:<14.4f} | "
            f"{r.generate_time_ms:<14.2f} | "
            f"{r.total_time_ms:<12.2f} | "
            f"{r.speedup_pct:<10.2f} | "
            f"{r.gpu_memory_gb:<12.2f}"
        )

    print("=" * 100)


def main():
    """Run all benchmarks."""
    # Setup
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    utils.setup_logging(logging.INFO)
    logger.info("Starting RAG Pipeline Benchmark")
    logger.info(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}"
    )
    logger.info(f"Model: {config.config.model.llm_model_name}")

    queries = get_test_queries()

    results = []

    # Baseline
    results.append(run_benchmark(queries, enable_kv_reuse=False))

    # KV only
    results.append(run_benchmark(queries, enable_kv_reuse=True))

    # Tiered only
    results.append(run_benchmark(queries, enable_tiered_kv=True))

    # KV + tiered
    results.append(run_benchmark(queries, enable_kv_reuse=True, enable_tiered_kv=True))

    # Overlap only (retrieval-inference overlap)
    results.append(run_benchmark(queries, enable_kv_reuse=False, enable_overlap=True))

    # Quantization
    results.append(run_benchmark(queries, enable_kv_reuse=False, quantization="fp8"))

    # KV + quantization
    results.append(run_benchmark(queries, enable_kv_reuse=True, quantization="fp8"))

    # Print table
    print_table(results)
    save_results_json(results)

    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    main()
