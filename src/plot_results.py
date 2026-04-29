"""
Generates matplotlib charts from benchmark and quality results.
Run after test_runner.py and eval_quality.py have produced their JSON files.

Usage:
    uv run python -m src.plot_results

Outputs (saved to plots/ directory):
    latency_breakdown.png   - Stacked bar: embed/retrieve/generate per config
    speedup.png             - % speedup vs baseline per config
    ttft_comparison.png     - TTFT per config (if vLLM metrics available)
    gpu_memory.png          - GPU memory footprint per config
    rouge_l.png             - ROUGE-L quality scores per config
"""

import json
import os
import sys

PLOTS_DIR = "plots"


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def setup_matplotlib():
    """Use non-interactive backend so it works on headless HPC nodes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save(plt, filename: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_latency_breakdown(plt, results: list):
    """Stacked bar chart: embed + retrieve + generate per config."""
    labels = [r["flags"] for r in results]
    embed = [r["embed_time_ms"] for r in results]
    retrieve = [r["retrieve_time_ms"] for r in results]
    generate = [r["generate_time_ms"] for r in results]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))

    bars_e = ax.bar(x, embed, label="Embedding", color="#4C72B0")
    bars_r = ax.bar(x, retrieve, bottom=embed, label="Retrieval", color="#DD8452")
    bottom_g = [e + r for e, r in zip(embed, retrieve)]
    bars_g = ax.bar(x, generate, bottom=bottom_g, label="Generation", color="#55A868")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("End-to-End Latency Breakdown by Configuration")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save(plt, "latency_breakdown.png")


def plot_speedup(plt, results: list):
    """Bar chart of speedup % vs baseline."""
    baseline = results[0]["total_time_ms"]
    labels = [r["flags"] for r in results]
    speedups = [
        (baseline - r["total_time_ms"]) / baseline * 100 for r in results
    ]

    colors = ["#55A868" if s >= 0 else "#C44E52" for s in speedups]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(x, speedups, color=colors)

    for bar, val in zip(bars, speedups):
        sign = "+" if val >= 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.3 if val >= 0 else -1.5),
            f"{sign}{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Speedup vs Baseline (%)")
    ax.set_title("Latency Speedup per Optimization Configuration")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save(plt, "speedup.png")


def plot_ttft(plt, results: list):
    """Bar chart of TTFT per config — only shown if metrics were captured."""
    ttft_values = [r["ttft_ms"] for r in results]
    if all(v == 0.0 for v in ttft_values):
        print("TTFT values all zero (vLLM metrics not available) — skipping ttft plot.")
        return

    labels = [r["flags"] for r in results]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, ttft_values, color="#8172B2")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time-To-First-Token (TTFT) per Configuration")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save(plt, "ttft_comparison.png")


def plot_gpu_memory(plt, results: list):
    """Bar chart of GPU memory usage per config."""
    labels = [r["flags"] for r in results]
    mem = [r["gpu_memory_gb"] for r in results]

    if all(v == 0.0 for v in mem):
        print("GPU memory all zero (no GPU?) — skipping memory plot.")
        return

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, mem, color="#CCB974")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("GPU Memory Allocated (GB)")
    ax.set_title("GPU Memory Footprint per Configuration")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save(plt, "gpu_memory.png")


def plot_rouge(plt, quality_results: list):
    """Bar chart of ROUGE-L scores per config."""
    labels = [r["config"] for r in quality_results]
    scores = [r["avg_rouge_l"] for r in quality_results]
    baseline = scores[0] if scores else 1.0

    colors = ["#4C72B0" if s >= baseline - 0.01 else "#C44E52" for s in scores]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(x, scores, color=colors)

    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("ROUGE-L F1")
    ax.set_title("Generation Quality (ROUGE-L) per Configuration")
    ax.set_ylim(0, min(1.0, max(scores) * 1.2) if scores else 1.0)
    ax.axhline(baseline, color="grey", linewidth=1, linestyle="--", label="Baseline")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save(plt, "rouge_l.png")


def main():
    bench_path = "benchmark_results.json"
    quality_path = "quality_results.json"

    if not os.path.exists(bench_path):
        print(f"ERROR: {bench_path} not found. Run test_runner.py first.")
        sys.exit(1)

    plt = setup_matplotlib()

    bench = load_json(bench_path)
    results = bench["results"]

    print(f"Plotting results for: {bench.get('model', 'unknown model')} on {bench.get('gpu', 'unknown GPU')}")

    plot_latency_breakdown(plt, results)
    plot_speedup(plt, results)
    plot_ttft(plt, results)
    plot_gpu_memory(plt, results)

    if os.path.exists(quality_path):
        quality = load_json(quality_path)
        plot_rouge(plt, quality)
    else:
        print(f"Note: {quality_path} not found — skipping ROUGE-L plot.")

    print(f"\nAll plots saved to ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()
