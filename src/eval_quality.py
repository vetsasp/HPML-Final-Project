"""
Quality evaluation for RAG pipeline.
Computes ROUGE-L scores comparing baseline vs. optimized configurations
to verify that performance optimizations do not degrade generation quality.

Usage:
    uv run python -m src.eval_quality
"""

import gc
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List

import torch
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from . import config, utils
    from .pipeline import Pipeline
except ImportError:
    import config
    import utils
    from pipeline import Pipeline

logger = logging.getLogger("rag_eval")


@dataclass
class EvalResult:
    """Quality evaluation result for one pipeline configuration."""

    config: str
    avg_rouge_l: float
    per_query_scores: List[float]


def normalize_text(text: str) -> List[str]:
    """Normalize text into lowercase word tokens for ROUGE-L."""
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    return re.findall(r"\w+", cleaned)


def lcs_length(a: List[str], b: List[str]) -> int:
    """Compute longest common subsequence length."""
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[j]))
        prev = curr
    return prev[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    """Compute ROUGE-L F1 from candidate/reference strings."""
    cand_tokens = normalize_text(candidate)
    ref_tokens = normalize_text(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def evaluate_config(
    pipeline: Pipeline, eval_pairs: List[Dict[str, str]], config_name: str
) -> EvalResult:
    """Run one pipeline configuration against the fixed evaluation queries."""
    scores = []
    for pair in eval_pairs:
        result = pipeline.query(pair["query"])
        score = rouge_l_f1(result.answer, pair["reference"])
        scores.append(score)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return EvalResult(
        config=config_name,
        avg_rouge_l=avg_score,
        per_query_scores=scores,
    )


def print_quality_table(results: List[EvalResult]) -> None:
    """Print a compact ROUGE-L comparison table."""
    print("\n" + "=" * 72)
    print("QUALITY EVALUATION (ROUGE-L)")
    print("=" * 72)
    print(f"{'Config':<20} | {'Avg ROUGE-L':<12} | {'Per-query scores'}")
    print("-" * 72)
    for result in results:
        per_query = ", ".join(f"{score:.3f}" for score in result.per_query_scores)
        print(f"{result.config:<20} | {result.avg_rouge_l:<12.4f} | {per_query}")
    print("=" * 72)


def save_quality_results(
    results: List[EvalResult], output_path: str = "quality_results.json"
) -> None:
    """Save quality results in the plotting script's expected JSON format."""
    payload = [
        {
            "config": result.config,
            "avg_rouge_l": result.avg_rouge_l,
            "per_query_scores": result.per_query_scores,
        }
        for result in results
    ]
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info(f"Saved quality results to {output_path}")


# Reference answers written for the 5 benchmark queries.
# These are ground-truth summaries used to score ROUGE-L.
EVAL_PAIRS: List[Dict[str, str]] = [
    {
        "query": "How do neural networks learn from data?",
        "reference": (
            "Neural networks learn from data through a process called backpropagation "
            "combined with gradient descent. During training, the network makes predictions, "
            "calculates the error using a loss function, and propagates that error backward "
            "through the network to update weights, gradually minimizing the loss."
        ),
    },
    {
        "query": "What distinguishes deep learning from machine learning?",
        "reference": (
            "Deep learning is a subset of machine learning that uses neural networks with "
            "many layers to automatically learn hierarchical feature representations from raw "
            "data. Traditional machine learning relies on hand-crafted features and shallower "
            "models, while deep learning can discover complex patterns directly from data."
        ),
    },
    {
        "query": "Explain the transformer architecture role in modern NLP.",
        "reference": (
            "The transformer architecture revolutionized NLP by replacing recurrent networks "
            "with self-attention mechanisms that process all tokens in parallel. This enables "
            "efficient capture of long-range dependencies and scales well on GPUs. Models like "
            "BERT and GPT, which power modern NLP systems, are built on the transformer."
        ),
    },
    {
        "query": "How does retrieval-augmented generation improve answers?",
        "reference": (
            "Retrieval-augmented generation improves answers by combining a retrieval system "
            "with a language model. Relevant documents are fetched from a knowledge base and "
            "provided as context to the LLM, grounding its output in factual information and "
            "reducing hallucinations compared to generation from parametric memory alone."
        ),
    },
    {
        "query": "What is the purpose of attention mechanisms?",
        "reference": (
            "Attention mechanisms allow a model to dynamically focus on the most relevant "
            "parts of the input when producing each output token. Rather than compressing the "
            "entire context into a fixed vector, attention computes a weighted sum over all "
            "positions, giving the model flexible access to any part of the input sequence."
        ),
    },
]


def main():

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    utils.setup_logging(logging.INFO)
    logger.info("Starting quality evaluation")

    eval_results: List[EvalResult] = []

    # Skip FP8 configs — they need H100 hardware and hog memory
    configs = [
        ("baseline", dict()),
        ("kv", dict(enable_kv_reuse=True)),
        ("tiered", dict(enable_tiered_kv=True)),
        ("kv+tiered", dict(enable_kv_reuse=True, enable_tiered_kv=True)),
        ("overlap", dict(enable_overlap=True)),
    ]

    for name, kwargs in configs:
        logger.info(f"\nEvaluating config: {name}")
        pipeline = Pipeline(**kwargs)
        result = evaluate_config(pipeline, EVAL_PAIRS, name)
        eval_results.append(result)
        # Explicitly free GPU memory before next pipeline
        pipeline.generator.engine = None
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import time

        time.sleep(3)

    print_quality_table(eval_results)
    save_quality_results(eval_results)
    logger.info("Quality evaluation complete.")


if __name__ == "__main__":
    main()
