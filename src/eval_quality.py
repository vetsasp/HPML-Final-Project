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
        torch.cuda.empty_cache()
        import time

        time.sleep(3)

    print_quality_table(eval_results)
    save_quality_results(eval_results)
    logger.info("Quality evaluation complete.")


if __name__ == "__main__":
    main()
