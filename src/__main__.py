"""
Entry point for RAG pipeline.
Provides CLI interface for running queries.
"""

import argparse
import logging

# Add parent directory to path
import os
import sys
from typing import Optional

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from . import config, utils
    from .pipeline import Pipeline, RAGResult
except ImportError:
    import config
    import utils
    from pipeline import Pipeline, RAGResult


def setup_args() -> argparse.ArgumentParser:
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline - Retrieval-Augmented Generation"
    )

    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        help="Query to process (if not provided, enters interactive mode)",
    )

    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        help="Number of passages to retrieve",
    )

    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=None,
        help="Maximum tokens to generate",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--kv",
        action="store_true",
        help="Enable KV cache prefix caching",
    )

    parser.add_argument(
        "--tiered",
        action="store_true",
        help="Enable batch processing (tiered memory)",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show pipeline statistics and exit",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="RAG Pipeline 0.1.0",
    )

    return parser


def print_result(result: RAGResult, verbose: bool = False):
    """Print RAG query result."""
    print("\n" + "=" * 60)
    print(f"Query: {result.query}")
    print("=" * 60)

    print(f"\nAnswer:\n{result.answer}")

    if verbose and result.retrieved_passages:
        print(f"\nRetrieved {len(result.retrieved_passages)} passages:")
        for i, (passage, score) in enumerate(
            zip(result.retrieved_passages, result.retrieved_scores)
        ):
            if passage:  # Only print non-empty passages
                print(f"  [{i + 1}] Score: {score:.4f}")
                print(f"      {passage[:100]}...")

    print(f"\nTimings:")
    for stage, time_val in result.timings.items():
        print(f"  {stage}: {time_val * 1000:.2f}ms")

    print("=" * 60 + "\n")


def interactive_mode(pipeline: Pipeline, args):
    """Run in interactive mode."""
    print("RAG Pipeline Interactive Mode")
    print("Type 'quit' or 'exit' to exit")
    print("Type 'stats' to see index statistics")
    print("Type 'help' to see available commands")
    print()

    while True:
        try:
            query = input("Query> ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                break

            if query.lower() == "stats":
                stats = pipeline.get_index_stats()
                print(f"Index stats: {stats}")
                continue

            if query.lower() == "help":
                print("Available commands:")
                print("  stats - Show index statistics")
                print("  add <text> - Add document to index")
                print("  quit/exit - Exit interactive mode")
                continue

            if query.lower().startswith("add "):
                doc = query[4:]
                pipeline.add_documents([doc])
                print(f"Added document: {doc[:50]}...")
                continue

            if not query:
                continue

            result = pipeline.query(
                query,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
            )
            print_result(result, verbose=args.verbose)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            break


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Parse arguments
    parser = setup_args()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    utils.setup_logging(log_level)

    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    batch_size = 4 if args.tiered else 1
    pipeline = Pipeline(
        enable_kv_reuse=args.kv,
        batch_size=batch_size,
    )

    # Handle stats option
    if args.stats:
        stats = pipeline.get_index_stats()
        print(f"Index statistics: {stats}")
        info = pipeline.get_component_info()
        print(f"\nComponent info:")
        for component, info_dict in info.items():
            print(f"  {component}: {info_dict}")
        return 0

    # Execute query or enter interactive mode
    if args.query:
        result = pipeline.query(
            args.query,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )
        print_result(result, verbose=args.verbose)
        return 0
    else:
        interactive_mode(pipeline, args)
        return 0


if __name__ == "__main__":
    sys.exit(main())
