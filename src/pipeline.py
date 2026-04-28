"""
Pipeline orchestrator for RAG pipeline.
Coordinates embedding, retrieval, and generation components.
"""

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from . import config, utils
    from .embedder import Embedder
    from .generator import Generator, format_rag_prompt
    from .kv_cache_manager import TieredKVManager
    from .prompt_blocks import build_rag_blocks
    from .retriever import Retriever, create_sample_data

    logger = logging.getLogger("rag_pipeline")
except ImportError:
    import config
    import utils
    from embedder import Embedder
    from generator import Generator, format_rag_prompt
    from kv_cache_manager import TieredKVManager
    from prompt_blocks import build_rag_blocks
    from retriever import Retriever, create_sample_data

    logger = utils.setup_logging(logging.INFO)


# Load environment variables early
try:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
except Exception:
    pass


@dataclass
class RAGResult:
    """Result of a RAG pipeline execution."""

    query: str
    answer: str
    retrieved_passages: List[str]
    retrieved_scores: List[float]
    timings: Dict[str, float]
    metadata: Dict[str, Any]


class Pipeline:
    """
    Orchestrates the RAG pipeline: embed query → retrieve → generate.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        enable_kv_reuse: bool = False,
        batch_size: int = 1,
        quantization: Optional[str] = None,
        enable_overlap: bool = False,
        enable_tiered_kv: bool = False,
    ):
        """
        Initialize the pipeline.

        Args:
            embedder: Embedder component (creates if None)
            retriever: Retriever component (creates if None)
            generator: Generator component (creates if None)
            enable_kv_reuse: Enable KV cache prefix caching
            batch_size: Number of queries to process in batch
            quantization: Quantization method (e.g., "awq", "gptq", "squeezequant")
            enable_overlap: Enable retrieval-inference overlap
        """
        self.config = config.config
        self.enable_kv_reuse = enable_kv_reuse
        self.batch_size = batch_size
        self.quantization = quantization
        self.enable_overlap = enable_overlap
        self.enable_tiered_kv = enable_tiered_kv
        logger.info("Initializing RAG Pipeline")
        logger.info(f"Enable KV reuse: {enable_kv_reuse}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Quantization: {quantization}")
        logger.info(f"Enable overlap: {enable_overlap}")
        logger.info(f"Enable tiered KV: {enable_tiered_kv}")

        self.kv_manager = TieredKVManager(self.config) if enable_tiered_kv else None

        # Initialize components
        logger.info("Initializing Embedder...")
        self.embedder = embedder or Embedder()

        logger.info("Initializing Retriever...")
        self.retriever = retriever or Retriever()

        logger.info("Initializing Generator...")
        self.generator = generator or Generator(
            enable_kv_reuse=enable_kv_reuse,
            batch_size=batch_size,
            quantization=quantization,
            enable_tiered_kv=enable_tiered_kv,
            kv_manager=self.kv_manager,
        )

        # Load corpus from file
        self._load_corpus()

        logger.info("Pipeline initialized successfully")

    def _prepare_query(
        self,
        query: str,
        top_k: int,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Prepare a query through embedding and retrieval."""
        timings = {}

        if query_embedding is None:
            query_embedding, embed_time = self.embedder.encode([query])
            timings["embedding"] = embed_time
        else:
            timings["embedding"] = 0.0

        distances, indices, mapped_ids, search_time = self.retriever.search(
            query_embedding, k=top_k
        )
        timings["retrieval"] = search_time

        flat_ids = [id_val for id_val in mapped_ids[0] if id_val != -1]
        retrieved_passages = self.retriever.get_documents_by_ids(flat_ids)
        retrieved_scores = distances[0].tolist()[: len(retrieved_passages)]

        return {
            "query": query,
            "retrieved_passages": retrieved_passages,
            "retrieved_scores": retrieved_scores,
            "timings": timings,
        }

    def _generate_prepared_query(
        self,
        prepared: Dict[str, Any],
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate an answer from a prepared query payload."""
        max_tokens = max_tokens or self.config.model.max_tokens

        if self.enable_tiered_kv:
            blocks = build_rag_blocks(
                prepared["query"],
                prepared["retrieved_passages"],
                max_context_length=self.config.model.llm_max_model_len,
            )
            generated_texts, gen_time = self.generator.generate_with_blocks(
                blocks,
                max_tokens=max_tokens,
            )
        else:
            prompt = format_rag_prompt(
                prepared["query"],
                prepared["retrieved_passages"],
                tokenizer=self.generator.tokenizer,
            )
            generated_texts, gen_time = self.generator.generate(
                prompt, max_tokens=max_tokens
            )

        timings = dict(prepared["timings"])
        timings["generation"] = gen_time
        timings["total"] = sum(timings.values())

        return {
            "query": prepared["query"],
            "answer": generated_texts[0] if generated_texts else "",
            "retrieved_passages": prepared["retrieved_passages"],
            "retrieved_scores": prepared["retrieved_scores"],
            "timings": timings,
        }

    def _cache_metadata(self) -> Dict[str, Any]:
        """Return common cache metadata for results."""
        return {
            "tiered_kv_enabled": self.enable_tiered_kv,
            "tiered_kv_cache": (
                self.generator.get_tiered_cache_stats() if self.enable_tiered_kv else {}
            ),
        }

    def _result_from_payload(
        self,
        payload: Dict[str, Any],
        top_k: int,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> RAGResult:
        """Build a RAGResult from generated payload data."""
        metadata = {
            "top_k": top_k,
            "model": self.config.model.llm_model_name,
            **self._cache_metadata(),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return RAGResult(
            query=payload["query"],
            answer=payload["answer"],
            retrieved_passages=payload["retrieved_passages"],
            retrieved_scores=payload["retrieved_scores"],
            timings=payload["timings"],
            metadata=metadata,
        )

    def _load_corpus(self):
        """Load documents from corpus file into the index."""
        corpus_path = self.config.paths.corpus_path

        if not corpus_path.exists():
            logger.warning(f"Corpus file not found: {corpus_path}")
            logger.info("Creating default sample corpus for testing...")
            self._load_default_corpus()
            return

        try:
            import json

            documents = []
            ids = []
            skipped = 0

            with open(corpus_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("text") or obj.get("content") or obj.get("doc")
                        if not text:
                            skipped += 1
                            continue
                        documents.append(text)
                        ids.append(obj.get("id", len(documents) - 1))
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

            if not documents:
                logger.warning(f"No valid documents found in {corpus_path}")
                logger.info("Loading default sample corpus...")
                self._load_default_corpus()
                return

            logger.info(f"Loading {len(documents)} documents from {corpus_path}")
            doc_ids = self.add_documents(documents, ids)
            logger.info(f"Indexed {len(doc_ids)} documents")

            if skipped > 0:
                logger.warning(f"Skipped {skipped} malformed lines in corpus")

        except Exception as e:
            logger.error(f"Failed to load corpus: {e}")
            logger.info("Loading default sample corpus...")
            self._load_default_corpus()

    def _load_default_corpus(self):
        """Load hardcoded default documents for testing."""
        default_docs = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve their performance without being explicitly programmed.",
            "Deep learning uses neural networks with multiple layers to learn complex representations from data, enabling breakthroughs in image and speech recognition.",
            "Natural language processing deals with understanding and generating human language by computers, combining linguistics with machine learning.",
            "The transformer architecture uses self-attention mechanisms to process sequential data efficiently, revolutionizing NLP tasks.",
            "Attention mechanisms allow models to focus on relevant parts of input when making predictions, improving accuracy in sequence tasks.",
            "Retrieval-augmented generation combines retrieval of relevant documents with text generation for more accurate, grounded responses.",
            "Vector databases store embeddings for efficient similarity search, essential for retrieval in RAG systems.",
            "FAISS is a library for efficient similarity search of dense vectors, supporting large-scale vector search operations.",
            "Large language models are neural networks trained on vast text data, capable of generating human-like text and answering questions.",
            "Fine-tuning adapts pre-trained models to specific tasks by continuing training on task-specific data.",
        ]
        ids = list(range(len(default_docs)))
        self.add_documents(default_docs, ids)
        logger.info("Default corpus loaded")

    def add_documents(
        self, documents: List[str], ids: Optional[List[int]] = None
    ) -> List[int]:
        """
        Add documents to the retrieval index.

        Args:
            documents: List of document strings
            ids: Optional list of IDs (generates if None)

        Returns:
            List of assigned IDs
        """
        logger.info(f"Adding {len(documents)} documents to index")

        # Encode documents
        embeddings, embed_time = self.embedder.encode(documents)
        logger.debug(f"Document embedding took {embed_time:.4f}s")

        # Add to index (including document texts)
        doc_ids, _ = self.retriever.add_embeddings(embeddings, ids, documents)

        logger.info(f"Added {len(doc_ids)} documents to index")
        return doc_ids

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> RAGResult:
        """
        Execute a RAG query.

        Args:
            query: User query string
            top_k: Number of passages to retrieve (uses config default if None)
            max_tokens: Maximum tokens to generate (uses config default if None)

        Returns:
            RAGResult with answer, passages, scores, and timings
        """
        top_k = top_k or self.config.retrieval.top_k
        if self.enable_overlap:
            logger.info(
                "Overlap mode only applies to multi-query workloads; using sequential path for single query"
            )

        prepared = self._prepare_query(query, top_k)
        payload = self._generate_prepared_query(prepared, max_tokens=max_tokens)
        return self._result_from_payload(payload, top_k)

    def query_with_passages(
        self,
        query: str,
        passages: List[str],
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> RAGResult:
        """
        Execute a RAG query with pre-provided passages.

        Args:
            query: User query string
            passages: List of pre-retrieved passage strings
            top_k: Number of passages to use (uses all if None)
            max_tokens: Maximum tokens to generate

        Returns:
            RAGResult with answer and timings
        """
        top_k = top_k or len(passages)
        timings = {}

        start = time.perf_counter()

        if self.enable_tiered_kv:
            blocks = build_rag_blocks(
                query,
                passages[:top_k],
                max_context_length=self.config.model.llm_max_model_len,
            )
            generated_texts, gen_time = self.generator.generate_with_blocks(
                blocks, max_tokens=max_tokens or self.config.model.max_tokens
            )
        else:
            prompt = format_rag_prompt(
                query, passages[:top_k], tokenizer=self.generator.tokenizer
            )
            generated_texts, gen_time = self.generator.generate(
                prompt, max_tokens=max_tokens or self.config.model.max_tokens
            )
        timings["generation"] = gen_time

        cache_stats = (
            self.generator.get_tiered_cache_stats() if self.enable_tiered_kv else {}
        )

        return RAGResult(
            query=query,
            answer=generated_texts[0] if generated_texts else "",
            retrieved_passages=passages[:top_k],
            retrieved_scores=[1.0] * top_k,  # Placeholder
            timings=timings,
            metadata={
                "top_k": top_k,
                "model": self.config.model.llm_model_name,
                "tiered_kv_enabled": self.enable_tiered_kv,
                "tiered_kv_cache": cache_stats,
            },
        )

    def get_index_stats(self) -> Dict[str, Any]:
        """Get information about the retrieval index."""
        return self.retriever.get_index_stats()

    def query_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[RAGResult]:
        """
        Execute batch of RAG queries.

        Args:
            queries: List of user query strings
            top_k: Number of passages to retrieve per query

        Returns:
            List of RAGResults
        """
        import concurrent.futures

        import torch

        top_k = top_k or self.config.retrieval.top_k
        batch_start = time.perf_counter()
        query_embeddings, batch_embed_time = self.embedder.encode(queries)

        results = []
        extra_metadata = {
            "gpu_memory_gb": (
                torch.cuda.memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0
            ),
        }

        if not self.enable_overlap:
            for i, query in enumerate(queries):
                prepared = self._prepare_query(
                    query, top_k, query_embeddings=query_embeddings[i : i + 1]
                )
                prepared["timings"]["embedding"] = batch_embed_time / max(
                    len(queries), 1
                )
                payload = self._generate_prepared_query(prepared)
                results.append(
                    self._result_from_payload(
                        payload, top_k, extra_metadata=extra_metadata
                    )
                )
            return results

        if not queries:
            return results

        embed_share = batch_embed_time / len(queries)
        overlap_wall_start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            next_future = executor.submit(
                self._prepare_query,
                queries[0],
                top_k,
                query_embeddings[0:1],
            )

            for i, query in enumerate(queries):
                prepared = next_future.result()
                prepared["timings"]["embedding"] = embed_share

                if i + 1 < len(queries):
                    next_future = executor.submit(
                        self._prepare_query,
                        queries[i + 1],
                        top_k,
                        query_embeddings[i + 1 : i + 2],
                    )
                else:
                    next_future = None

                payload = self._generate_prepared_query(prepared)
                results.append(
                    self._result_from_payload(
                        payload,
                        top_k,
                        extra_metadata={
                            **extra_metadata,
                            "overlap_enabled": True,
                        },
                    )
                )

        overlap_wall_time = time.perf_counter() - overlap_wall_start
        batch_wall_time = time.perf_counter() - batch_start
        per_query_wall = overlap_wall_time / len(queries)
        for result in results:
            result.timings["wall_time"] = per_query_wall
            result.metadata["overlap_wall_time"] = overlap_wall_time
            result.metadata["batch_wall_time"] = batch_wall_time

        return results

    def get_component_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all components."""
        return {
            "embedder": self.embedder.get_model_info(),
            "retriever": self.get_index_stats(),
            "generator": self.generator.get_model_info(),
        }


def test_pipeline():
    """Test function for the pipeline component."""
    logger.info("Testing Pipeline component")

    # Initialize pipeline
    pipeline = Pipeline()

    # Get component info
    info = pipeline.get_component_info()
    logger.info(f"Embedder info: {info['embedder']}")
    logger.info(f"Retriever info: {info['retriever']}")
    logger.info(f"Generator info: {info['generator']}")

    # Add sample documents to index
    logger.info("Adding sample documents...")
    sample_documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn.",
        "Natural language processing deals with understanding human language by computers.",
        "Deep learning uses neural networks with multiple layers to learn representations.",
        "The transformer architecture revolutionized natural language processing tasks.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "BERT is a pre-trained language model that can be fine-tuned for various tasks.",
        "GPT models generate text autoregressively based on given prompts.",
        "Vector databases store embeddings for efficient similarity search.",
        "Retrieval-augmented generation combines retrieval and generation for better answers.",
    ]

    doc_ids = pipeline.add_documents(sample_documents)
    logger.info(f"Added documents with IDs: {doc_ids}")

    # Get index stats
    stats = pipeline.get_index_stats()
    logger.info(f"Index stats: {stats}")

    # Test query
    logger.info("\n--- Testing query ---")
    result = pipeline.query("What is machine learning?", top_k=3)

    logger.info(f"Query: {result.query}")
    logger.info(f"Answer: {result.answer}")
    logger.info(f"Timings: {result.timings}")

    # Test query with pre-provided passages
    logger.info("\n--- Testing query with passages ---")
    test_passages = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to learn representations from data.",
        "Natural language processing deals with understanding and generating human language.",
    ]

    result2 = pipeline.query_with_passages(
        "Explain the relationship between ML and deep learning.",
        test_passages,
        top_k=3,
    )

    logger.info(f"Query: {result2.query}")
    logger.info(f"Answer: {result2.answer}")
    logger.info(f"Timings: {result2.timings}")

    logger.info("Pipeline test completed successfully")


if __name__ == "__main__":
    test_pipeline()
