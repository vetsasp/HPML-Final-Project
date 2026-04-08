"""
Pipeline orchestrator for RAG pipeline.
Coordinates embedding, retrieval, and generation components.
"""

import logging
import os
import sys
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np

from dotenv import load_dotenv

# Add the parent directory to sys.path to allow imports when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Try relative imports first (when used as package), fall back to absolute (when run directly)
try:
    from . import config, utils
    from .embedder import Embedder
    from .retriever import Retriever, create_sample_data
    from .generator import Generator, format_rag_prompt

    logger = logging.getLogger("rag_pipeline")
except ImportError:
    import config
    import utils
    from embedder import Embedder
    from retriever import Retriever, create_sample_data
    from generator import Generator, format_rag_prompt

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
    ):
        """
        Initialize the pipeline.

        Args:
            embedder: Embedder component (creates if None)
            retriever: Retriever component (creates if None)
            generator: Generator component (creates if None)
        """
        self.config = config.config
        logger.info("Initializing RAG Pipeline")

        # Initialize components
        logger.info("Initializing Embedder...")
        self.embedder = embedder or Embedder()

        logger.info("Initializing Retriever...")
        self.retriever = retriever or Retriever()

        logger.info("Initializing Generator...")
        self.generator = generator or Generator()

        # Load corpus from file
        self._load_corpus()

        logger.info("Pipeline initialized successfully")

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
        timings = {}

        # Step 1: Embed the query
        start = time.perf_counter()
        query_embedding, embed_time = self.embedder.encode([query])
        timings["embedding"] = embed_time

        # Step 2: Retrieve similar documents
        start = time.perf_counter()
        distances, indices, mapped_ids, search_time = self.retriever.search(
            query_embedding, k=top_k
        )
        timings["retrieval"] = search_time

        # Get the actual passages using the mapped IDs
        flat_ids = [id_val for id_val in mapped_ids[0] if id_val != -1]
        retrieved_passages = self.retriever.get_documents_by_ids(flat_ids)
        retrieved_scores = distances[0].tolist()[: len(retrieved_passages)]

        # Step 3: Generate response
        start = time.perf_counter()

        # Format RAG prompt (uses model's chat template automatically)
        prompt = format_rag_prompt(
            query, retrieved_passages, tokenizer=self.generator.tokenizer
        )

        # Generate
        generated_texts, gen_time = self.generator.generate(
            prompt, max_tokens=max_tokens or self.config.model.max_tokens
        )
        timings["generation"] = gen_time
        timings["total"] = sum(timings.values())

        return RAGResult(
            query=query,
            answer=generated_texts[0] if generated_texts else "",
            retrieved_passages=retrieved_passages,
            retrieved_scores=retrieved_scores,
            timings=timings,
            metadata={
                "top_k": top_k,
                "model": self.config.model.llm_model_name,
            },
        )

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

        # Format RAG prompt with provided passages
        prompt = format_rag_prompt(query, passages[:top_k])

        # Generate
        generated_texts, gen_time = self.generator.generate(
            prompt, max_tokens=max_tokens or self.config.model.max_tokens
        )
        timings["generation"] = gen_time

        return RAGResult(
            query=query,
            answer=generated_texts[0] if generated_texts else "",
            retrieved_passages=passages[:top_k],
            retrieved_scores=[1.0] * top_k,  # Placeholder
            timings=timings,
            metadata={
                "top_k": top_k,
                "model": self.config.model.llm_model_name,
            },
        )

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval index."""
        return self.retriever.get_index_stats()

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
