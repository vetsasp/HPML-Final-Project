"""
Text embedding component for RAG pipeline.
Handles loading sentence transformer model and encoding text.
"""

import logging
import os
import sys
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Add the parent directory to sys.path to allow imports when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Try relative imports first (when used as package), fall back to absolute (when run directly)
try:
    from . import config, utils

    logger = logging.getLogger("rag_pipeline")
except ImportError:
    import config
    import utils

    logger = utils.setup_logging(logging.INFO)

try:
    load_dotenv()
    # Set HF_TOKEN environment variable for sentence-transformers
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
except Exception as e:
    logger.warning(f"No .env file found: {e}")


class Embedder:
    """
    Handles text embedding using sentence-transformers models.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the embedder.

        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on ('cuda', 'cpu', 'mps')
        """
        self.config = config.config
        self.model_name = model_name or self.config.model.embedding_model_name
        self.device = device or self.config.device.device

        logger.info(f"Loading embedding model: {self.model_name}")
        logger.info(f"Using device: {self.device}")

        # Load the model
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # Set max sequence length if specified in config
        if hasattr(self.config.model, "max_seq_length"):
            self.model.max_seq_length = self.config.model.max_seq_length

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # Warm up the model
        self._warmup()

    def _warmup(self):
        """Warm up the model with a dummy input to avoid initial latency spike."""
        try:
            dummy_text = "This is a warmup sentence."
            _ = self.encode([dummy_text])
            logger.debug("Model warmed up successfully")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text string or list of text strings
            batch_size: Batch size for encoding (uses config default if None)
            show_progress_bar: Whether to show progress bar
            normalize_embeddings: Whether to L2-normalize embeddings

        Returns:
            Tuple of (embeddings array, elapsed time in seconds)
            Embeddings array has shape (len(texts), embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.config.retrieval.embedding_batch_size

        logger.debug(f"Encoding {len(texts)} texts with batch size {batch_size}")

        start_time = time.perf_counter()
        # Encode using sentence-transformers
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
        elapsed = time.perf_counter() - start_time

        return embeddings, elapsed

    def encode_single(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        """
        Encode a single text string.

        Args:
            text: Input text string
            normalize_embeddings: Whether to L2-normalize the embedding

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        embeddings, _ = self.encode([text], normalize_embeddings=normalize_embeddings)
        return embeddings[0]

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of the embeddings."""
        return self.embedding_dim

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.embedding_dim,
            "max_seq_length": getattr(self.model, "max_seq_length", None),
        }


def test_embedder():
    """Test function for the embedder component."""
    logger.info("Testing Embedder component")

    # Initialize embedder
    embedder = Embedder()

    # Test single encoding
    test_text = "This is a test sentence for embedding."
    embedding = embedder.encode_single(test_text)
    _, encode_time = embedder.encode([test_text])

    logger.info(f"Single encoding took {encode_time:.4f}s")
    logger.info(f"Embedding shape: {embedding.shape}")
    logger.info(f"Embedding norm: {np.linalg.norm(embedding):.6f}")

    # Test batch encoding
    test_texts = [
        "This is the first test sentence.",
        "This is the second test sentence.",
        "Another test sentence for batch processing.",
        "Yet another example text to encode.",
    ]

    embeddings, encode_time = embedder.encode(test_texts)

    logger.info(f"Batch encoding took {encode_time:.4f}s")
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(
        f"Average embedding norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.6f}"
    )

    # Test model info
    info = embedder.get_model_info()
    logger.info(f"Model info: {info}")

    logger.info("Embedder test completed successfully")


if __name__ == "__main__":
    test_embedder()
