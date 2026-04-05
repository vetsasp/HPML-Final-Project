"""
Retrieval component for RAG pipeline.
Handles FAISS-based vector similarity search.
"""

import logging
import os
import sys
import time
from typing import List, Optional, Tuple, Union

import faiss
import numpy as np
import torch

# Add the parent directory to sys.path to allow imports when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Try relative imports first (when used as package), fall back to absolute (when run directly)
try:
    from . import config, embedder, utils

    logger = logging.getLogger("rag_pipeline")
except ImportError:
    import config
    import embedder
    import utils

    logger = utils.setup_logging(logging.INFO)


class Retriever:
    """
    Handles vector similarity search using FAISS.
    """

    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        index_type: str = "IndexFlatIP",
        device: Optional[str] = None,
    ):
        """
        Initialize the retriever.

        Args:
            embedding_dim: Dimensionality of embeddings
            index_type: Type of FAISS index to use
            device: Device for FAISS operations ('cpu' or GPU index)
        """
        self.config = config.config
        self.embedding_dim = embedding_dim or self.config.model.embedding_dimension
        self.index_type = index_type or self.config.retrieval.faiss_index_type
        self.device = device or self.config.device.device

        logger.info(f"Initializing FAISS index: {self.index_type}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        logger.info(f"Device: {self.device}")

        # Initialize FAISS index
        self.index = self._create_index(self.index_type, self.embedding_dim)

        # Move to GPU if specified and available
        if self.device.startswith("cuda") and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Moved FAISS index to GPU")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}. Using CPU index.")

        # Storage for IDs and metadata
        self.id_map = {}  # Maps FAISS internal IDs to actual IDs
        self.documents = {}  # Maps actual IDs to document texts
        self.next_id = 0
        self.is_trained = self._needs_training()

        logger.info(f"Retriever initialized. Index is trained: {self.is_trained}")

    def _create_index(self, index_type: str, dim: int) -> faiss.Index:
        """
        Create a FAISS index based on the specified type.

        Args:
            index_type: Type of FAISS index
            dim: Dimensionality of vectors

        Returns:
            Initialized FAISS index
        """
        if index_type == "IndexFlatIP":
            # Inner product (equivalent to cosine similarity for L2-normalized vectors)
            index = faiss.IndexFlatIP(dim)
        elif index_type == "IndexFlatL2":
            # L2 distance
            index = faiss.IndexFlatL2(dim)
        elif index_type == "IndexIVFFlat":
            # Inverted file with flat quantization
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, min(100, dim))  # nlist parameter
        elif index_type == "IndexHNSWFlat":
            # Hierarchical Navigable Small World
            index = faiss.IndexHNSWFlat(dim, 32)  # M=32
        else:
            logger.warning(
                f"Unknown index type {index_type}, falling back to IndexFlatIP"
            )
            index = faiss.IndexFlatIP(dim)

        return index

    def _needs_training(self) -> bool:
        """Check if the index type requires training."""
        # Index types that need training
        train_types = {"IndexIVFFlat", "IndexIVFPQ", "IndexIVFSQ"}
        return any(t in self.index_type for t in train_types)

    @utils.timing_decorator
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        ids: Optional[List[int]] = None,
        documents: Optional[List[str]] = None,
    ) -> List[int]:
        """
        Add embeddings to the FAISS index.

        Args:
            embeddings: Numpy array of shape (n, embedding_dim)
            ids: Optional list of IDs for the embeddings
            documents: Optional list of document texts to store

        Returns:
            List of IDs assigned to the embeddings
        """
        if embeddings.size == 0:
            logger.warning("Attempted to add empty embeddings array")
            return []

        # Ensure embeddings are float32 (FAISS requirement)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Train index if needed
        if not self.is_trained and self.index.is_trained == False:
            logger.info(f"Training index with {len(embeddings)} vectors")
            self.index.train(embeddings)
            self.is_trained = True
            logger.info("Index training completed")

        # Assign IDs if not provided
        if ids is None:
            ids = list(range(self.next_id, self.next_id + len(embeddings)))
            self.next_id += len(embeddings)

        # Store ID mapping and documents
        start_idx = self.index.ntotal
        for i, id_val in enumerate(ids):
            self.id_map[start_idx + i] = id_val
            if documents is not None and i < len(documents):
                self.documents[id_val] = documents[i]

        # Add to index
        logger.debug(f"Adding {len(embeddings)} vectors to index")
        self.index.add(embeddings)

        logger.debug(f"Index now contains {self.index.ntotal} vectors")
        return ids

    def get_documents_by_ids(self, doc_ids: List[int]) -> List[str]:
        """Retrieve document texts by their IDs."""
        return [self.documents.get(id_val, "") for id_val in doc_ids]

    def search(
        self, query_embeddings: np.ndarray, k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[List[int]], float]:
        """
        Search for similar vectors in the index.

        Args:
            query_embeddings: Numpy array of shape (n_queries, embedding_dim)
            k: Number of nearest neighbors to return (uses config default if None)

        Returns:
            Tuple of (distances, indices, mapped_ids, elapsed_time)
            - distances: Similarity scores (higher is better for inner product)
            - indices: FAISS internal indices
            - mapped_ids: Actual IDs corresponding to results
            - elapsed_time: Time taken in seconds
        """
        if self.index.ntotal == 0:
            logger.warning("Attempted to search empty index")
            empty_shape = (len(query_embeddings), k or self.config.retrieval.top_k)
            return (
                np.empty(empty_shape),
                np.empty(empty_shape, dtype=int),
                [[] for _ in range(len(query_embeddings))],
                0.0,
            )

        k = k or self.config.retrieval.top_k

        # Ensure query embeddings are float32
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)

        logger.debug(
            f"Searching for {k} nearest neighbors among {self.index.ntotal} vectors"
        )

        start_time = time.perf_counter()
        # Perform search
        distances, indices = self.index.search(query_embeddings, k)
        elapsed = time.perf_counter() - start_time

        # Map FAISS indices to actual IDs
        mapped_ids = []
        for i in range(len(indices)):
            id_list = []
            for idx in indices[i]:
                if idx == -1:  # FAISS returns -1 for empty slots when index.ntotal < k
                    id_list.append(-1)
                else:
                    id_list.append(self.id_map.get(idx, -1))
            mapped_ids.append(id_list)

        return distances, indices, mapped_ids, elapsed

    def search_single(
        self, query_embedding: np.ndarray, k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Search for a single query vector.

        Args:
            query_embedding: Single query vector of shape (embedding_dim,)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices, mapped_ids) for single query
        """
        # Reshape to 2D array for search method
        query_2d = query_embedding.reshape(1, -1)
        distances, indices, id_lists, _ = self.search(query_2d, k)

        # Return first (and only) row
        return distances[0], indices[0], id_lists[0]

    def remove_ids(self, ids_to_remove: List[int]) -> bool:
        """
        Remove specific IDs from the index.
        Note: This is inefficient for most index types as it requires rebuilding.

        Args:
            ids_to_remove: List of IDs to remove

        Returns:
            True if removal was attempted
        """
        logger.warning(
            "Removing specific IDs requires index rebuild - not implemented for efficiency"
        )
        # For production systems, you would either:
        # 1. Use an index that supports removal (like IndexIDMap)
        # 2. Mark items as deleted and filter during search
        # 3. Periodically rebuild the index
        return False

    def get_index_stats(self) -> dict:
        """Get statistics about the current index."""
        return {
            "total_vectors": self.index.ntotal,
            "is_trained": self.index.is_trained,
            "embedding_dimension": self.index.d,
            "index_type": type(self.index).__name__,
            "device": (
                "GPU" if hasattr(self.index, "on_gpu") and self.index.on_gpu else "CPU"
            ),
        }

    def reset_index(self):
        """Reset the index to empty state."""
        logger.info("Resetting FAISS index")
        self.index = self._create_index(self.index_type, self.embedding_dim)

        # Move to GPU if specified and available
        if self.device.startswith("cuda") and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Moved FAISS index to GPU")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}. Using CPU index.")

        self.id_map = {}
        self.next_id = 0
        self.is_trained = self._needs_training()


def create_sample_data(
    num_vectors: int = 100, dim: int = 384
) -> Tuple[np.ndarray, List[int]]:
    """
    Create sample data for testing.

    Args:
        num_vectors: Number of vectors to generate
        dim: Dimensionality of vectors

    Returns:
        Tuple of (embeddings, ids)
    """
    # Generate random vectors
    embeddings = np.random.randn(num_vectors, dim).astype(np.float32)

    # Normalize for inner product similarity (cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Generate sequential IDs
    ids = list(range(num_vectors))

    return embeddings, ids


def test_retriever():
    """Test function for the retriever component."""
    logger.info("Testing Retriever component")

    # Initialize retriever
    retriever = Retriever()

    # Create sample data
    logger.info("Creating sample data...")
    embeddings, ids = create_sample_data(num_vectors=50, dim=384)

    # Add embeddings to index
    logger.info("Adding embeddings to index...")
    added_ids, _ = retriever.add_embeddings(embeddings, ids)
    logger.info(f"Added {len(added_ids)} vectors to index")

    # Check index stats
    stats = retriever.get_index_stats()
    logger.info(f"Index stats: {stats}")

    # Create query vectors
    logger.info("Creating query vectors...")
    query_embeddings, query_ids = create_sample_data(num_vectors=3, dim=384)

    # Search for similar vectors
    logger.info("Performing search...")
    distances, indices, mapped_ids, search_time = retriever.search(query_embeddings, k=5)

    logger.info(f"Query shape: {query_embeddings.shape}")
    logger.info(f"Distances shape: {distances.shape}")
    logger.info(f"Indices shape: {indices.shape}")
    logger.info(f"Mapped IDs shape: {[len(ids) for ids in mapped_ids]}")
    logger.info(f"Search time: {search_time:.4f}s")

    # Display results
    for i in range(len(query_embeddings)):
        logger.info(f"Query {i} (ID {query_ids[i]}):")
        for j, (dist, idx, map_id) in enumerate(
            zip(distances[i], indices[i], mapped_ids[i])
        ):
            if map_id != -1:  # Valid result
                logger.info(
                    f"  {j+1}. ID: {map_id}, Distance: {dist:.4f}, FAISS Index: {idx}"
                )
            else:
                logger.info(f"  {j+1}. No result (padding)")

    # Test single search
    logger.info("Testing single query search...")
    single_query = query_embeddings[0]
    dists, inds, ids_list = retriever.search_single(single_query, k=3)
    logger.info(f"Single search results:")
    for j, (dist, idx, map_id) in enumerate(zip(dists, inds, ids_list)):
        if map_id != -1:
            logger.info(
                f"  {j+1}. ID: {map_id}, Distance: {dist:.4f}, FAISS Index: {idx}"
            )

    logger.info("Retriever test completed successfully")


if __name__ == "__main__":
    test_retriever()

