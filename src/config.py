"""
Configuration management for RAG pipeline.
Handles model selection, paths, and device configuration.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# Set up local model cache directories
_models_dir = Path(__file__).parent.parent / "models"
os.environ.setdefault("HF_HOME", str(_models_dir / "hf_cache"))
os.environ.setdefault("VLLM_HOME", str(_models_dir / "vllm"))


@dataclass
class ModelConfig:
    """Model configuration settings."""

    # Embedding model (sentence-transformers)
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    max_seq_length: int = 256

    # Language model selection
    # NOTE: Using a smaller model for testing on local machine
    # llm_model_name: str = "mistralai/Mistral-7B-v0.1"
    llm_model_name: str = "Qwen/Qwen2-0.5B-Instruct"

    llm_max_model_len: int = 2048
    llm_tensor_parallel_size: int = 1
    llm_gpu_memory_utilization: float = 0.8

    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stop_token_ids: Optional[list] = None


@dataclass
class RetrievalConfig:
    """Retrieval configuration settings."""

    top_k: int = 5
    faiss_index_type: str = (
        "IndexFlatIP"  # Inner product for cosine similarity after L2 norm
    )
    embedding_batch_size: int = 32


@dataclass
class PathsConfig:
    """File path configuration."""

    # Base directories
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = base_dir / "models"
    cache_dir: Path = base_dir / "cache"

    # Data files
    corpus_path: Path = data_dir / "corpus.jsonl"
    queries_path: Path = data_dir / "queries.jsonl"
    qrels_path: Path = data_dir / "qrels.jsonl"

    # Cache files
    embedding_cache_path: Path = cache_dir / "embeddings.npy"
    faiss_index_path: Path = cache_dir / "faiss_index.idx"


@dataclass
class DeviceConfig:
    """Device configuration settings."""

    @staticmethod
    def get_device() -> str:
        """Auto-detect and return the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    device: str = get_device()

    @staticmethod
    def get_torch_device() -> torch.device:
        """Get torch.device object."""
        return torch.device(DeviceConfig.get_device())


class Config:
    """Main configuration container."""

    def __init__(self):
        self.model = ModelConfig()
        self.retrieval = RetrievalConfig()
        self.paths = PathsConfig()
        self.device = DeviceConfig()

        # Create directories if they don't exist
        self.paths.data_dir.mkdir(exist_ok=True)
        self.paths.models_dir.mkdir(exist_ok=True)
        self.paths.cache_dir.mkdir(exist_ok=True)


# Global config instance
config = Config()

if __name__ == "__main__":
    # Simple test when run directly
    print("Config Test:")
    print(f"  Device: {config.device.device}")
    print(f"  Embedding model: {config.model.embedding_model_name}")
    print(f"  LLM model: {config.model.llm_model_name}")
    print(f"  Data dir: {config.paths.data_dir}")
    print(f"  Models dir: {config.paths.models_dir}")
    print(f"  Cache dir: {config.paths.cache_dir}")
