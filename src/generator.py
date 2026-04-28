"""
Generator component for RAG pipeline.
Handles text generation using vLLM.
"""

import atexit
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist


def _cleanup_torch_dist():
    """Clean up torch distributed process group on exit."""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


atexit.register(_cleanup_torch_dist)

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from . import config, utils
    from .kv_cache_manager import TieredKVManager
    from .prompt_blocks import SYSTEM_PROMPT, render_blocks_to_prompt

    logger = logging.getLogger("rag_pipeline")
except ImportError:
    import config
    import utils
    from kv_cache_manager import TieredKVManager
    from prompt_blocks import SYSTEM_PROMPT, render_blocks_to_prompt

    logger = utils.setup_logging(logging.INFO)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


def check_gpu_memory() -> tuple:
    """Check GPU availability.

    Returns (can_run, message). If can_run is False, will exit with error.
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available"

    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_properties(0).name

    return True, f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)"


class Generator:
    """Handles text generation using vLLM engine."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        tensor_parallel_size: Optional[int] = None,
        enable_kv_reuse: bool = False,
        batch_size: int = 1,
        quantization: Optional[str] = None,
        enable_tiered_kv: bool = False,
        kv_manager: Optional[TieredKVManager] = None,
    ):
        self.config = config.config
        self.model_name = model_name or self.config.model.llm_model_name
        self.max_model_len = max_model_len or self.config.model.llm_max_model_len
        self.gpu_memory_utilization = (
            gpu_memory_utilization or self.config.model.llm_gpu_memory_utilization
        )
        self.tensor_parallel_size = (
            tensor_parallel_size or self.config.model.llm_tensor_parallel_size
        )
        self.enable_kv_reuse = enable_kv_reuse
        self.batch_size = batch_size
        self.quantization = quantization
        self.enable_tiered_kv = enable_tiered_kv
        self.kv_manager = kv_manager

        logger.info(f"Initializing Generator with model: {self.model_name}")
        logger.info(f"Max model length: {self.max_model_len}")
        logger.info(f"GPU memory utilization: {self.gpu_memory_utilization}")
        logger.info(f"KV cache reuse: {enable_kv_reuse}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Quantization: {quantization}")
        logger.info(f"Tiered KV cache: {enable_tiered_kv}")

        # Check GPU availability before trying to initialize
        can_run, msg = check_gpu_memory()
        logger.info(msg)

        if not can_run:
            logger.error(f"ERROR: {msg}")
            logger.error("CUDA is required for text generation. Exiting.")
            sys.exit(1)

        self.engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the vLLM engine."""
        try:
            from vllm import LLM, SamplingParams

            self.SamplingParams = SamplingParams
        except ImportError:
            logger.error("vLLM not installed. Install with: uv install vllm")
            sys.exit(1)

        try:
            logger.info("Initializing vLLM engine...")
            self.engine = LLM(
                model=self.model_name,
                trust_remote_code=True,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
                dtype="auto",
                enforce_eager=True,
                enable_prefix_caching=self.enable_kv_reuse,
                quantization=self.quantization,
            )
            # Get tokenizer for chat template
            self.tokenizer = self.engine.get_tokenizer()
            logger.info("vLLM engine initialized successfully")
        except RuntimeError as e:
            err = str(e)
            if "out of memory" in err.lower() or "not enough gpu memory" in err.lower():
                logger.error("ERROR: Not enough GPU memory to load model")
                logger.error(f"  Tried to load: {self.model_name}")
                logger.error(
                    f"  With settings: gpu_memory_utilization={self.gpu_memory_utilization}, max_model_len={self.max_model_len}"
                )
                logger.error(
                    "  Try reducing max_model_len or gpu_memory_utilization in config"
                )
                sys.exit(1)
            else:
                logger.error(f"ERROR: vLLM initialization failed: {e}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"ERROR: Unexpected error initializing vLLM: {e}")
            sys.exit(1)

    def generate(
        self,
        prompts: Union[str, List[str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> Tuple[List[str], float]:
        import contextlib
        import io

        if isinstance(prompts, str):
            prompts = [prompts]

        temperature = temperature or self.config.model.temperature
        top_p = top_p or self.config.model.top_p
        max_tokens = max_tokens or self.config.model.max_tokens

        start_time = time.perf_counter()

        sampling_params = self.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )

        with contextlib.redirect_stderr(io.StringIO()):
            outputs = self.engine.generate(prompts, sampling_params, use_tqdm=False)
        results = [output.outputs[0].text for output in outputs]

        elapsed = time.perf_counter() - start_time
        return results, elapsed

    def _get_block_token_count(self, block) -> int:
        """Estimate token count for a prompt block using the active tokenizer."""
        if getattr(block, "token_count", None) is not None:
            return block.token_count
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            return len(self.tokenizer.encode(block.text, add_special_tokens=False))
        return max(1, len(block.text) // 4)

    def prepare_blocks(self, blocks: List[Any]) -> List[Any]:
        """Register cacheable blocks with the tiered cache manager."""
        if not self.enable_tiered_kv or self.kv_manager is None:
            return []

        entries = []
        for block in blocks:
            if not getattr(block, "cacheable", False):
                continue
            token_count = self._get_block_token_count(block)
            block.token_count = token_count
            entries.append(self.kv_manager.prepare_entry(block, token_count))
        return entries

    def generate_with_blocks(
        self,
        blocks: List[Any],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> Tuple[List[str], float]:
        """Generate from cache-aware prompt blocks."""
        self.prepare_blocks(blocks)
        prompt = render_blocks_to_prompt(blocks, tokenizer=self.tokenizer)
        return self.generate(
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )

    def generate_single(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        results, _ = self.generate(
            [prompt],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )
        return results[0]

    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "model_name": self.model_name,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "enable_kv_reuse": self.enable_kv_reuse,
            "batch_size": self.batch_size,
            "quantization": self.quantization,
            "enable_tiered_kv": self.enable_tiered_kv,
            "engine_initialized": self.engine is not None,
        }
        if self.kv_manager is not None:
            info["tiered_kv_cache"] = self.kv_manager.get_stats()
        return info

    def get_tiered_cache_stats(self) -> Dict[str, Any]:
        """Return tiered cache statistics when enabled."""
        if self.kv_manager is None:
            return {}
        return self.kv_manager.get_stats()

    def get_gpu_memory(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0

    def cleanup(self):
        """Clean up distributed process group to avoid NCCL warnings."""
        if dist.is_initialized():
            dist.destroy_process_group()


def format_rag_prompt(
    query: str,
    retrieved_passages: List[str],
    tokenizer=None,
    max_context_length: int = 2048,
) -> str:
    if not retrieved_passages or all(not p for p in retrieved_passages):
        return query

    # Filter out empty passages
    valid_passages = [p for p in retrieved_passages if p]
    if not valid_passages:
        return query

    context = "\n\n".join(valid_passages)

    if tokenizer is not None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return f"""Context:
{context}

Question: {query}

Answer:"""


def test_generator():
    logger.info("Testing Generator component")
    generator = Generator()
    info = generator.get_model_info()
    logger.info(f"Model info: {info}")

    test_prompt = "What is the capital of France?"
    logger.info(f"Testing generation with prompt: {test_prompt}")
    result, gen_time = generator.generate(test_prompt, max_tokens=50)
    logger.info(f"Generation took {gen_time:.4f}s")
    logger.info(f"Result: {result}")

    generator.cleanup()
    logger.info("Generator test completed successfully")


if __name__ == "__main__":
    test_generator()
