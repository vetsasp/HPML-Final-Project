"""
Generator component for RAG pipeline.
Handles text generation using vLLM.
"""

import logging
import os
import sys
import time
import atexit
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

    logger = logging.getLogger("rag_pipeline")
except ImportError:
    import config
    import utils

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

        logger.info(f"Initializing Generator with model: {self.model_name}")
        logger.info(f"Max model length: {self.max_model_len}")
        logger.info(f"GPU memory utilization: {self.gpu_memory_utilization}")

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
            logger.error("vLLM not installed. Install with: uv pip install vllm")
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
            )
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
        import io
        import contextlib

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
        return {
            "model_name": self.model_name,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "engine_initialized": self.engine is not None,
        }

    def cleanup(self):
        """Clean up distributed process group to avoid NCCL warnings."""
        if dist.is_initialized():
            dist.destroy_process_group()


def format_rag_prompt(
    query: str, retrieved_passages: List[str], max_context_length: int = 2048
) -> str:
    context = "\n\n".join(
        [f"Passage {i+1}: {p}" for i, p in enumerate(retrieved_passages)]
    )
    prompt = f"""Context information:
{context}

User query: {query}

Based on the context information above, please answer the user query.
Answer:"""
    return prompt


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
