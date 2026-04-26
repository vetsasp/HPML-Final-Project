"""Prompt block helpers for cache-aware RAG prompt construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import List, Optional


SYSTEM_PROMPT = (
    "Use ONLY the information provided in the context to answer. "
    "Respond in your own words. If insufficient, say so."
)


@dataclass
class PromptBlock:
    """A stable, cacheable prompt unit."""

    key: str
    block_type: str
    text: str
    cacheable: bool = True
    token_count: Optional[int] = None


def canonicalize_text(text: str) -> str:
    """Normalize text so semantically identical blocks hash the same."""
    return re.sub(r"\s+", " ", text.strip())


def make_block_key(block_type: str, text: str) -> str:
    """Create a stable key for a prompt block."""
    canonical = canonicalize_text(text)
    digest = hashlib.sha256(f"{block_type}:{canonical}".encode("utf-8")).hexdigest()
    return digest


def build_system_block() -> PromptBlock:
    """Build the standard system prompt block."""
    return PromptBlock(
        key=make_block_key("system", SYSTEM_PROMPT),
        block_type="system",
        text=SYSTEM_PROMPT,
    )


def build_passage_blocks(
    passages: List[str],
    max_context_length: int = 2048,
) -> List[PromptBlock]:
    """Build cacheable passage blocks, truncating the total character budget."""
    blocks = []
    remaining_chars = max_context_length
    for passage in passages:
        if not passage:
            continue
        trimmed = passage.strip()
        if not trimmed:
            continue
        if remaining_chars <= 0:
            break
        passage_text = trimmed[:remaining_chars]
        remaining_chars -= len(passage_text)
        blocks.append(
            PromptBlock(
                key=make_block_key("passage", passage_text),
                block_type="passage",
                text=passage_text,
            )
        )
    return blocks


def build_query_block(query: str) -> PromptBlock:
    """Build the query-specific block."""
    query_text = query.strip()
    return PromptBlock(
        key=make_block_key("query", query_text),
        block_type="query",
        text=query_text,
        cacheable=False,
    )


def build_rag_blocks(
    query: str,
    passages: List[str],
    max_context_length: int = 2048,
) -> List[PromptBlock]:
    """Build the ordered prompt blocks for a RAG query."""
    blocks = [build_system_block()]
    blocks.extend(build_passage_blocks(passages, max_context_length=max_context_length))
    blocks.append(build_query_block(query))
    return blocks


def render_blocks_to_prompt(blocks: List[PromptBlock], tokenizer=None) -> str:
    """Render prompt blocks into the final model prompt string."""
    if not blocks:
        return ""

    system_blocks = [block.text for block in blocks if block.block_type == "system"]
    passage_blocks = [block.text for block in blocks if block.block_type == "passage"]
    query_blocks = [block.text for block in blocks if block.block_type == "query"]

    query = query_blocks[-1] if query_blocks else ""
    if not passage_blocks:
        return query

    system_prompt = system_blocks[-1] if system_blocks else SYSTEM_PROMPT
    context = "\n\n".join(passage_blocks)

    if tokenizer is not None:
        messages = [
            {"role": "system", "content": system_prompt},
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
