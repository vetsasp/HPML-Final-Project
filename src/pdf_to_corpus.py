"""Convert a PDF into a JSONL corpus for the RAG pipeline."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List

from pypdf import PdfReader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from . import config
except ImportError:
    import config


def normalize_text(text: str) -> str:
    """Normalize extracted PDF text into readable paragraphs."""
    text = text.replace("\r", "\n").replace("\t", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]

    paragraphs = []
    current = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue

        # Keep obvious section headers and bullets on their own lines.
        if current and (
            re.match(r"^(Table|Figure|Appendix|Chapter|Section)\b", line)
            or line.startswith(("-", "*", "•"))
        ):
            paragraphs.append(" ".join(current))
            current = [line]
            continue

        current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    normalized = "\n\n".join(paragraphs)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not paragraphs:
        return []

    chunks = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > chunk_size:
            chunks.append(current)
            if chunk_overlap > 0:
                overlap_text = current[-chunk_overlap:]
                current = f"{overlap_text}\n\n{paragraph}".strip()
            else:
                current = paragraph
            if len(current) > chunk_size:
                start = 0
                step = chunk_size - chunk_overlap
                while start < len(current):
                    piece = current[start : start + chunk_size].strip()
                    if piece:
                        chunks.append(piece)
                    start += step
                current = ""
        else:
            current = candidate

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk]


def pdf_to_corpus(
    pdf_path: Path,
    output_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    start_page: int,
    end_page: int | None,
) -> int:
    """Extract text from a PDF and write repo-compatible JSONL chunks."""
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    if start_page < 1 or start_page > total_pages:
        raise ValueError(f"start_page must be between 1 and {total_pages}")
    if end_page is None:
        end_page = total_pages
    if end_page < start_page or end_page > total_pages:
        raise ValueError(f"end_page must be between {start_page} and {total_pages}")

    page_documents = []
    for page_number in range(start_page, end_page + 1):
        text = reader.pages[page_number - 1].extract_text() or ""
        normalized = normalize_text(text)
        if not normalized:
            continue
        for chunk_index, chunk in enumerate(
            chunk_text(normalized, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            page_documents.append(
                {
                    "id": len(page_documents) + 1,
                    "text": chunk,
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_index,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for doc in page_documents:
            handle.write(json.dumps(doc, ensure_ascii=True) + "\n")

    return len(page_documents)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a PDF into corpus.jsonl")
    parser.add_argument("pdf_path", type=Path, help="Input PDF path")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.config.paths.corpus_path,
        help="Output JSONL path (default: data/corpus.jsonl)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Target chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Character overlap between chunks",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="1-indexed start page",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="1-indexed end page (default: last page)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    num_docs = pdf_to_corpus(
        pdf_path=args.pdf_path,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        start_page=args.start_page,
        end_page=args.end_page,
    )
    print(f"Wrote {num_docs} chunks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
