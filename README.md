# HPML Project

Aryaman Chakraborty - ac11927@nyu.edu

Dongting Gao - dg4528@nyu.edu

Mansour Ndiaye - mv2330@nyu.edu

Philip Vetsas - pmv264@nyu.edu

Dependencies handled with uv, for simplicity over managing a conda installation.

## Functionality

This project implements a small Retrieval-Augmented Generation (RAG) pipeline.
Given a user query, it:

- embeds the query with a sentence-transformer model
- retrieves relevant passages from a FAISS-backed vector index
- builds a grounded prompt from the retrieved context
- generates a final answer with a language model through vLLM

The app also includes a benchmark mode for comparing the baseline pipeline against the optional optimization paths, including KV-prefix caching, tiered cache management, overlap, and quantization.

## Usage

To run with a prompt:
`uv run python -m src "some prompt"`

To run interactively:
`uv run python -m src`

To run all tests:
`uv run python -m src.test_runner`
