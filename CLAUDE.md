# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Constraints

- **No local testing on work laptop**: Some libraries are not approved by the organization. Testing happens on personal laptop only - feedback loop is long.
- **Precision required**: Due to the long feedback loop, code changes must be accurate and complete the first time.
- **Minimize API costs**: User pays for OpenAI API calls out of pocket. Avoid unnecessary LLM calls in code and testing.

## Project Overview

Agentic RAG system for document search and Q&A using multi-agent reasoning. Processes various document types, performs semantic chunking, and uses hybrid search (BM25 + semantic vectors) for retrieval. CrewAI orchestrates specialized agents for complex queries.

## Commands

```bash
# Process documents into the knowledge base
python src/main.py process path/to/documents

# Run Streamlit UI (opens at http://localhost:8501)
python src/main.py ui

# Test retrieval from command line
python src/main.py test "your query" --top-k 5
```

## Configuration

Set `OPENAI_API_KEY` environment variable or add to `.env` file. See `.env.example` for all configurable settings.

All settings centralized in `src/config/config.py` including:
- Document processing: `DOCUMENT_SETTINGS`
- Vector store: `VECTOR_STORE_SETTINGS`
- Retrieval weights: `RETRIEVAL_SETTINGS`
- Agent config: `AGENT_SETTINGS`

## Architecture

### Module Structure

- `src/agents/` - CrewAI multi-agent system
- `src/document_processing/` - Document parsing and chunking
- `src/retrieval/` - Hybrid search (BM25 + semantic)
- `src/vector_store/` - Chroma DB wrapper
- `src/ui/` - Streamlit interface
- `src/config/` - Centralized configuration
- `src/utils/` - Helper functions

### Core Components

**RAGCrew** (`src/agents/crew.py`): Sequential 4-agent workflow:
1. Query Planner - analyzes intent, breaks down complex questions
2. Information Retriever - searches using `search_documents` tool
3. Information Synthesizer - combines retrieved info into answers
4. Self-Critic - verifies accuracy, suggests improvements

Short keyword queries (<=3 words) bypass agents via fast path.

**HybridRetriever** (`src/retrieval/hybrid.py`): Combines BM25 keyword search with semantic vector search. Auto-detects query type - keywords get 0.7 keyword/0.3 semantic weight, questions get inverse.

**DocumentProcessor** (`src/document_processing/processor.py`): Routes files to appropriate processors:
- PDFs: Custom PyMuPDF-based processor
- Tabular (.csv, .xlsx): Polars processor
- Others: Unstructured.io with Chonkie semantic chunking

**ChromaVectorStore** (`src/vector_store/chroma_store.py`): Persistent Chroma DB with cosine distance. Embedding function injected externally (uses OpenAI `text-embedding-3-small`).

## Key Dependencies

- CrewAI for agent orchestration
- Chroma DB for vector storage
- rank-bm25 for keyword search
- Unstructured.io for document parsing
- Chonkie for semantic chunking
- Polars for tabular data
- Streamlit for UI

## Python Version

Requires Python 3.10-3.12 (`requires-python = "<3.13,>=3.10"`)
