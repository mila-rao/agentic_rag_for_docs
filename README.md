# Agentic RAG Knowledge Base System

An intelligent document search and question answering system that uses advanced retrieval augmented generation (RAG) with agent-based reasoning.

## Features

- **Document Processing:** Process various document types (PDF, Word, Excel, CSV, Text, etc.)
- **Semantic Chunking:** Intelligent document chunking that preserves context using Chonkie (with fallback for Windows)
- **Hybrid Search with RRF:** Combines keyword (BM25) and semantic vector search using Reciprocal Rank Fusion
- **Cross-Encoder Reranking:** Optional second-stage reranking for improved relevance (runs locally, no API cost)
- **Agent-Based Reasoning:** Uses CrewAI to coordinate multiple specialized agents for complex queries
- **Dual Interface:** Supports both keyword search and natural language questions
- **Streamlit UI:** User-friendly interface for document upload, searching, and knowledge base management

## How It Works

### Retrieval Pipeline

```
Query
  │
  ├─→ BM25 Keyword Search ──→ Top 50 by term matching
  │
  └─→ Semantic Search ──────→ Top 50 by embedding similarity
                │
                ▼
        ┌───────────────┐
        │  RRF Fusion   │  Combines rankings: score = 1/(60+rank)
        │  (by rank)    │  Documents in both lists get boosted
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │  Cross-Encoder│  Optional: Re-scores top candidates
        │  Reranker     │  using query-document attention
        └───────┬───────┘
                ▼
           Top K Results
```

**Why RRF?** Traditional score fusion fails because BM25 and embedding scores are on different scales. RRF uses rank positions instead, which are comparable across any retrieval method.

**Why Reranking?** First-stage retrievers (BM25, embeddings) encode queries and documents separately. Cross-encoders see them together, enabling deeper relevance matching at the cost of speed. We use it only on the top candidates.

### Agent Workflow

For complex questions, the system uses a multi-agent pipeline:

1. **Query Planner** - Analyzes intent, extracts key terms
2. **Information Retriever** - Searches the knowledge base using hybrid retrieval
3. **Information Synthesizer** - Combines retrieved information into a coherent answer
4. **Self-Critic** - Verifies accuracy and completeness

Simple keyword queries bypass the agents for faster response.

### Vector Storage

Documents are stored in **Chroma DB**, a persistent vector database:

- **Embeddings:** Generated using OpenAI's embedding models (default: `text-embedding-3-small`)
- **Persistence:** Stored locally in `./chroma_db/` - survives restarts
- **Metadata:** Each chunk stores source file, page number, chunk index for citation
- **Distance metric:** Cosine similarity for semantic matching

The BM25 index is built in-memory from the same documents for keyword search.

```
Document → Chunking → Embeddings → Chroma DB
                          ↓
                    BM25 Index (in-memory)
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mila-rao/agentic_rag_for_docs.git
   cd agentic_rag_for_docs
   ```

2. Install dependencies using PDM:
   ```bash
   pip install pdm
   pdm install
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY
   ```

## Usage

### Process Documents

Ingest documents into the knowledge base:

```bash
pdm run python src/main.py process path/to/documents
```

Supported formats: PDF, DOCX, XLSX, CSV, TXT, MD, JSON, HTML

### Run the UI

```bash
pdm run python src/main.py ui
```

Open http://localhost:8501 in your browser.

### Test Retrieval (CLI)

Test the retrieval system without the UI:

```bash
pdm run python src/main.py test "your search query" --top-k 5
```

This runs hybrid RRF retrieval and displays results with scores.

## Configuration

All settings are in `.env`. Key options:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `OPENAI_CHAT_MODEL` | Chat model for agents | `o3-mini` |
| `OPENAI_TEMPERATURE` | LLM temperature (leave empty for o1/o3) | `0.3` |
| `OPENAI_MAX_TOKENS` | Max response tokens (leave empty for o1/o3) | `4000` |
| `RERANK_ENABLED` | Enable cross-encoder reranking | `false` |
| `RERANK_MODEL` | Reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

> **Note:** Reasoning models (o1, o3) don't support `temperature` or `max_tokens` parameters. Leave these empty in `.env` when using o3-mini or similar models.

See `.env.example` for the complete list.

### Reranker Models

| Model | Speed | Quality |
|-------|-------|---------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast (~50ms) | Good |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Medium | Better |
| `BAAI/bge-reranker-base` | Medium | Better |
| `BAAI/bge-reranker-large` | Slow | Best |

## Architecture

```
src/
├── agents/          # CrewAI multi-agent system
├── config/          # Centralized configuration
├── document_processing/  # Parsing and chunking
├── retrieval/       # Hybrid search, RRF, reranking
├── vector_store/    # Chroma DB wrapper
├── ui/              # Streamlit interface
└── utils/           # Helpers and embeddings
```

## Requirements

- Python 3.10-3.12
- OpenAI API key
- ~2GB disk space for reranker models (if enabled)

## License

MIT
