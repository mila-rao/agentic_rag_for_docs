# Agentic RAG Knowledge Base System

## Project Overview

This project implements an intelligent document search and question-answering system leveraging Retrieval Augmented Generation (RAG) with agent-based reasoning. It processes various document types, performs semantic chunking, and utilizes a hybrid search mechanism combining keyword (BM25) and semantic vector search for enhanced retrieval. The core of the system includes an agentic layer built with CrewAI for complex query planning, information retrieval, synthesis, and self-critique. A user-friendly Streamlit interface allows for document upload, keyword searching, and natural language questioning.

**Key Technologies:**
*   **Document Processing:** Unstructured.io, Chonkie (semantic chunking), Polars (tabular data)
*   **Vector Database:** Chroma DB
*   **Retrieval:** BM25, OpenAI Embeddings (for semantic search), Hybrid Ranking
*   **Agent Framework:** CrewAI
*   **User Interface:** Streamlit
*   **LLM Integration:** OpenAI API

**Architecture Highlights:**
The system is modular, comprising a Document Processing Pipeline, Vector Storage, a Hybrid Retrieval System, an Agentic Layer, and a Streamlit-based User Interface. Configuration is centralized, and environment variables are used for sensitive information like API keys.

## Building and Running

The project is a Python application.

**1. Installation:**

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/mila-rao/agentic_rag_for_docs.git
cd agentic_rag_for_docs
python -m venv venv
source venv/bin/activate # On Windows: .venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```
*(Note: `pyproject.toml` lists direct dependencies, but `requirements.txt` is the canonical way to install them for this project according to the `README.md`)*

**2. Configuration:**

Set your OpenAI API key either as an environment variable or in a `.env` file at the project root:

```bash
export OPENAI_API_KEY="your-api-key"
# Or in a .env file:
# OPENAI_API_KEY="your-api-key"
```

**3. Usage Commands:**

*   **Process Documents:** To ingest documents from a directory into the knowledge base:
    ```bash
    python src/main.py process path/to/your/documents
    ```

*   **Run the Streamlit UI:** To start the web interface for document upload and searching:
    ```bash
    python src/main.py ui
    # Open your browser to http://localhost:8501
    ```

*   **Test Retrieval:** To test the retrieval system directly from the command line:
    ```bash
    python src/main.py test "your search query" --top-k 5
    ```

## Development Conventions

*   **Dependencies:** Project dependencies are managed via `pyproject.toml` for metadata and `requirements.txt` for installation.
*   **Configuration:** All key settings and paths are centralized in `src/config/config.py`, making it easy to manage environment-specific configurations.
*   **Logging:** The `src/main.py` file sets up basic logging to both console and `app.log`.
*   **Modular Design:** The codebase is structured into logical modules (`agents`, `config`, `document_processing`, `retrieval`, `ui`, `utils`, `vector_store`) promoting maintainability and separation of concerns.
*   **Agentic Workflow:** The `src/agents/crew.py` defines a sophisticated, sequential agent workflow using CrewAI, with distinct roles and tasks for query processing, information retrieval, synthesis, and self-critique.
*   **Environment Variables:** Sensitive information (like API keys) and configurable paths/settings are expected to be managed through environment variables or a `.env` file.
