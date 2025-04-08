# Agentic RAG Knowledge Base System

An intelligent document search and question answering system that uses advanced retrieval augmented generation (RAG) with agent-based reasoning.

## Features

- **Document Processing:** Process various document types (PDF, Word, Excel, CSV, Text, etc.)
- **Semantic Chunking:** Intelligent document chunking that preserves context using Chonkie
- **Hybrid Search:** Combines keyword (BM25) and semantic vector search for better results
- **Agent-Based Reasoning:** Uses CrewAI to coordinate multiple specialized agents for complex queries
- **Dual Interface:** Supports both keyword search and natural language questions
- **Streamlit UI:** User-friendly interface for document upload and searching

## Architecture

The system consists of the following components:

1. **Document Processing Pipeline**
   - Document parsing with Unstructured.io
   - Semantic chunking with Chonkie
   - Tabular data processing with Polars

2. **Vector Storage**
   - Chroma vector database
   - Metadata indexing and filtering

3. **Retrieval System**
   - BM25 keyword search
   - Semantic vector search
   - Hybrid ranking algorithm

4. **Agentic Layer**
   - Query planning
   - Information retrieval
   - Information synthesis
   - Self-critique and verification

5. **User Interface**
   - Streamlit-based web interface
   - Document upload capabilities
   - Search history and filtering

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mila-rao/agentic_rag_for_docs.git
   cd agentic_rag_for_docs
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   # On Windows: set OPENAI_API_KEY=your-api-key
   ```
   Or add them to a .env file

## Usage

### Processing Documents

To process a directory of documents and add them to the knowledge base:

```bash
python main.py process path/to/your/documents
```

### Running the UI

To start the Streamlit UI:

```bash
python main.py ui
```

Then open your browser to `http://localhost:8501`

### Testing Retrieval

To test the retrieval system directly from the command line:

```bash
python main.py test "your search query" --top-k 5
```

## Example Workflow

1. Start the application UI:
   ```bash
   python main.py ui
   ```

2. Upload documents using the sidebar upload interface

3. Use the search bar to:
   - Enter keywords for simple searches
   - Ask natural language questions about the document content

## Performance Optimizations

This implementation includes several optimizations:

- **Caching:** Document embeddings and retrieval results are cached
- **Async processing:** Background processing for document ingestion
- **Hybrid retrieval:** Combines multiple search methods for better accuracy
- **Query classification:** Automatically routes to the appropriate search strategy

## Future Enhancements

- Reranking with a more sophisticated model
- Integration with web sources
- Multi-user support with permissions
- More sophisticated agent workflows
- Performance optimizations for larger document collections

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Architecture Overview

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 700">
  <!-- Background -->
  <rect width="900" height="700" fill="#f8f9fa" rx="10" ry="10"/>
  
  <!-- Title -->
  <text x="450" y="40" font-family="Arial" font-size="24" font-weight="bold" text-anchor="middle" fill="#333">Agentic RAG System Architecture</text>
  
  <!-- Document Ingestion Pipeline -->
  <rect x="50" y="80" width="240" height="160" rx="5" ry="5" fill="#e1f5fe" stroke="#01579b" stroke-width="2"/>
  <text x="170" y="100" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#01579b">Document Ingestion Pipeline</text>
  <rect x="70" y="120" width="200" height="30" rx="4" ry="4" fill="#b3e5fc" stroke="#0288d1" stroke-width="1"/>
  <text x="170" y="140" font-family="Arial" font-size="12" text-anchor="middle">Unstructured.io Parser</text>
  <rect x="70" y="160" width="200" height="30" rx="4" ry="4" fill="#b3e5fc" stroke="#0288d1" stroke-width="1"/>
  <text x="170" y="180" font-family="Arial" font-size="12" text-anchor="middle">Chonkie Semantic Chunker</text>
  <rect x="70" y="200" width="200" height="30" rx="4" ry="4" fill="#b3e5fc" stroke="#0288d1" stroke-width="1"/>
  <text x="170" y="220" font-family="Arial" font-size="12" text-anchor="middle">Polars Tabular Processor</text>
  
  <!-- Vector Store -->
  <rect x="50" y="280" width="240" height="100" rx="5" ry="5" fill="#e8f5e9" stroke="#2e7d32" stroke-width="2"/>
  <text x="170" y="300" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#2e7d32">Vector Storage</text>
  <rect x="70" y="320" width="200" height="30" rx="4" ry="4" fill="#c8e6c9" stroke="#388e3c" stroke-width="1"/>
  <text x="170" y="340" font-family="Arial" font-size="12" text-anchor="middle">Chroma Vector DB</text>
  <rect x="70" y="360" width="200" height="30" rx="4" ry="4" fill="#c8e6c9" stroke="#388e3c" stroke-width="1"/>
  <text x="170" y="380" font-family="Arial" font-size="12" text-anchor="middle">Metadata Index</text>
  
  <!-- Search & Retrieval -->
  <rect x="330" y="80" width="240" height="160" rx="5" ry="5" fill="#fff3e0" stroke="#e65100" stroke-width="2"/>
  <text x="450" y="100" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#e65100">Search &amp; Retrieval</text>
  <rect x="350" y="120" width="200" height="30" rx="4" ry="4" fill="#ffe0b2" stroke="#f57c00" stroke-width="1"/>
  <text x="450" y="140" font-family="Arial" font-size="12" text-anchor="middle">BM25 Keyword Search</text>
  <rect x="350" y="160" width="200" height="30" rx="4" ry="4" fill="#ffe0b2" stroke="#f57c00" stroke-width="1"/>
  <text x="450" y="180" font-family="Arial" font-size="12" text-anchor="middle">Semantic Vector Search</text>
  <rect x="350" y="200" width="200" height="30" rx="4" ry="4" fill="#ffe0b2" stroke="#f57c00" stroke-width="1"/>
  <text x="450" y="220" font-family="Arial" font-size="12" text-anchor="middle">Hybrid Ranking</text>
  
  <!-- Agentic Layer -->
  <rect x="610" y="80" width="240" height="200" rx="5" ry="5" fill="#e0f7fa" stroke="#006064" stroke-width="2"/>
  <text x="730" y="100" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#006064">Agentic Layer (CrewAI)</text>
  <rect x="630" y="120" width="200" height="30" rx="4" ry="4" fill="#b2ebf2" stroke="#0097a7" stroke-width="1"/>
  <text x="730" y="140" font-family="Arial" font-size="12" text-anchor="middle">Query Planner</text>
  <rect x="630" y="160" width="200" height="30" rx="4" ry="4" fill="#b2ebf2" stroke="#0097a7" stroke-width="1"/>
  <text x="730" y="180" font-family="Arial" font-size="12" text-anchor="middle">Information Synthesizer</text>
  <rect x="630" y="200" width="200" height="30" rx="4" ry="4" fill="#b2ebf2" stroke="#0097a7" stroke-width="1"/>
  <text x="730" y="220" font-family="Arial" font-size="12" text-anchor="middle">Self-Critic</text>
  <rect x="630" y="240" width="200" height="30" rx="4" ry="4" fill="#b2ebf2" stroke="#0097a7" stroke-width="1"/>
  <text x="730" y="260" font-family="Arial" font-size="12" text-anchor="middle">Citation Manager</text>
  
  <!-- LLM Integration -->
  <rect x="330" y="280" width="240" height="100" rx="5" ry="5" fill="#f3e5f5" stroke="#4a148c" stroke-width="2"/>
  <text x="450" y="300" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#4a148c">LLM Integration</text>
  <rect x="350" y="320" width="200" height="30" rx="4" ry="4" fill="#e1bee7" stroke="#6a1b9a" stroke-width="1"/>
  <text x="450" y="340" font-family="Arial" font-size="12" text-anchor="middle">Context Management</text>
  <rect x="350" y="360" width="200" height="30" rx="4" ry="4" fill="#e1bee7" stroke="#6a1b9a" stroke-width="1"/>
  <text x="450" y="380" font-family="Arial" font-size="12" text-anchor="middle">Prompt Engineering</text>
  
  <!-- User Interface -->
  <rect x="330" y="420" width="240" height="140" rx="5" ry="5" fill="#fffde7" stroke="#f57f17" stroke-width="2"/>
  <text x="450" y="440" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#f57f17">User Interface</text>
  <rect x="350" y="460" width="200" height="30" rx="4" ry="4" fill="#fff9c4" stroke="#fbc02d" stroke-width="1"/>
  <text x="450" y="480" font-family="Arial" font-size="12" text-anchor="middle">Streamlit (MVP)</text>
  <rect x="350" y="500" width="200" height="30" rx="4" ry="4" fill="#fff9c4" stroke="#fbc02d" stroke-width="1"/>
  <text x="450" y="520" font-family="Arial" font-size="12" text-anchor="middle">FastAPI + Next.js (Production)</text>
  <rect x="350" y="540" width="200" height="30" rx="4" ry="4" fill="#fff9c4" stroke="#fbc02d" stroke-width="1"/>
  <text x="450" y="560" font-family="Arial" font-size="12" text-anchor="middle">Dual-mode Interface</text>
  
  <!-- Cache & Optimization -->
  <rect x="610" y="320" width="240" height="120" rx="5" ry="5" fill="#ffebee" stroke="#b71c1c" stroke-width="2"/>
  <text x="730" y="340" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#b71c1c">Cache &amp; Optimization</text>
  <rect x="630" y="360" width="200" height="30" rx="4" ry="4" fill="#ffcdd2" stroke="#c62828" stroke-width="1"/>
  <text x="730" y="380" font-family="Arial" font-size="12" text-anchor="middle">Query Cache</text>
  <rect x="630" y="400" width="200" height="30" rx="4" ry="4" fill="#ffcdd2" stroke="#c62828" stroke-width="1"/>
  <text x="730" y="420" font-family="Arial" font-size="12" text-anchor="middle">Async Processing</text>
  
  <!-- MVP Flow -->
  <path d="M170 240 L170 280" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <path d="M290 180 L330 180" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <path d="M570 160 L610 160" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <path d="M450 240 L450 280" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <path d="M730 280 L730 320" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <path d="M450 380 L450 420" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  <path d="M570 330 L610 380" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Arrowhead definition -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
    </marker>
  </defs>
  
  <!-- Data Flow Annotations -->
  <text x="450" y="620" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#333">Implementation Flow</text>
  <rect x="100" y="640" width="700" height="40" rx="5" ry="5" fill="#e3f2fd" stroke="#1565c0" stroke-width="2"/>
  <text x="450" y="665" font-family="Arial" font-size="14" text-anchor="middle" fill="#1565c0">MVP in Python → Profile → Optimize Critical Paths → Refactor for Scale</text>
</svg>