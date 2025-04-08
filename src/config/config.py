import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = Path(os.getenv('DATA_DIR', './data'))
CHROMA_DIR = Path(os.getenv('VECTOR_STORE_DIR', './chroma_db'))
UPLOAD_DIR = DATA_DIR / "uploads"

# Document processing settings
DOCUMENT_SETTINGS = {
    "chunk_size": int(os.getenv('CHUNK_SIZE', 512)),
    "use_semantic_chunking": True,
}

# Vector store settings
VECTOR_STORE_SETTINGS = {
    "collection_name": os.getenv('VECTOR_COLLECTION_NAME', 'company_docs'),
    "distance_function": "cosine"
}

# Embedding settings
EMBEDDING_SETTINGS = {
    "model_name": os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
    "dimensions": 1536
}

# Retrieval settings
RETRIEVAL_SETTINGS = {
    "keyword_weight": 0.5,
    "semantic_weight": 0.5,
    "default_top_k": 5,
    "rerank_enabled": False
}

# Agent settings
AGENT_SETTINGS = {
    "model_name": os.getenv('MODEL_NAME', 'o3-mini'),
    "verbose": False,
    "temperature": 0.3,
    "max_tokens": 1000
}

# UI settings
UI_SETTINGS = {
    "page_title": "Company Knowledge Base",
    "page_icon": "📚",
    "max_history_items": 20
}

# File type mappings
FILE_TYPE_MAPPINGS = {
    "pdf": "PDF Document",
    "docx": "Word Document",
    "xlsx": "Excel Spreadsheet",
    "xls": "Excel Spreadsheet",
    "csv": "CSV File",
    "txt": "Text File",
    "md": "Markdown File",
    "json": "JSON File",
    "html": "HTML File"
}

# Supported file extensions
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.xls', '.csv', '.txt', '.md', '.json', '.html']


def ensure_directories():
    """Ensure all required directories exist."""
    DATA_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)