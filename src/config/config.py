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

# =============================================================================
# OpenAI Settings - Single source of truth for all OpenAI configuration
# =============================================================================
def _parse_optional_float(env_var: str, default: float = None):
    """Parse optional float from env var. Returns None if empty/unset."""
    val = os.getenv(env_var, '')
    if val.strip() == '':
        return default
    return float(val)

def _parse_optional_int(env_var: str, default: int = None):
    """Parse optional int from env var. Returns None if empty/unset."""
    val = os.getenv(env_var, '')
    if val.strip() == '':
        return default
    return int(val)

OPENAI_SETTINGS = {
    "api_key": os.getenv('OPENAI_API_KEY'),
    "api_base": os.getenv('OPENAI_API_BASE'),  # Optional: for Azure or custom endpoints
    "embedding_model": os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
    "chat_model": os.getenv('OPENAI_CHAT_MODEL', 'o3-mini'),
    # Temperature: set to empty string or omit for reasoning models (o1, o3) that don't support it
    "temperature": _parse_optional_float('OPENAI_TEMPERATURE', 0.3),
    # Max tokens: set to empty string or omit to use model default
    "max_tokens": _parse_optional_int('OPENAI_MAX_TOKENS', 1000),
}

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

# Embedding settings (references OPENAI_SETTINGS)
EMBEDDING_SETTINGS = {
    "model_name": OPENAI_SETTINGS["embedding_model"],
    "dimensions": int(os.getenv('OPENAI_EMBEDDING_DIMENSIONS', '1536'))
}

# Retrieval settings
RETRIEVAL_SETTINGS = {
    "keyword_weight": 0.5,
    "semantic_weight": 0.5,
    "default_top_k": 5,
    # Reranker settings - cross-encoder for improved relevance
    "rerank_enabled": os.getenv('RERANK_ENABLED', 'false').lower() == 'true',
    "rerank_model": os.getenv('RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
}

# Agent settings (references OPENAI_SETTINGS)
AGENT_SETTINGS = {
    "model_name": OPENAI_SETTINGS["chat_model"],
    "verbose": False,
    "temperature": OPENAI_SETTINGS["temperature"],
    "max_tokens": OPENAI_SETTINGS["max_tokens"]
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