import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import openai

logger = logging.getLogger(__name__)


def load_environment():
    """Load environment variables from .env file."""
    env_path = Path('.') / '.env'
    load_dotenv(dotenv_path=env_path)

    # Check for required variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file")
        return False

    return True


def get_file_extension(file_path: str) -> str:
    """Get lowercase file extension without the dot."""
    return Path(file_path).suffix.lower().lstrip('.')


def is_supported_file_type(file_path: str) -> bool:
    """Check if file type is supported."""
    from config import SUPPORTED_EXTENSIONS
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


def get_file_type_display_name(file_path: str) -> str:
    """Get display name for file type."""
    from config import FILE_TYPE_MAPPINGS
    ext = get_file_extension(file_path)
    return FILE_TYPE_MAPPINGS.get(ext, f"{ext.upper()} File")


def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace and controlling line breaks."""
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)

    # Strip whitespace from start and end
    text = text.strip()

    return text


def tokenize_text(text: str) -> List[str]:
    """Simple tokenization for text processing."""
    # Remove punctuation and split on whitespace
    return re.findall(r'\w+', text.lower())


def create_metadata(file_path: str, additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create metadata dictionary for a file."""
    file_path = Path(file_path)

    metadata = {
        'source': str(file_path),
        'filename': file_path.name,
        'file_type': get_file_extension(file_path),
        'file_size': file_path.stat().st_size,
        'creation_date': os.path.getctime(file_path),
        'modification_date': os.path.getmtime(file_path),
        'processed_date': datetime.now().timestamp()
    }

    # Add additional metadata if provided
    if additional_metadata:
        metadata.update(additional_metadata)

    return metadata


def format_date(timestamp: float) -> str:
    """Format a Unix timestamp as a readable date string."""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> bool:
    """Save data as JSON Lines file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        return True
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        return False


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON Lines file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return []


def get_embedding_function(api_key: Optional[str] = None):
    """Get OpenAI embedding function."""

    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key is required")

    embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

    client = openai.OpenAI(api_key=api_key)

    def embedding_function(texts):
        """Generate embeddings for input texts."""
        if isinstance(texts, str):
            texts = [texts]

        try:
            response = client.embeddings.create(
                model=embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return zero embeddings as fallback
            return [[0.0] * 1536 for _ in texts]

    return embedding_function