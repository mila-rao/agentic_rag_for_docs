import os
import logging
import time
from typing import List, Optional, Union
import numpy as np
import openai

logger = logging.getLogger(__name__)


def get_embedding_function(api_key: Optional[str] = None, max_retries: int = 3, retry_delay: float = 1.0):
    """Get OpenAI embedding function with robust error handling.

    Args:
        api_key: OpenAI API key (will use environment variable if None)
        max_retries: Maximum number of retries for API calls
        retry_delay: Delay between retries in seconds

    Returns:
        Function that generates embeddings
    """

    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key is required")

    embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

    client = openai.OpenAI(api_key=api_key)

    # Cache for previously computed embeddings to avoid duplicate API calls
    embedding_cache = {}

    def embedding_function(texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for input texts.

        Args:
            texts: Single text string or list of text strings

        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]

        # Filter out empty texts
        non_empty_texts = []
        empty_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append(text)
            else:
                empty_indices.append(i)
                logger.warning(f"Empty text at index {i}")

        # If all texts are empty, return zero embeddings
        if not non_empty_texts:
            logger.warning("All texts are empty")
            return [[0.0] * 1536 for _ in texts]

        # Check cache for existing embeddings to avoid redundant API calls
        cache_hits = []
        texts_to_embed = []
        text_to_idx = {}

        for i, text in enumerate(non_empty_texts):
            text_hash = hash(text)
            if text_hash in embedding_cache:
                cache_hits.append((i, embedding_cache[text_hash]))
            else:
                text_to_idx[len(texts_to_embed)] = i
                texts_to_embed.append(text)

        # If all embeddings are in cache, no need for API call
        if not texts_to_embed:
            # Reconstruct the embeddings array
            embeddings = [None] * len(non_empty_texts)
            for i, embedding in cache_hits:
                embeddings[i] = embedding

            # Insert zero embeddings for empty texts
            full_embeddings = []
            empty_idx = 0
            for i in range(len(texts)):
                if i in empty_indices:
                    full_embeddings.append([0.0] * 1536)
                else:
                    full_embeddings.append(embeddings[empty_idx])
                    empty_idx += 1

            return full_embeddings

        # Attempt API call with retries
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=embedding_model,
                    input=texts_to_embed
                )

                # Extract embeddings
                api_embeddings = [item.embedding for item in response.data]

                # Update cache with new embeddings
                for i, embedding in enumerate(api_embeddings):
                    orig_idx = text_to_idx[i]
                    text = non_empty_texts[orig_idx]
                    embedding_cache[hash(text)] = embedding

                # Reconstruct the embeddings array
                embeddings = [None] * len(non_empty_texts)

                # First, fill in cache hits
                for i, embedding in cache_hits:
                    embeddings[i] = embedding

                # Then fill in new embeddings from API
                for api_idx, embedding in enumerate(api_embeddings):
                    orig_idx = text_to_idx[api_idx]
                    embeddings[orig_idx] = embedding

                # Insert zero embeddings for empty texts
                full_embeddings = []
                non_empty_idx = 0
                for i in range(len(texts)):
                    if i in empty_indices:
                        full_embeddings.append([0.0] * 1536)
                    else:
                        full_embeddings.append(embeddings[non_empty_idx])
                        non_empty_idx += 1

                return full_embeddings

            except Exception as e:
                logger.error(f"Error generating embeddings (attempt {attempt + 1}/{max_retries}): {str(e)}")

                # If this was the last attempt, return zero embeddings
                if attempt == max_retries - 1:
                    logger.error("All embedding attempts failed, returning zero embeddings")
                    return [[0.0] * 1536 for _ in texts]

                # Otherwise wait before retrying
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff

    return embedding_function


def fallback_embedding_function(texts: Union[str, List[str]]) -> List[List[float]]:
    """Simple fallback embedding function that doesn't use API calls.

    This is a very basic approach that just hashes the text to create a simple vector.
    Only use this for testing when you can't make API calls.

    Args:
        texts: Single text string or list of text strings

    Returns:
        List of "fake" embedding vectors
    """
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    for text in texts:
        if not text or not text.strip():
            # Empty text gets zero embedding
            embeddings.append([0.0] * 1536)
            continue

        # Create a deterministic but simple vector based on the text
        np.random.seed(hash(text) % 2 ** 32)
        embedding = list(np.random.normal(0, 0.1, 1536))

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]

        embeddings.append(embedding)

    return embeddings