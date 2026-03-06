"""Cross-encoder reranker for improving retrieval quality."""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Lazy load to avoid import overhead when reranking is disabled
_cross_encoder = None


def _get_cross_encoder(model_name: str):
    """Lazy load the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder model: {model_name}")
            _cross_encoder = CrossEncoder(model_name)
            logger.info("Cross-encoder model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise
    return _cross_encoder


class CrossEncoderReranker:
    """Reranker using cross-encoder models for improved relevance scoring."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: Optional[int] = None
    ):
        """Initialize the reranker.

        Args:
            model_name: Name of the cross-encoder model from HuggingFace
            top_k: If set, only return top_k results after reranking
        """
        self.model_name = model_name
        self.top_k = top_k
        self._model = None

    def _ensure_model_loaded(self):
        """Ensure the model is loaded (lazy loading)."""
        if self._model is None:
            self._model = _get_cross_encoder(model_name=self.model_name)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[int]:
        """Rerank documents based on relevance to query.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (overrides instance top_k)

        Returns:
            List of indices sorted by relevance (most relevant first)
        """
        if not documents:
            return []

        self._ensure_model_loaded()

        # Create query-document pairs for scoring
        pairs = [[query, doc] for doc in documents]

        try:
            # Get relevance scores from cross-encoder
            scores = self._model.predict(pairs)

            # Sort by score (descending) and get indices
            ranked_indices = np.argsort(scores)[::-1].tolist()

            # Apply top_k limit if specified
            k = top_k or self.top_k
            if k is not None and k < len(ranked_indices):
                ranked_indices = ranked_indices[:k]

            logger.debug(f"Reranked {len(documents)} documents, returning top {len(ranked_indices)}")
            return ranked_indices

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Return original order on error
            return list(range(len(documents)))

    def rerank_with_scores(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[tuple]:
        """Rerank documents and return indices with scores.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            List of (index, score) tuples sorted by relevance
        """
        if not documents:
            return []

        self._ensure_model_loaded()

        pairs = [[query, doc] for doc in documents]

        try:
            scores = self._model.predict(pairs)

            # Create (index, score) pairs and sort by score descending
            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # Apply top_k limit
            k = top_k or self.top_k
            if k is not None and k < len(indexed_scores):
                indexed_scores = indexed_scores[:k]

            return indexed_scores

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return [(i, 0.0) for i in range(len(documents))]


def create_reranker_function(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: Optional[int] = None
):
    """Factory function to create a reranker function for HybridRetriever.

    Args:
        model_name: Name of the cross-encoder model
        top_k: Optional limit on results

    Returns:
        A function that takes (query, documents) and returns reranked indices
    """
    reranker = CrossEncoderReranker(model_name=model_name, top_k=top_k)

    def reranker_function(query: str, documents: List[str]) -> List[int]:
        return reranker.rerank(query=query, documents=documents)

    return reranker_function
