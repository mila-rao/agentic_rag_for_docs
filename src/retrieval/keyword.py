import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from utils.helpers import tokenize_text

logger = logging.getLogger(__name__)


class KeywordRetriever:
    """Keyword-based retrieval using BM25."""

    def __init__(self):
        """Initialize the keyword retriever."""
        self.corpus = []
        self.doc_ids = []
        self.metadatas = []
        self.bm25 = None
        self.is_initialized = False

    def add_documents(self, texts: List[str], ids: List[str], metadatas: List[Dict[str, Any]]):
        """Add documents to the BM25 index.

        Args:
            texts: List of document texts
            ids: List of document IDs
            metadatas: List of document metadata dictionaries
        """
        # Ensure all lists have the same length
        if not (len(texts) == len(ids) == len(metadatas)):
            raise ValueError("texts, ids, and metadatas must have the same length")

        # Update our stored corpus, ids, and metadata
        self.corpus.extend(texts)
        self.doc_ids.extend(ids)
        self.metadatas.extend(metadatas)

        # Reset the BM25 model to force re-initialization
        self.is_initialized = False

    def initialize(self, force: bool = False):
        """Initialize or re-initialize the BM25 model.

        Args:
            force: Whether to force reinitialization
        """
        if self.is_initialized and not force:
            return

        if not self.corpus:
            logger.warning("No documents in corpus, BM25 initialization skipped")
            return

        # Tokenize the corpus
        tokenized_corpus = [tokenize_text(doc) for doc in self.corpus]

        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.is_initialized = True

        logger.info(f"BM25 index initialized with {len(self.corpus)} documents")

    def search(
            self,
            query: str,
            filter_dict: Optional[Dict[str, Any]] = None,
            top_k: int = 5
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Search for documents using BM25.

        Args:
            query: Search query
            filter_dict: Optional metadata filter
            top_k: Number of results to return

        Returns:
            Tuple of (texts, metadatas, scores)
        """
        if not self.is_initialized:
            self.initialize()

        if not self.bm25:
            logger.warning("BM25 index not initialized, returning empty results")
            return [], [], []

        # Tokenize the query
        tokenized_query = tokenize_text(query)

        if not tokenized_query:
            logger.warning("Query contains no valid tokens, returning empty results")
            return [], [], []

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get all result indices sorted by score
        indices = np.argsort(scores)[::-1]

        # Apply metadata filter if provided
        if filter_dict:
            filtered_indices = []
            for idx in indices:
                if idx < len(self.metadatas):
                    if self._matches_filter(self.metadatas[idx], filter_dict):
                        filtered_indices.append(idx)
            indices = filtered_indices

        # Limit to top_k
        indices = indices[:top_k]

        # Gather results
        result_texts = [self.corpus[idx] for idx in indices if idx < len(self.corpus)]
        result_metadatas = [self.metadatas[idx] for idx in indices if idx < len(self.metadatas)]
        result_scores = [float(scores[idx]) for idx in indices if idx < len(scores)]

        return result_texts, result_metadatas, result_scores

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter criteria.

        Args:
            metadata: Document metadata
            filter_dict: Filter criteria

        Returns:
            True if metadata matches all filter criteria, False otherwise
        """
        for key, value in filter_dict.items():
            if key not in metadata:
                return False

            # Handle different filter types
            if isinstance(value, list):
                # List of possible values
                if metadata[key] not in value:
                    return False
            elif isinstance(value, dict):
                # Range queries with operators like $gt, $lt
                for op, op_value in value.items():
                    if op == "$gt" and not (metadata[key] > op_value):
                        return False
                    elif op == "$gte" and not (metadata[key] >= op_value):
                        return False
                    elif op == "$lt" and not (metadata[key] < op_value):
                        return False
                    elif op == "$lte" and not (metadata[key] <= op_value):
                        return False
                    elif op == "$ne" and metadata[key] == op_value:
                        return False
            else:
                # Exact match
                if metadata[key] != value:
                    return False

        return True