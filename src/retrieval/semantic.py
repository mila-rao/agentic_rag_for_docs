import logging
from typing import List, Dict, Any, Tuple, Optional, Callable

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Semantic retrieval using vector search."""

    def __init__(
            self,
            vector_store,
            embedding_function: Callable[[str], List[float]]
    ):
        """Initialize the semantic retriever.

        Args:
            vector_store: Vector store instance
            embedding_function: Function to generate embeddings from text
        """
        self.vector_store = vector_store
        self.embedding_function = embedding_function

    def search(
            self,
            query: str,
            filter_dict: Optional[Dict[str, Any]] = None,
            top_k: int = 5
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Search for semantically similar documents.

        Args:
            query: Search query
            filter_dict: Optional metadata filter
            top_k: Number of results to return

        Returns:
            Tuple of (texts, metadatas, scores)
        """
        if not query.strip():
            logger.warning("Empty query, returning empty results")
            return [], [], []

        try:
            # Generate query embedding
            query_embedding = self.embedding_function(query)

            # Search vector store
            texts, metadatas, _, distances = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                filter_dict=filter_dict,
                top_k=top_k
            )

            # Convert distances to similarity scores (1.0 - distance)
            # This assumes cosine distance; adjust for other distance metrics
            scores = [1.0 - min(dist, 1.0) for dist in distances]

            return texts, metadatas, scores

        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return [], [], []

    def batch_search(
            self,
            queries: List[str],
            filter_dict: Optional[Dict[str, Any]] = None,
            top_k: int = 5
    ) -> List[Tuple[List[str], List[Dict[str, Any]], List[float]]]:
        """Perform semantic search for multiple queries.

        Args:
            queries: List of search queries
            filter_dict: Optional metadata filter
            top_k: Number of results to return per query

        Returns:
            List of (texts, metadatas, scores) tuples, one for each query
        """
        results = []

        for query in queries:
            result = self.search(query, filter_dict, top_k)
            results.append(result)

        return results