from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
import numpy as np
from rank_bm25 import BM25Okapi
from vector_store.chroma_store import ChromaVectorStore
import re

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval system combining keyword and semantic search."""

    def __init__(
            self,
            vector_store: ChromaVectorStore,
            embedding_function: Callable[[str], List[float]],
            reranker_function: Optional[Callable] = None,
            keyword_weight: float = 0.5,
            semantic_weight: float = 0.5
    ):
        """Initialize the hybrid retriever.

        Args:
            vector_store: ChromaVectorStore instance
            embedding_function: Function to generate embeddings from text
            reranker_function: Optional function to rerank results
            keyword_weight: Weight for keyword search results (0-1)
            semantic_weight: Weight for semantic search results (0-1)
        """
        self.vector_store = vector_store
        self.embedding_function = embedding_function
        self.reranker_function = reranker_function
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight

        # Set up in-memory BM25 index (will be populated on demand)
        self.corpus = []
        self.doc_ids = []
        self.bm25 = None
        self.bm25_initialized = False

    def initialize_bm25_index(self, force_rebuild: bool = False):
        """Initialize or rebuild the BM25 index.

        Args:
            force_rebuild: Whether to force a rebuild of the index
        """
        if self.bm25_initialized and not force_rebuild:
            return

        logger.info("Initializing BM25 index")

        # Get all documents from the vector store
        total_docs = self.vector_store.count_documents()

        if total_docs == 0:
            logger.warning("No documents in vector store to build BM25 index")
            return

        try:
            # Use collection's native get method to retrieve all documents
            # This is more efficient than multiple queries
            results = self.vector_store.collection.get(
                include=["documents", "metadatas"]
            )

            # Get document IDs from the results
            self.doc_ids = results.get("ids", [])

            # If IDs weren't included, generate some
            if not self.doc_ids:
                self.doc_ids = [f"doc_{i}" for i in range(len(results.get("documents", [])))]

            self.corpus = results.get("documents", [])

            # Tokenize the corpus for BM25
            tokenized_corpus = [self._tokenize(doc) for doc in self.corpus if doc]

            # Initialize BM25
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.bm25_initialized = True

            logger.info(f"BM25 index initialized with {len(self.corpus)} documents")
        except Exception as e:
            logger.error(f"Error initializing BM25 index: {str(e)}")
            self.bm25_initialized = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization function for BM25.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Convert to lowercase and split on non-alphanumeric characters
        return re.findall(r'\w+', text.lower())

    def is_keyword_search(self, query: str) -> bool:
        """Determine if the query is likely a keyword search vs. a natural question.

        Args:
            query: The user's query

        Returns:
            True if likely a keyword search, False if likely a question
        """
        # Check for question words or question mark
        question_indicators = ["what", "why", "how", "when", "where", "who", "which", "?"]

        query_lower = query.lower()

        # Look for question patterns
        for indicator in question_indicators:
            if query_lower.startswith(indicator) or query_lower.endswith("?"):
                return False

        # Check for complete sentence structure
        if len(query.split()) > 3 and any(query.endswith(p) for p in [".", "?", "!"]):
            return False

        # Count stopwords percentage
        stopwords = {"the", "a", "an", "and", "or", "but", "if", "because", "as", "what",
                     "which", "is", "are", "was", "were", "be", "been", "being"}

        tokens = query_lower.split()
        stopword_ratio = sum(1 for token in tokens if token in stopwords) / max(1, len(tokens))

        # High stopword ratio suggests natural language question
        if stopword_ratio > 0.3:
            return False

        # Default to keyword search
        return True

    def retrieve(
            self,
            query: str,
            filter_dict: Optional[Dict[str, Any]] = None,
            top_k: int = 5,
            force_mode: Optional[str] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Retrieve documents using hybrid search.

        Args:
            query: The user's query
            filter_dict: Optional metadata filter
            top_k: Number of results to return
            force_mode: Force "keyword" or "semantic" mode, or None for automatic

        Returns:
            Tuple of (texts, metadatas, scores)
        """
        # Determine search mode
        if force_mode is not None:
            is_keyword = (force_mode.lower() == "keyword")
        else:
            is_keyword = self.is_keyword_search(query)

        # Adjust weights based on query type
        if is_keyword:
            kw_weight = 0.7  # Higher weight for keyword search
            sem_weight = 0.3
            logger.info(f"Query interpreted as keyword search: '{query}'")
        else:
            kw_weight = 0.3  # Higher weight for semantic search
            sem_weight = 0.7
            logger.info(f"Query interpreted as semantic question: '{query}'")

        # Get expanded top_k to ensure we have enough candidates for reranking
        expanded_k = min(top_k * 3, 30)  # Get more candidates but cap it

        # Initialize BM25 if needed
        if not self.bm25_initialized:
            self.initialize_bm25_index()

        # Combined retrieval
        results_dict = {}  # Will store combined results

        # 1. Keyword search with BM25
        tokenized_query = self._tokenize(query)
        if tokenized_query and self.bm25:
            bm25_scores = self.bm25.get_scores(tokenized_query)

            # Get top results
            top_indices = np.argsort(bm25_scores)[-expanded_k:][::-1]

            # Add to results dictionary with normalized scores
            max_bm25 = max(bm25_scores[top_indices]) if len(top_indices) > 0 else 1.0
            for idx in top_indices:
                if bm25_scores[idx] > 0:  # Only include matches
                    doc_id = self.doc_ids[idx]
                    doc_text = self.corpus[idx]

                    # Get metadata from vector store
                    _, metadata = self.vector_store.get_document_by_id(doc_id)

                    # Apply metadata filter if specified
                    if filter_dict and not self._matches_filter(metadata, filter_dict):
                        continue

                    # Normalize score to 0-1 range
                    normalized_score = bm25_scores[idx] / max_bm25

                    # Store in results with source information
                    if doc_id not in results_dict:
                        results_dict[doc_id] = {
                            "text": doc_text,
                            "metadata": metadata,
                            "keyword_score": normalized_score,
                            "semantic_score": 0.0
                        }
                    else:
                        results_dict[doc_id]["keyword_score"] = normalized_score

        # 2. Semantic search with embeddings
        query_embedding = self.embedding_function(query)

        # Search vector store
        sem_texts, sem_metadatas, _, sem_scores = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            filter_dict=filter_dict,
            top_k=expanded_k
        )

        # Normalize semantic scores (convert distances to similarities)
        max_sem_dist = max(sem_scores) if sem_scores else 1.0
        sem_similarities = [1.0 - (dist / max_sem_dist) for dist in sem_scores]

        # Add semantic results
        for i, (text, metadata, score) in enumerate(zip(sem_texts, sem_metadatas, sem_similarities)):
            doc_id = metadata.get("id", f"sem_{i}")

            if doc_id not in results_dict:
                results_dict[doc_id] = {
                    "text": text,
                    "metadata": metadata,
                    "keyword_score": 0.0,
                    "semantic_score": score
                }
            else:
                results_dict[doc_id]["semantic_score"] = score

        # 3. Combine scores
        for doc_id, result in results_dict.items():
            # Combined score with weightings
            result["combined_score"] = (
                    (kw_weight * result["keyword_score"]) +
                    (sem_weight * result["semantic_score"])
            )

        # Sort by combined score
        sorted_results = sorted(
            results_dict.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )

        # 4. Apply reranker if available
        if self.reranker_function and sorted_results:
            # Prepare inputs for reranker
            rerank_candidates = [r["text"] for r in sorted_results[:expanded_k]]

            # Get reranked indices
            reranked_indices = self.reranker_function(query, rerank_candidates)

            # Reorder results
            reranked_results = [sorted_results[i] for i in reranked_indices]
            sorted_results = reranked_results

        # Limit to top_k
        final_results = sorted_results[:top_k]

        # Format return values
        texts = [r["text"] for r in final_results]
        metadatas = [r["metadata"] for r in final_results]
        scores = [r["combined_score"] for r in final_results]

        logger.info(f"Hybrid retrieval returned {len(texts)} results")

        return texts, metadatas, scores

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
