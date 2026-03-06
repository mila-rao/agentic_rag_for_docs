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

    def _compute_rrf_score(self, bm25_rank: Optional[int], semantic_rank: Optional[int], k: int = 60) -> float:
        """Compute Reciprocal Rank Fusion score.

        RRF combines rankings from multiple retrievers using the formula:
        RRF(d) = sum(1 / (k + rank_i(d))) for each retriever i

        Args:
            bm25_rank: Rank in BM25 results (1-indexed), or None if not present
            semantic_rank: Rank in semantic results (1-indexed), or None if not present
            k: Constant to prevent high scores for top ranks (default 60)

        Returns:
            RRF score (higher is better)
        """
        score = 0.0
        if bm25_rank is not None:
            score += 1.0 / (k + bm25_rank)
        if semantic_rank is not None:
            score += 1.0 / (k + semantic_rank)
        return score

    def retrieve(
            self,
            query: str,
            filter_dict: Optional[Dict[str, Any]] = None,
            top_k: int = 5,
            force_mode: Optional[str] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Retrieve documents using hybrid search with Reciprocal Rank Fusion.

        Args:
            query: The user's query
            filter_dict: Optional metadata filter
            top_k: Number of results to return
            force_mode: Force "keyword" or "semantic" mode, or None for hybrid

        Returns:
            Tuple of (texts, metadatas, scores)
        """
        # Determine search mode
        if force_mode is not None:
            is_keyword = (force_mode.lower() == "keyword")
            is_semantic = (force_mode.lower() == "semantic")
        else:
            is_keyword = False
            is_semantic = False

        if is_keyword:
            logger.info(f"Forced keyword-only search: '{query}'")
        elif is_semantic:
            logger.info(f"Forced semantic-only search: '{query}'")
        else:
            logger.info(f"Hybrid search with RRF: '{query}'")

        # Get more candidates for fusion (RRF works better with larger candidate pools)
        expanded_k = min(top_k * 5, 50)

        # Initialize BM25 if needed
        if not self.bm25_initialized:
            self.initialize_bm25_index()

        # Results dictionary keyed by doc_id
        results_dict = {}  # {doc_id: {text, metadata, bm25_rank, semantic_rank}}

        # 1. BM25 keyword search (skip if forced semantic-only)
        if not is_semantic:
            tokenized_query = self._tokenize(query)
            if tokenized_query and self.bm25:
                bm25_scores = self.bm25.get_scores(tokenized_query)

                # Get indices sorted by score (descending)
                sorted_indices = np.argsort(bm25_scores)[::-1]

                # Assign ranks (1-indexed) to documents with positive scores
                rank = 1
                for idx in sorted_indices:
                    if rank > expanded_k:
                        break
                    if bm25_scores[idx] <= 0:
                        continue

                    doc_id = self.doc_ids[idx]
                    doc_text = self.corpus[idx]

                    # Get metadata from vector store
                    _, metadata = self.vector_store.get_document_by_id(doc_id=doc_id)
                    if metadata is None:
                        metadata = {}

                    # Apply metadata filter if specified
                    if filter_dict and not self._matches_filter(metadata=metadata, filter_dict=filter_dict):
                        continue

                    # Store with BM25 rank
                    if doc_id not in results_dict:
                        results_dict[doc_id] = {
                            "text": doc_text,
                            "metadata": metadata,
                            "bm25_rank": rank,
                            "semantic_rank": None
                        }
                    else:
                        results_dict[doc_id]["bm25_rank"] = rank

                    rank += 1

        # 2. Semantic search with embeddings (skip if forced keyword-only)
        if not is_keyword:
            query_embedding = self.embedding_function(query)
            # Handle nested list from embedding function
            if isinstance(query_embedding, list) and len(query_embedding) > 0:
                if isinstance(query_embedding[0], list):
                    query_embedding = query_embedding[0]
                elif hasattr(query_embedding[0], "__len__") and not isinstance(query_embedding[0], (int, float)):
                    query_embedding = list(query_embedding[0])

            # Search vector store
            sem_texts, sem_metadatas, _, sem_distances = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                filter_dict=filter_dict,
                top_k=expanded_k
            )

            # Assign ranks (1-indexed) - results already sorted by distance
            for rank, (text, metadata, distance) in enumerate(zip(sem_texts, sem_metadatas, sem_distances), start=1):
                # Try to get a consistent doc_id from metadata
                doc_id = metadata.get("id")
                if doc_id is None:
                    # Fallback: create ID from source + chunk_id if available
                    source = metadata.get("source", "")
                    chunk_id = metadata.get("chunk_id", rank)
                    doc_id = f"{source}_{chunk_id}"

                if doc_id not in results_dict:
                    results_dict[doc_id] = {
                        "text": text,
                        "metadata": metadata,
                        "bm25_rank": None,
                        "semantic_rank": rank
                    }
                else:
                    results_dict[doc_id]["semantic_rank"] = rank

        # 3. Compute RRF scores
        for doc_id, result in results_dict.items():
            result["rrf_score"] = self._compute_rrf_score(
                bm25_rank=result["bm25_rank"],
                semantic_rank=result["semantic_rank"]
            )

        # Sort by RRF score (descending)
        sorted_results = sorted(
            results_dict.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        # 4. Apply reranker if available
        if self.reranker_function and sorted_results:
            rerank_candidates = [r["text"] for r in sorted_results[:expanded_k]]
            reranked_indices = self.reranker_function(query=query, documents=rerank_candidates)
            sorted_results = [sorted_results[i] for i in reranked_indices]

        # Limit to top_k
        final_results = sorted_results[:top_k]

        # Format return values
        texts = [r["text"] for r in final_results]
        metadatas = [r["metadata"] for r in final_results]
        scores = [r["rrf_score"] for r in final_results]

        logger.info(f"Hybrid retrieval (RRF) returned {len(texts)} results")

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
