import os
import uuid
from typing import Dict, List, Any, Optional, Tuple
import logging

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Vector store implementation using Chroma DB."""

    def __init__(
            self,
            persist_directory: str = "./chroma_db",
            collection_name: str = "company_docs",
            embedding_function=None,
            distance_function: str = "cosine"
    ):
        """Initialize the Chroma vector store.

        Args:
            persist_directory: Directory to persist the Chroma DB
            collection_name: Name of the collection to use
            embedding_function: Function to generate embeddings (must be provided externally)
            distance_function: Distance function to use for similarity
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function

        # Ensure persistence directory exists
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_function}
        )

        logger.info(f"Initialized Chroma vector store with collection '{collection_name}'")

    def add_documents(
            self,
            texts: List[str],
            metadatas: List[Dict[str, Any]],
            embeddings: Optional[List[List[float]]] = None,
            ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            texts: List of document texts
            metadatas: List of metadata dictionaries, one per document
            embeddings: Optional pre-computed embeddings
            ids: Optional document IDs (will be generated if not provided)

        Returns:
            List of document IDs
        """
        if not texts:
            logger.warning("No documents to add to vector store")
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        # Generate embeddings if not provided and embedding function is available
        if embeddings is None:
            if self.embedding_function is None:
                raise ValueError("No embedding function provided and no pre-computed embeddings")

            logger.info(f"Generating embeddings for {len(texts)} documents")
            embeddings = self.embedding_function(texts)

        # Ensure all lists have the same length
        if not (len(texts) == len(metadatas) == len(embeddings) == len(ids)):
            raise ValueError("Texts, metadatas, embeddings, and ids must have the same length")

        # Add documents to the collection
        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            logger.info(f"Added {len(texts)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def similarity_search(
            self,
            query_embedding: List[float],
            filter_dict: Optional[Dict[str, Any]] = None,
            top_k: int = 5,
            include_embeddings: bool = False
    ) -> tuple[list[str], list[dict[str, Any]], Optional[list[list[float]]], list[float]]:
        """Search for similar documents using a query embedding.

        Args:
            query_embedding: Query embedding vector
            filter_dict: Optional metadata filter
            top_k: Number of results to return
            include_embeddings: Whether to include embeddings in results

        Returns:
            Tuple of (texts, metadatas, embeddings, scores)
        """
        try:
            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
            }

            # Add filter if provided
            if filter_dict:
                query_params["where"] = filter_dict

            # Add includes based on what's needed
            includes = ["documents", "metadatas", "distances"]
            if include_embeddings:
                includes.append("embeddings")

            query_params["include"] = includes

            # Execute query
            results = self.collection.query(**query_params)

            # Handle case when no results are found
            if not results["documents"] or len(results["documents"]) == 0 or len(results["documents"][0]) == 0:
                logger.warning("No documents found in similarity search")
                return [], [], None, []

            # Extract results
            texts = results["documents"][0]
            metadatas = results["metadatas"][0]
            scores = results["distances"][0]

            # Extract embeddings if requested and available
            embeddings = None
            if include_embeddings and "embeddings" in results:
                embeddings = results["embeddings"][0]

            return texts, metadatas, embeddings, scores

        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return [], [], None, []

    def keyword_search(
            self,
            query: str,
            filter_dict: Optional[Dict[str, Any]] = None,
            top_k: int = 5
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Perform keyword search using Chroma's capabilities.

        Args:
            query: Keyword query string
            filter_dict: Optional metadata filter
            top_k: Number of results to return

        Returns:
            Tuple of (texts, metadatas, scores)
        """
        try:
            # Prepare query parameters
            query_params = {
                "query_texts": [query],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }

            # Add filter if provided
            if filter_dict:
                query_params["where"] = filter_dict

            # Execute query
            results = self.collection.query(**query_params)

            # Extract results
            texts = results["documents"][0]
            metadatas = results["metadatas"][0]
            scores = results["distances"][0]

            return texts, metadatas, scores

        except Exception as e:
            logger.error(f"Error during keyword search: {str(e)}")
            raise

    def get_document_by_id(self, doc_id: str) -> Optional[tuple[Optional[str], Optional[dict[str, Any]]]]:
        """Retrieve a document by its ID.

        Args:
            doc_id: Document ID

        Returns:
            Tuple of (text, metadata) or (None, None) if not found
        """
        try:
            result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])

            if not result["documents"]:
                return None

            return result["documents"][0], result["metadatas"][0]

        except Exception as e:
            logger.error(f"Error retrieving document by ID: {str(e)}")
            return None, None

    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from vector store")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False

    def count_documents(self) -> int:
        """Count the number of documents in the vector store.

        Returns:
            Number of documents
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error counting documents: {str(e)}")
            return 0