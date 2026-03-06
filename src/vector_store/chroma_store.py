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

    # Valid metadata fields that can be used in filters
    VALID_FILTER_FIELDS = {
        "source", "filename", "file_type", "file_size",
        "creation_date", "modification_date", "chunk_id",
        "page_number", "section_header", "total_chunks"
    }

    def _sanitize_filter(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sanitize and validate a filter dictionary for ChromaDB.

        ChromaDB where clauses only accept known metadata fields.
        Invalid fields are dropped with a warning.

        Args:
            filter_dict: Raw filter dictionary

        Returns:
            Sanitized filter dict or None if empty/invalid
        """
        if not filter_dict:
            return None

        # If it's already a ChromaDB operator query ($and, $or), validate recursively
        if "$and" in filter_dict or "$or" in filter_dict:
            return filter_dict

        # Filter to only valid fields
        sanitized = {}
        dropped_fields = []

        for key, value in filter_dict.items():
            if key.startswith("$"):
                # ChromaDB operator, keep as-is
                sanitized[key] = value
            elif key in self.VALID_FILTER_FIELDS:
                sanitized[key] = value
            else:
                dropped_fields.append(key)

        if dropped_fields:
            logger.warning(f"Dropped invalid filter fields: {dropped_fields}")

        if not sanitized:
            return None

        # ChromaDB requires $and for multiple field conditions
        if len(sanitized) > 1:
            # Convert {"field1": val1, "field2": val2} to {"$and": [{...}, {...}]}
            conditions = [{k: v} for k, v in sanitized.items()]
            return {"$and": conditions}

        return sanitized

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

            # Add filter if provided (sanitize to prevent invalid filter errors)
            sanitized_filter = self._sanitize_filter(filter_dict)
            if sanitized_filter:
                query_params["where"] = sanitized_filter

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

            # Add filter if provided (sanitize to prevent invalid filter errors)
            sanitized_filter = self._sanitize_filter(filter_dict)
            if sanitized_filter:
                query_params["where"] = sanitized_filter

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

    def list_documents(self) -> Dict[str, int]:
        """List all unique source documents with their chunk counts.

        Returns:
            Dictionary mapping source filename to number of chunks
        """
        try:
            # Get all documents with metadata
            results = self.collection.get(include=["metadatas"])

            if not results["metadatas"]:
                return {}

            # Count chunks per source
            source_counts: Dict[str, int] = {}
            for metadata in results["metadatas"]:
                source = metadata.get("source", "Unknown")
                # Extract just the filename for display
                filename = os.path.basename(source) if source != "Unknown" else source
                source_counts[filename] = source_counts.get(filename, 0) + 1

            return source_counts

        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return {}

    def delete_by_source(self, source_filename: str) -> bool:
        """Delete all chunks belonging to a specific source document.

        Args:
            source_filename: The filename (basename) of the source document

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all documents with metadata
            results = self.collection.get(include=["metadatas"])

            if not results["ids"] or not results["metadatas"]:
                logger.warning("No documents found in collection")
                return False

            # Find IDs matching the source filename
            ids_to_delete = []
            for doc_id, metadata in zip(results["ids"], results["metadatas"]):
                source = metadata.get("source", "")
                if os.path.basename(source) == source_filename:
                    ids_to_delete.append(doc_id)

            if not ids_to_delete:
                logger.warning(f"No chunks found for source: {source_filename}")
                return False

            # Delete the matching documents
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for source: {source_filename}")
            return True

        except Exception as e:
            logger.error(f"Error deleting by source: {str(e)}")
            return False

    def clear_collection(self) -> bool:
        """Delete all documents from the collection.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all document IDs
            results = self.collection.get()

            if not results["ids"]:
                logger.info("Collection is already empty")
                return True

            # Delete all documents
            self.collection.delete(ids=results["ids"])
            logger.info(f"Cleared collection: deleted {len(results['ids'])} documents")
            return True

        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False

    def document_exists(self, source_filename: str) -> bool:
        """Check if a document with the given source filename already exists.

        Args:
            source_filename: The filename (basename) to check

        Returns:
            True if document exists, False otherwise
        """
        try:
            results = self.collection.get(include=["metadatas"])

            if not results["metadatas"]:
                return False

            for metadata in results["metadatas"]:
                source = metadata.get("source", "")
                if os.path.basename(source) == source_filename:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking document existence: {str(e)}")
            return False