from typing import List
import logging
from chonkie import SemanticChunker

logger = logging.getLogger(__name__)


class ChonkieChunker:
    """Wrapper for Chonkie library to perform semantic chunking."""

    def __init__(
            self,
            chunk_size: int = 512,
            semantic_splitter: bool = True
    ):
        """Initialize the chunker.

        Args:
            chunk_size: Target size of each chunk
            semantic_splitter: Whether to use semantic boundaries for splitting
        """
        self.chonkie = SemanticChunker(
            chunk_size=chunk_size,
        )
        self.chunk_size = chunk_size
        self.semantic_splitter = semantic_splitter

    def create_chunks(self, texts: List[str]) -> List[str]:
        """Create semantically meaningful chunks from input texts.

        Args:
            texts: List of text segments to chunk

        Returns:
            List of chunked text segments
        """
        try:
            # Join texts with appropriate separators
            # This maintains document structure while allowing Chonkie to find semantic boundaries
            processed_text = "\n\n".join(texts)

            # Use Chonkie to create chunks
            chunks = self.chonkie.split_text(processed_text)

            logger.info(f"Created {len(chunks)} chunks from {len(texts)} text segments")

            # Post-process chunks (remove empty chunks, clean up formatting)
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

            return chunks

        except Exception as e:
            logger.error(f"Error in chunking: {str(e)}")
            # Fallback to simple chunking if Chonkie fails
            return self._fallback_chunking(texts)

    def _fallback_chunking(self, texts: List[str]) -> List[str]:
        """Simple fallback chunking method if semantic chunking fails.

        Args:
            texts: List of text segments to chunk

        Returns:
            List of chunked text segments
        """
        logger.warning("Using fallback chunking method")

        chunks = []
        current_chunk = []
        current_size = 0

        for text in texts:
            # If adding this text would exceed chunk size
            if current_size + len(text) > self.chunk_size and current_chunk:
                # Save current chunk and start a new one
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                overlap_size = 0
                overlap_texts = []

                # Add texts from the end of previous chunk for overlap
                for t in reversed(current_chunk):
                    if overlap_size + len(t) <= self.chunk_overlap:
                        overlap_texts.insert(0, t)
                        overlap_size += len(t)
                    else:
                        break

                current_chunk = overlap_texts
                current_size = overlap_size

            # Add text to current chunk
            current_chunk.append(text)
            current_size += len(text)

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks