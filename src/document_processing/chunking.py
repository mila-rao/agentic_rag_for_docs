from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class ChonkieChunker:
    """Wrapper for Chonkie library to perform semantic chunking."""

    def __init__(
            self,
            chunk_size: int = 512,
            chunk_overlap: int = 50,
            semantic_splitter: bool = True
    ):
        """Initialize the chunker.

        Args:
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between chunks for fallback chunking
            semantic_splitter: Whether to use semantic boundaries for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_splitter = semantic_splitter
        self.chonkie: Optional[object] = None

        # Try to initialize SemanticChunker, fall back gracefully if it fails
        # (can fail on Windows due to AutoEmbeddings issues)
        if semantic_splitter:
            try:
                from chonkie import SemanticChunker
                self.chonkie = SemanticChunker(
                    chunk_size=chunk_size,
                )
                logger.info("Initialized SemanticChunker successfully")
            except Exception as e:
                # Log the full error with traceback for debugging
                logger.error(
                    f"SEMANTIC CHUNKER INIT FAILED: {type(e).__name__}: {e}",
                    exc_info=True
                )
                logger.warning("Falling back to simple character-based chunking.")
                self.chonkie = None

    def create_chunks(self, texts: List[str]) -> List[str]:
        """Create semantically meaningful chunks from input texts.

        Args:
            texts: List of text segments to chunk

        Returns:
            List of chunked text segments
        """
        # Use fallback if SemanticChunker wasn't initialized
        if self.chonkie is None:
            return self._fallback_chunking(texts=texts)

        try:
            # Join texts with appropriate separators
            # This maintains document structure while allowing Chonkie to find semantic boundaries
            processed_text = "\n\n".join(texts)

            # Use Chonkie to create chunks
            chunks = self.chonkie.split_text(processed_text)

            logger.info(f"Created {len(chunks)} semantic chunks from {len(texts)} text segments")

            # Post-process chunks (remove empty chunks, clean up formatting)
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

            return chunks

        except Exception as e:
            logger.error(f"Error in semantic chunking: {str(e)}")
            # Fallback to simple chunking if Chonkie fails at runtime
            return self._fallback_chunking(texts=texts)

    def _fallback_chunking(self, texts: List[str]) -> List[str]:
        """Simple fallback chunking method if semantic chunking is unavailable.

        Uses character-based chunking with overlap.

        Args:
            texts: List of text segments to chunk

        Returns:
            List of chunked text segments
        """
        logger.info("Using fallback character-based chunking")

        # Join all texts first
        full_text = "\n\n".join(texts)

        if not full_text.strip():
            return []

        chunks = []
        start = 0
        text_length = len(full_text)

        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size

            # If this is not the last chunk, try to break at a sentence or paragraph
            if end < text_length:
                # Look for paragraph break first
                paragraph_break = full_text.rfind("\n\n", start, end)
                if paragraph_break > start + self.chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    # Look for sentence break
                    for sep in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
                        sentence_break = full_text.rfind(sep, start, end)
                        if sentence_break > start + self.chunk_size // 2:
                            end = sentence_break + len(sep)
                            break

            # Extract chunk and clean it
            chunk = full_text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position, accounting for overlap
            start = end - self.chunk_overlap if end < text_length else text_length

        logger.info(f"Created {len(chunks)} chunks using fallback method")
        return chunks