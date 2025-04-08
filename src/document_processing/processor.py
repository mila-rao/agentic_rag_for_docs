import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

from unstructured.partition.auto import partition
from document_processing.chunking import ChonkieChunker
from document_processing.pdf_processor import process_pdf

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main document processing class that handles different document types."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize the document processor.

        Args:
            chunk_size: The target size of each document chunk
            chunk_overlap: The overlap between adjacent chunks
        """
        self.chunker = ChonkieChunker(
            chunk_size=chunk_size,
        )

    def process_file(self, file_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process a single file and return chunks with metadata.

        Args:
            file_path: Path to the document file

        Returns:
            A tuple of (chunks, metadata_list)
        """
        file_path = Path(file_path)

        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract base metadata from file
        base_metadata = self._extract_base_metadata(file_path)

        # Process based on file type
        file_ext = file_path.suffix.lower()

        if file_ext in ['.csv', '.xlsx', '.xls']:
            # Handle tabular data separately
            from document_processing.tabular import process_tabular_file
            chunks, metadata_list = process_tabular_file(file_path, base_metadata)
        elif file_ext == '.pdf':
            # Use our custom PDF processor that doesn't require Poppler
            chunks, metadata_list = process_pdf(file_path, base_metadata)
        else:
            # Use Unstructured for all other document types
            try:
                elements = partition(str(file_path))
                logger.info(f"Extracted {len(elements)} elements from {file_path}")

                # Convert elements to text for chunking
                texts = [str(element) for element in elements]

                # Process with Chonkie for semantic chunking
                chunks = self.chunker.create_chunks(texts)

                # Create metadata for each chunk
                metadata_list = [
                    {**base_metadata, 'chunk_id': i}
                    for i in range(len(chunks))
                ]

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                raise

        return chunks, metadata_list

    def process_directory(self, dir_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process all supported documents in a directory.

        Args:
            dir_path: Path to directory containing documents

        Returns:
            A tuple of (all_chunks, all_metadata_list)
        """
        dir_path = Path(dir_path)

        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Invalid directory path: {dir_path}")

        all_chunks = []
        all_metadata = []

        # Supported extensions
        supported_extensions = [
            '.pdf', '.docx', '.pptx', '.txt', '.md',
            '.csv', '.xlsx', '.xls', '.json', '.html'
        ]

        # Process each file
        for file_path in dir_path.glob('**/*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                logger.info(f"Processing file: {file_path}")
                try:
                    chunks, metadata_list = self.process_file(str(file_path))
                    all_chunks.extend(chunks)
                    all_metadata.extend(metadata_list)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    # Continue with other files

        return all_chunks, all_metadata

    @staticmethod
    def _extract_base_metadata(file_path: Path) -> Dict[str, Any]:
        """Extract basic metadata from file.

        Args:
            file_path: Path to the document file

        Returns:
            A dictionary of metadata
        """
        return {
            'source': str(file_path),
            'filename': file_path.name,
            'file_type': file_path.suffix.lstrip('.'),
            'file_size': file_path.stat().st_size,
            'creation_date': os.path.getctime(file_path),
            'modification_date': os.path.getmtime(file_path),
        }