import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import fitz  

logger = logging.getLogger(__name__)


def process_pdf(file_path: Path, base_metadata: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Process PDF documents without requiring Poppler.

    Args:
        file_path: Path to the PDF file
        base_metadata: Base metadata to include with each chunk

    Returns:
        A tuple of (chunks, metadata_list)
    """
    chunks = []
    metadata_list = []

    try:
        # Open the PDF with PyMuPDF
        logger.info(f"Processing PDF: {file_path}")
        doc = fitz.open(file_path)

        # Extract PDF document metadata
        pdf_metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "page_count": len(doc),
        }

        # Add PDF metadata to base metadata
        enhanced_metadata = {**base_metadata, **pdf_metadata}

        # Process each page and create page-level chunks
        for page_num, page in enumerate(doc):
            # Extract text
            text = page.get_text()

            # Skip empty pages
            if not text.strip():
                continue

            # Create page metadata
            page_metadata = {
                **enhanced_metadata,
                "page_number": page_num + 1,
                "chunk_type": "page",
            }

            # Add page as a chunk
            chunks.append(text)
            metadata_list.append(page_metadata)

            # Try to extract structured content if possible
            try:
                # Extract tables if any (simple heuristic approach)
                tables = extract_tables_from_page(page)

                # Add tables as separate chunks if found
                for i, table_text in enumerate(tables):
                    if table_text.strip():
                        table_metadata = {
                            **enhanced_metadata,
                            "page_number": page_num + 1,
                            "chunk_type": "table",
                            "table_index": i,
                        }
                        chunks.append(f"Table {i + 1} on page {page_num + 1}:\n{table_text}")
                        metadata_list.append(table_metadata)
            except Exception as e:
                logger.warning(f"Could not extract structured content from page {page_num + 1}: {str(e)}")
                # Continue processing other pages

        logger.info(f"Extracted {len(chunks)} chunks from PDF: {file_path}")
        return chunks, metadata_list

    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        # Create a single error chunk
        error_chunk = f"Error processing PDF file: {str(e)}"
        error_metadata = {**base_metadata, "chunk_type": "error", "error": str(e)}
        return [error_chunk], [error_metadata]


def extract_tables_from_page(page) -> List[str]:
    """Extract tables from a PDF page using heuristics.

    This is a simplified approach as accurate table extraction from PDFs
    without specialized tools is challenging.

    Args:
        page: A PyMuPDF page object

    Returns:
        List of extracted table text
    """
    tables = []

    # Simple heuristic - look for blocks of text that might be tables
    # based on tab characters, consistent spacing, etc.
    blocks = page.get_text("blocks")

    for block in blocks:
        block_text = block[4]  # Text content is at index 4

        # Heuristics to detect tabular content
        # 1. Contains multiple tab characters
        # 2. Has multiple lines with similar character positions
        # 3. Contains regular patterns of spaces or delimiters

        if '\t' in block_text and block_text.count('\n') > 1:
            tables.append(block_text)
            continue

        # Check for aligned columns using spaces
        lines = block_text.split('\n')
        if len(lines) > 2:
            # Look for consistent positioning of spaces across lines
            space_positions = [
                [i for i, char in enumerate(line) if char == ' ']
                for line in lines if line.strip()
            ]

            # If we have consistent space positions across multiple lines,
            # it might be a table
            if space_positions and all(len(pos) > 3 for pos in space_positions):
                # Check for alignment consistency
                is_aligned = True
                for i in range(1, len(space_positions)):
                    # If space positions differ significantly, it's probably not aligned
                    if len(space_positions[i]) != len(space_positions[0]):
                        is_aligned = False
                        break

                if is_aligned:
                    tables.append(block_text)

    return tables