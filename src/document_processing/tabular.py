from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
import polars as pl

logger = logging.getLogger(__name__)


def process_tabular_file(file_path: Path, base_metadata: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Process tabular data files (CSV, Excel) using Polars.

    Args:
        file_path: Path to the tabular file
        base_metadata: Base metadata to include with each chunk

    Returns:
        A tuple of (chunks, metadata_list)
    """
    file_ext = file_path.suffix.lower()

    try:
        # Read the file based on its extension
        if file_ext == '.csv':
            df = pl.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pl.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported tabular file format: {file_ext}")

        logger.info(f"Loaded tabular data with {df.shape[0]} rows and {df.shape[1]} columns")

        # Extract table metadata
        table_metadata = {
            'num_rows': df.shape[0],
            'num_columns': df.shape[1],
            'column_names': df.columns,
        }

        # Combine with base metadata
        metadata = {**base_metadata, **table_metadata}

        # Create chunks for tabular data
        chunks = []
        metadata_list = []

        # Add table overview as a chunk
        overview = f"Table Overview: {file_path.name}\n"
        overview += f"Number of rows: {df.shape[0]}\n"
        overview += f"Number of columns: {df.shape[1]}\n"
        overview += f"Columns: {', '.join(df.columns)}\n"

        # Add sample statistics if numerical columns exist
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        if numeric_cols:
            stats = df.select(numeric_cols).describe()
            overview += f"\nStatistics Summary:\n{stats.to_pandas().to_string()}"

        chunks.append(overview)
        metadata_list.append({**metadata, 'chunk_type': 'overview'})

        # Process by reasonable chunk sizes to avoid huge chunks
        chunk_size = min(100, max(10, df.shape[0] // 10))  # Adaptive chunk size

        # Process dataframe in chunks
        for i in range(0, df.shape[0], chunk_size):
            end_idx = min(i + chunk_size, df.shape[0])

            # Create chunk from dataframe subset
            chunk_df = df.slice(i, end_idx - i)

            # Convert chunk to formatted string
            if chunk_df.shape[0] <= 20:  # Small enough for full representation
                chunk_text = f"Data rows {i} to {end_idx - 1}:\n{chunk_df.to_pandas().to_string(index=False)}"
            else:
                # For larger chunks, include summary and sample rows
                sample_rows = min(10, chunk_df.shape[0])
                chunk_text = f"Data rows {i} to {end_idx - 1} (showing first {sample_rows} rows):\n"
                chunk_text += chunk_df.head(sample_rows).to_pandas().to_string(index=False)

            chunks.append(chunk_text)
            metadata_list.append({
                **metadata,
                'chunk_type': 'data',
                'row_start': i,
                'row_end': end_idx - 1
            })

        # For smaller tables, include specialized views like column descriptions
        if df.shape[0] <= 500:
            # Column descriptions (useful for understanding the schema)
            for col in df.columns:
                col_data = df.select(col)
                col_chunk = f"Column: {col}\n"

                # Add column statistics
                if col in numeric_cols:
                    col_stats = col_data.describe()
                    col_chunk += f"Statistics: {col_stats.to_pandas().to_string()}\n"
                else:
                    # For categorical/string columns, show unique values
                    unique_vals = col_data.unique().head(20)
                    col_chunk += f"Sample unique values: {unique_vals.to_pandas().to_string()}\n"
                    total_unique = col_data.unique().shape[0]
                    col_chunk += f"Total unique values: {total_unique}\n"

                chunks.append(col_chunk)
                metadata_list.append({
                    **metadata,
                    'chunk_type': 'column_profile',
                    'column_name': col
                })

        return chunks, metadata_list

    except Exception as e:
        logger.error(f"Error processing tabular file {file_path}: {str(e)}")

        # Return a minimal chunk with error info - don't fail completely
        error_chunk = f"Error processing tabular file {file_path}: {str(e)}"
        error_metadata = {**base_metadata, 'chunk_type': 'error', 'error': str(e)}

        return [error_chunk], [error_metadata]