import os
import logging
import argparse
from pathlib import Path
import openai

from config.config import (
    ROOT_DIR,
    DATA_DIR,
    CHROMA_DIR,
    UPLOAD_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment variables and directories."""
    # Load environment variables from .env file
    from utils.helpers import load_environment
    if not load_environment():
        return False
    # Load config


    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found in environment variables.")
        return False

    # Verify directories
    for directory in [ROOT_DIR, DATA_DIR, CHROMA_DIR, UPLOAD_DIR]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    return True


def run_document_processor(args):
    """Run the document processor on a directory."""
    if not setup_environment():
        return

    from document_processing.processor import DocumentProcessor
    from vector_store.chroma_store import ChromaVectorStore
    from config.config import DOCUMENT_SETTINGS, VECTOR_STORE_SETTINGS

    # Setup OpenAI API
    openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Create embedding function
    def embedding_function(texts):
        if isinstance(texts, str):
            texts = [texts]

        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

    # Initialize components
    processor = DocumentProcessor(
        chunk_size=DOCUMENT_SETTINGS["chunk_size"],
    )

    vector_store = ChromaVectorStore(
        persist_directory=str(Path("./chroma_db")),
        collection_name=VECTOR_STORE_SETTINGS["collection_name"],
        embedding_function=embedding_function,
        distance_function=VECTOR_STORE_SETTINGS["distance_function"]
    )

    # Process directory
    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    logger.info(f"Processing documents in: {input_dir}")
    chunks, metadatas = processor.process_directory(input_dir)

    if not chunks:
        logger.warning("No documents processed.")
        return

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = embedding_function(chunks)

    # Add to vector store
    logger.info(f"Adding {len(chunks)} chunks to vector store...")
    vector_store.add_documents(chunks, metadatas, embeddings)

    logger.info(f"Successfully processed {len(chunks)} chunks from directory: {input_dir}")


def run_streamlit():
    """Run the Streamlit UI."""
    if not setup_environment():
        return

    import subprocess
    import sys

    # Run Streamlit
    streamlit_path = Path(__file__).parent / "ui" / "streamlit_app.py"

    logger.info(f"Starting Streamlit UI from: {streamlit_path}")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(streamlit_path), "--server.port=8501", "--server.address=0.0.0.0"
        ])
    except Exception as e:
        logger.error(f"Error running Streamlit: {str(e)}")


def test_retrieval(args):
    """Test the retrieval system with a query."""
    if not setup_environment():
        return

    from retrieval.hybrid import HybridRetriever
    from vector_store.chroma_store import ChromaVectorStore
    from config.config import VECTOR_STORE_SETTINGS, RETRIEVAL_SETTINGS

    # Setup OpenAI API
    openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Create embedding function
    def embedding_function(texts):
        if isinstance(texts, str):
            texts = [texts]

        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

    # Initialize components
    vector_store = ChromaVectorStore(
        persist_directory=str(Path("./chroma_db")),
        collection_name=VECTOR_STORE_SETTINGS["collection_name"],
        embedding_function=embedding_function,
        distance_function=VECTOR_STORE_SETTINGS["distance_function"]
    )

    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_function=embedding_function,
        keyword_weight=RETRIEVAL_SETTINGS["keyword_weight"],
        semantic_weight=RETRIEVAL_SETTINGS["semantic_weight"]
    )

    # Initialize BM25 index
    retriever.initialize_bm25_index()

    # Get query
    query = args.query
    logger.info(f"Testing retrieval with query: {query}")

    # Determine query type
    is_keyword = retriever.is_keyword_search(query)
    logger.info(f"Query type: {'Keyword' if is_keyword else 'Semantic'}")

    # Retrieve documents
    texts, metadatas, scores = retriever.retrieve(
        query=query,
        top_k=args.top_k or RETRIEVAL_SETTINGS["default_top_k"]
    )

    # Display results
    print(f"\nResults for query: {query}\n")
    print(f"Found {len(texts)} results\n")

    for i, (text, metadata, score) in enumerate(zip(texts, metadatas, scores)):
        print(f"Result {i + 1} [Score: {score:.4f}]")
        print(f"Source: {metadata.get('source', 'Unknown')}")
        print(f"Text: {text[:300]}..." if len(text) > 300 else f"Text: {text}")
        print("-" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Agentic RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Document processor command
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_parser.add_argument("input_dir", help="Input directory containing documents")

    # UI command
    ui_parser = subparsers.add_parser("ui", help="Run the Streamlit UI")

    # Test retrieval command
    test_parser = subparsers.add_parser("test", help="Test retrieval")
    test_parser.add_argument("query", help="Query to test")
    test_parser.add_argument("--top-k", type=int, help="Number of results to return")

    args = parser.parse_args()

    if args.command == "process":
        run_document_processor(args)
    elif args.command == "ui":
        run_streamlit()
    elif args.command == "test":
        test_retrieval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()