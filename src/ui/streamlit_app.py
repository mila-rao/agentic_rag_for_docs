import streamlit as st
import os
import time
from pathlib import Path
import logging

from utils.helpers import load_environment
from utils.embeddings_helper import get_embedding_function, fallback_embedding_function
from document_processing.processor import DocumentProcessor
from vector_store.chroma_store import ChromaVectorStore
from retrieval.hybrid import HybridRetriever
from agents.crew import RAGCrew

load_environment()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Company Knowledge Base",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "doc_upload_key" not in st.session_state:
    st.session_state.doc_upload_key = 0
if "processing_docs" not in st.session_state:
    st.session_state.processing_docs = False


# Helper functions
def get_embedding_function_for_app():
    """Get the embedding function for the app."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set it in your .env file.")
        st.stop()

    try:
        return get_embedding_function(api_key, max_retries=3)
    except Exception as e:
        st.error(f"Error initializing embedding function: {str(e)}")
        st.warning("Using fallback embedding function. Search results will not be optimal.")
        return fallback_embedding_function


def initialize_components():
    """Initialize all RAG system components."""
    # Directory setup
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    upload_dir = data_dir / "uploads"
    upload_dir.mkdir(exist_ok=True)

    # Get OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please check your .env file.")
        st.stop()

    # Set up components
    doc_processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)

    embedding_function = get_embedding_function_for_app()

    vector_store = ChromaVectorStore(
        persist_directory="./chroma_db",
        collection_name="company_docs",
        embedding_function=embedding_function
    )

    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_function=embedding_function,
        keyword_weight=0.5,
        semantic_weight=0.5
    )

    rag_crew = RAGCrew(
        retriever=retriever,
        llm_api_key=api_key,
        model_name="o3-mini",
        verbose=False
    )

    return {
        "doc_processor": doc_processor,
        "vector_store": vector_store,
        "retriever": retriever,
        "rag_crew": rag_crew,
        "upload_dir": upload_dir
    }


def process_uploaded_files(uploaded_files, components):
    """Process uploaded files and add to vector store."""
    if not uploaded_files:
        return 0

    doc_processor = components["doc_processor"]
    vector_store = components["vector_store"]
    upload_dir = components["upload_dir"]
    embedding_function = get_embedding_function()

    num_processed = 0

    with st.spinner("Processing uploaded documents..."):
        for uploaded_file in uploaded_files:
            # Save file to disk
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process file
            try:
                chunks, metadatas = doc_processor.process_file(file_path)

                if chunks:
                    # Generate embeddings
                    embeddings = embedding_function(chunks)

                    # Add to vector store
                    vector_store.add_documents(chunks, metadatas, embeddings)

                    num_processed += 1
                    st.success(f"Processed: {uploaded_file.name}")
                else:
                    st.warning(f"No content extracted from: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    # Initialize BM25 index after adding documents
    components["retriever"].initialize_bm25_index(force_rebuild=True)

    return num_processed


def display_search_results(results, query):
    """Display search results."""
    st.markdown(f"### Results for: {query}")

    if not results.get("results"):
        st.info("No results found. Try a different search query.")
        return

    # Display each result
    for i, result in enumerate(results["results"]):
        with st.expander(f"Result {i + 1}: {result['source'].split('/')[-1]}"):
            # Display metadata
            st.markdown("**Source:** " + result["source"])
            st.markdown("**Relevance Score:** " + f"{result['score']:.2f}")

            # Display content with highlighting
            content = result["text"]

            # Highlight query terms in the content
            query_terms = query.lower().split()
            highlighted_content = content
            for term in query_terms:
                if len(term) > 3:  # Only highlight terms with more than 3 characters
                    highlighted_content = highlighted_content.replace(
                        term, f"**{term}**"
                    )

            st.markdown(highlighted_content)

            # Display additional metadata
            if "metadata" in result:
                with st.expander("Document Metadata"):
                    for k, v in result["metadata"].items():
                        if k not in ["source", "text"]:
                            st.markdown(f"**{k}:** {v}")


def display_qa_results(results, query):
    """Display question answering results."""
    st.markdown(f"### Answer to: {query}")

    if not results.get("answer"):
        st.info("No answer generated. Try rephrasing your question.")
        return

    # Display the answer
    st.markdown(results["answer"])

    # Display sources if available
    if results.get("sources"):
        with st.expander("Sources"):
            for i, source in enumerate(results["sources"]):
                st.markdown(f"{i + 1}. {source}")

    # Add to search history
    if query not in [item["query"] for item in st.session_state.search_history]:
        st.session_state.search_history.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "type": "question",
            "answer": results.get("answer", "")
        })


def run_streamlit_app():
    """Main Streamlit application."""
    # Page header
    st.title("📚 Company Knowledge Base")
    st.markdown("Search internal documents or ask questions about company information.")

    # Initialize components
    components = initialize_components()

    # Sidebar - Document Upload
    st.sidebar.header("Document Management")

    with st.sidebar.expander("Upload Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload PDF, Word, Excel, Text files",
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.doc_upload_key}"
        )

        if st.button("Process Uploaded Documents"):
            if uploaded_files:
                st.session_state.processing_docs = True
                num_processed = process_uploaded_files(uploaded_files, components)
                st.session_state.doc_upload_key += 1  # Reset uploader
                st.session_state.processing_docs = False
                st.success(f"Successfully processed {num_processed} documents.")
            else:
                st.warning("No documents uploaded.")

    # Document count
    doc_count = components["vector_store"].count_documents()
    st.sidebar.markdown(f"**Documents in Database:** {doc_count}")

    # Sidebar - Search History
    with st.sidebar.expander("Search History", expanded=False):
        if not st.session_state.search_history:
            st.info("No search history yet.")
        else:
            for i, item in enumerate(reversed(st.session_state.search_history[-10:])):
                st.markdown(f"**{item['timestamp']}**")
                st.markdown(f"_{item['query']}_")
                st.markdown("---")

    # Sidebar - Filters
    with st.sidebar.expander("Search Filters", expanded=False):
        # File type filter
        file_types = ["pdf", "docx", "xlsx", "csv", "txt", "md", "json"]
        selected_file_types = st.multiselect(
            "File Types",
            options=file_types,
            default=[]
        )

        # Date range filter
        date_range = st.date_input(
            "Date Range",
            value=[],
            help="Filter documents by creation date"
        )

    # Main area - Search/Question Interface
    st.markdown("### Search or Ask a Question")

    # Search mode toggle
    search_mode = st.radio(
        "Select Mode:",
        options=["Keyword Search", "Ask a Question"],
        horizontal=True
    )

    # Search input
    query = st.text_input(
        "Keyword Search" if search_mode == "Keyword Search" else "Ask a Question",
        placeholder="Enter keywords to search..." if search_mode == "Keyword Search"
        else "Ask a question about company documents..."
    )

    # Build filter dict from sidebar selections
    filter_dict = {}
    if selected_file_types:
        filter_dict["file_type"] = selected_file_types

    if date_range and len(date_range) == 2:
        filter_dict["creation_date"] = {
            "$gte": date_range[0].timestamp(),
            "$lte": date_range[1].timestamp()
        }

    # Search button
    if st.button("Search" if search_mode == "Keyword Search" else "Ask"):
        if not query:
            st.warning("Please enter a search query or question.")
        elif doc_count == 0:
            st.warning("No documents in the database. Please upload some documents first.")
        else:
            with st.spinner("Processing your query..."):
                try:
                    # Process the query
                    rag_crew = components["rag_crew"]

                    # Force keyword mode if selected
                    if search_mode == "Keyword Search":
                        # Use retriever directly for keyword search
                        texts, metadatas, scores = components["retriever"].retrieve(
                            query,
                            filter_dict=filter_dict,
                            top_k=10,
                            force_mode="keyword"
                        )

                        # Format results
                        results = {
                            "type": "search_results",
                            "query": query,
                            "results": [
                                {
                                    "text": text,
                                    "source": metadata.get("source", "Unknown"),
                                    "score": float(score),
                                    "metadata": metadata
                                }
                                for text, metadata, score in zip(texts, metadatas, scores)
                            ],
                            "success": True
                        }

                        # Add to search history
                        st.session_state.search_history.append({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "query": query,
                            "type": "keyword",
                            "results_count": len(texts)
                        })

                        # Display results
                        display_search_results(results, query)

                    else:
                        # Use full RAG crew for questions
                        results = rag_crew.process_query(query, filter_dict)

                        if results["type"] == "search_results":
                            display_search_results(results, query)
                        else:
                            display_qa_results(results, query)

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    logger.error(f"Error processing query: {str(e)}", exc_info=True)

    # Display database status if empty
    if doc_count == 0:
        st.info("The knowledge base is empty. Please upload documents using the sidebar.")

        # Example documents info
        with st.expander("Supported Document Types"):
            st.markdown("""
            - PDF documents (.pdf)
            - Word documents (.docx)
            - Excel spreadsheets (.xlsx, .xls)
            - CSV files (.csv)
            - Text files (.txt)
            - Markdown files (.md)
            - JSON files (.json)
            - HTML files (.html)
            """)


if __name__ == "__main__":
    run_streamlit_app()