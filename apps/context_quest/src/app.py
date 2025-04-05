"""
ContextQuest: Hybrid Retrieval
A Streamlit application for hybrid retrieval-augmented generation using Gemini 2.5 Pro.
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from core.llm.gemini_utils import GeminiModelType, generate_content
from core.utils.evaluation import evaluate_retrieval_relevance
from core.utils.retrieval_utils import (
    bm25_retrieval,
    embedding_retrieval,
    hybrid_retrieval,
)
from core.utils.ui_helpers import (
    create_footer,
    create_header,
    create_query_input,
    create_sidebar,
    display_documents,
    file_uploader,
)
from core.utils.web_crawler import (
    check_crawl4ai_installed,
    crawl_topic,
    crawl_url,
    mock_crawl_url,
)

# Initialize session state
if "embeddings_cache" not in st.session_state:
    st.session_state.embeddings_cache = {}

if "crawled_content" not in st.session_state:
    st.session_state.crawled_content = []

if "using_crawled_content" not in st.session_state:
    st.session_state.using_crawled_content = False


# Function to check if crawl4ai is installed
def check_dependencies():
    """Check if required dependencies are installed."""
    if not check_crawl4ai_installed():
        st.warning(
            "crawl4ai is not installed. Using mock implementation for demonstration purposes. "
            "To use the full functionality, please install crawl4ai with 'pip install crawl4ai'."
        )
        return False
    return True


# Function to crawl web content
def crawl_web_content(url_or_topic, is_url=True, max_pages=5, max_depth=2):
    """Crawl web content from a URL or topic."""
    # Check if crawl4ai is installed
    crawl4ai_available = check_crawl4ai_installed()

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "data" / "crawled_content"
    os.makedirs(output_dir, exist_ok=True)

    if is_url:
        # Crawl URL
        if crawl4ai_available:
            return crawl_url(
                url_or_topic,
                max_pages=max_pages,
                max_depth=max_depth,
                output_dir=str(output_dir),
            )
        else:
            return mock_crawl_url(
                url_or_topic, max_pages=max_pages, max_depth=max_depth
            )
    else:
        # Crawl topic
        if crawl4ai_available:
            return crawl_topic(
                url_or_topic,
                num_results=3,
                max_pages_per_result=max_pages // 3,
                max_depth=max_depth,
                output_dir=str(output_dir),
            )
        else:
            # Use mock implementation
            mock_urls = [
                f"https://example.com/search/{url_or_topic}/result1",
                f"https://example.com/search/{url_or_topic}/result2",
                f"https://example.com/search/{url_or_topic}/result3",
            ]
            all_results = []
            for url in mock_urls:
                all_results.extend(
                    mock_crawl_url(url, max_pages=max_pages // 3, max_depth=max_depth)
                )
            return all_results


# Function to load and preprocess the context data
@st.cache_data
def load_context_data(file_path: Path) -> pd.DataFrame | None:
    """
    Load and preprocess the context data from a CSV file.

    Args:
        file_path: Path to the CSV file containing context data.

    Returns:
        DataFrame with the context data or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading context data: {e}")
        return None


# Streamlit app
create_header(
    "ContextQuest: Hybrid Retrieval",
    "This application demonstrates hybrid retrieval-augmented generation using a combination of keyword-based (BM25) and semantic (embedding-based) retrieval methods, powered by Gemini 2.5 Pro.",
    icon="🔍",
)

# Sidebar for API keys and settings
create_sidebar(
    "Retrieval Settings", "Configure how documents are retrieved and ranked."
)

with st.sidebar:
    # Data source selection
    st.header("Data Source")
    data_source = st.radio(
        "Select data source:", options=["Local Knowledge Base", "Web Crawling"], index=0
    )

    # Web crawling settings (shown only when web crawling is selected)
    if data_source == "Web Crawling":
        st.subheader("Web Crawling Settings")
        crawl_type = st.radio("Crawl type:", options=["URL", "Topic/Keywords"], index=0)

        max_pages = st.slider(
            "Maximum pages to crawl:", min_value=1, max_value=20, value=5
        )

        max_depth = st.slider("Maximum crawl depth:", min_value=1, max_value=3, value=2)

    # Retrieval settings
    st.header("Retrieval Settings")
    retrieval_method = st.selectbox(
        "Retrieval Method:", options=["Hybrid", "BM25 Only", "Embedding Only"], index=0
    )

    top_k = st.slider(
        "Number of documents to retrieve:", min_value=1, max_value=10, value=3
    )

    if retrieval_method == "Hybrid":
        alpha = st.slider(
            "BM25 weight (α):", min_value=0.0, max_value=1.0, value=0.5, step=0.1
        )
    else:
        alpha = 1.0 if retrieval_method == "BM25 Only" else 0.0

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    ContextQuest uses a hybrid retrieval approach to find the most relevant information
    from a knowledge base or web content, combining keyword-based and semantic search methods.
    """)

# Main content
if "GOOGLE_API_KEY" in os.environ:
    # Check if using web crawling
    if data_source == "Web Crawling":
        # Web crawling section
        st.subheader("Web Content Retrieval")

        # Check dependencies
        check_dependencies()

        if crawl_type == "URL":
            url = st.text_input(
                "Enter URL to crawl:", placeholder="https://example.com"
            )
            crawl_input = url
            is_url = True
        else:  # Topic/Keywords
            topic = st.text_input(
                "Enter topic or keywords:", placeholder="artificial intelligence ethics"
            )
            crawl_input = topic
            is_url = False

        # Crawl button
        if st.button("Crawl Web Content"):
            if crawl_input:
                with st.spinner(
                    f"Crawling {'URL' if is_url else 'topic'}: {crawl_input}..."
                ):
                    # Crawl web content
                    crawled_content = crawl_web_content(
                        crawl_input,
                        is_url=is_url,
                        max_pages=max_pages,
                        max_depth=max_depth,
                    )

                    # Save to session state
                    st.session_state.crawled_content = crawled_content
                    st.session_state.using_crawled_content = True

                    # Display crawling results
                    st.success(f"Crawled {len(crawled_content)} pages successfully!")
            else:
                st.warning(f"Please enter a {'URL' if is_url else 'topic'} to crawl.")

        # Display crawled content
        if st.session_state.using_crawled_content and st.session_state.crawled_content:
            # Extract documents from crawled content
            documents = [
                item.get("text", "") for item in st.session_state.crawled_content
            ]

            # Display the number of documents
            st.info(f"Using {len(documents)} documents from web crawling.")

            # Allow users to view the crawled content
            with st.expander("View Crawled Content"):
                for i, item in enumerate(st.session_state.crawled_content):
                    st.markdown(f"**Page {i + 1}: {item.get('title', 'Untitled')}**")
                    st.markdown(f"URL: {item.get('url', 'Unknown')}")
                    st.markdown("Content Preview:")
                    content = item.get("text", "")
                    st.markdown(
                        content[:300] + "..." if len(content) > 300 else content
                    )
                    st.markdown("---")

            # Query input
            query, search_clicked = create_query_input(
                "Enter your query:", button_text="Search"
            )

            if search_clicked:
                if query:
                    with st.spinner("Retrieving relevant information..."):
                        start_time = time.time()

                        # Retrieve documents based on selected method
                        if retrieval_method == "BM25 Only":
                            retrieved_docs = bm25_retrieval(
                                query, documents, top_k=top_k
                            )
                        elif retrieval_method == "Embedding Only":
                            retrieved_docs = embedding_retrieval(
                                query, documents, top_k=top_k
                            )
                        else:  # Hybrid
                            retrieved_docs = hybrid_retrieval(
                                query, documents, top_k=top_k, alpha=alpha
                            )

                        retrieval_time = time.time() - start_time

                        # Display retrieved documents
                        st.subheader("Retrieved Documents")
                        st.info(
                            f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f} seconds."
                        )

                        # Convert to format expected by display_documents
                        docs_with_scores = [
                            {"text": doc, "score": score}
                            for doc, score in retrieved_docs
                        ]
                        scores = [score for _, score in retrieved_docs]

                        display_documents(docs_with_scores, scores=scores, expand=False)

                        # Generate answer
                        st.subheader("Generated Answer")
                        with st.spinner("Generating answer..."):
                            # Prepare context
                            context = "\n\n".join([doc for doc, _ in retrieved_docs])

                            # Generate answer
                            prompt = f"""
                            Answer the following question based on the provided context.
                            If the context doesn't contain enough information to answer the question,
                            say so clearly and provide what information you can.

                            Question: {query}

                            Context:
                            {context}
                            """

                            answer = generate_content(
                                GeminiModelType.GEMINI_2_5_PRO, prompt
                            )
                            st.markdown(answer)

                        # Evaluate retrieval relevance
                        with st.expander("Retrieval Evaluation"):
                            st.subheader("Relevance Evaluation")
                            with st.spinner("Evaluating retrieval relevance..."):
                                evaluation = evaluate_retrieval_relevance(
                                    query, retrieved_docs
                                )
                                st.markdown(evaluation)
                else:
                    st.warning("Please enter a query.")
        elif st.session_state.using_crawled_content:
            st.warning("No crawled content available. Please crawl web content first.")
    else:
        # Load local context data
        context_data = load_context_data(
            Path(__file__).parent.parent / "data" / "context.csv"
        )

        if context_data is not None:
            # Extract documents
            documents = context_data["text"].tolist()

            # Reset crawled content flag
            st.session_state.using_crawled_content = False

            # Display the number of documents
            st.info(f"Loaded {len(documents)} documents from the knowledge base.")

            # Allow users to view the knowledge base
            with st.expander("View Knowledge Base"):
                for i, doc in enumerate(documents):
                    st.markdown(f"**Document {i + 1}:** {doc}")

            # Query input
            query, search_clicked = create_query_input(
                "Enter your query:", button_text="Search"
            )

            if search_clicked:
                if query:
                    with st.spinner("Retrieving relevant information..."):
                        start_time = time.time()

                        # Retrieve documents based on selected method
                        if retrieval_method == "BM25 Only":
                            retrieved_docs = bm25_retrieval(
                                query, documents, top_k=top_k
                            )
                        elif retrieval_method == "Embedding Only":
                            retrieved_docs = embedding_retrieval(
                                query, documents, top_k=top_k
                            )
                        else:  # Hybrid
                            retrieved_docs = hybrid_retrieval(
                                query, documents, top_k=top_k, alpha=alpha
                            )

                    retrieval_time = time.time() - start_time

                    # Display retrieved documents
                    st.subheader("Retrieved Documents")
                    st.info(
                        f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f} seconds."
                    )

                    # Convert to format expected by display_documents
                    docs_with_scores = [
                        {"text": doc, "score": score} for doc, score in retrieved_docs
                    ]
                    scores = [score for _, score in retrieved_docs]

                    display_documents(docs_with_scores, scores=scores, expand=False)

                    # Generate answer using Gemini
                    with st.spinner("Generating answer..."):
                        # Prepare context for the prompt
                        context = "\n\n".join([doc for doc, _ in retrieved_docs])

                        prompt = f"""
                        Answer the following question based on the provided context.
                        If the context doesn't contain enough information to answer the question,
                        say so clearly and provide what information you can.

                        Question: {query}

                        Context:
                        {context}
                        """

                        response = generate_content(
                            GeminiModelType.GEMINI_2_5_PRO, prompt
                        )

                        if response:
                            st.subheader("Answer")
                            st.markdown(response)
                        else:
                            st.error("Failed to generate an answer. Please try again.")

                    # Evaluate retrieval relevance
                    with st.expander("Retrieval Evaluation"):
                        st.subheader("Relevance Evaluation")
                        with st.spinner("Evaluating retrieval relevance..."):
                            evaluation = evaluate_retrieval_relevance(
                                query, retrieved_docs
                            )
                            st.markdown(evaluation)
                else:
                    st.warning("Please enter a query.")
        else:
            st.error("Failed to load context data. Please check the file path.")

# Add footer
create_footer()
