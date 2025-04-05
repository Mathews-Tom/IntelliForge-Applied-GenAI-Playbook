"""
WebQuestRAG: Dynamic Web RAG Agent
A Streamlit application for dynamic web crawling and RAG using Gemini 2.5 Pro.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from core.llm.gemini_utils import GeminiModelType, generate_content
from core.utils.retrieval_utils import embedding_retrieval
from core.utils.ui_helpers import (
    create_footer,
    create_header,
    create_query_input,
    create_sidebar,
    display_documents,
    display_processing_steps,
    timed_operation,
)
from core.utils.web_crawler import (
    check_crawl4ai_installed,
    crawl_multiple_urls,
    crawl_topic,
    crawl_url,
    load_crawled_content,
    mock_crawl_url,
    save_crawled_content,
)


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
def crawl_web_content(
    crawl_type: str,
    urls: List[str] = None,
    topic: str = None,
    max_pages: int = 10,
    max_depth: int = 2,
) -> List[Dict[str, Any]]:
    """
    Crawl web content based on the specified type.

    Args:
        crawl_type: Type of crawling ('url', 'multiple_urls', or 'topic')
        urls: List of URLs to crawl (for 'url' and 'multiple_urls' types)
        topic: Topic to crawl (for 'topic' type)
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum depth of crawling

    Returns:
        List of dictionaries containing crawled content
    """
    # Check if crawl4ai is installed
    crawl4ai_available = check_crawl4ai_installed()

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "data" / "crawled_content"
    os.makedirs(output_dir, exist_ok=True)

    if crawl_type == "url" and urls and len(urls) > 0:
        url = urls[0]
        if crawl4ai_available:
            return crawl_url(
                url,
                max_pages=max_pages,
                max_depth=max_depth,
                output_dir=str(output_dir),
            )
        else:
            return mock_crawl_url(url, max_pages=max_pages, max_depth=max_depth)

    elif crawl_type == "multiple_urls" and urls and len(urls) > 0:
        if crawl4ai_available:
            return crawl_multiple_urls(
                urls,
                max_pages_per_url=max_pages // len(urls)
                if len(urls) > 0
                else max_pages,
                max_depth=max_depth,
                output_dir=str(output_dir),
            )
        else:
            # Use mock implementation
            all_results = []
            for url in urls:
                all_results.extend(
                    mock_crawl_url(
                        url, max_pages=max_pages // len(urls), max_depth=max_depth
                    )
                )
            return all_results

    elif crawl_type == "topic" and topic:
        if crawl4ai_available:
            return crawl_topic(
                topic,
                num_results=5,
                max_pages_per_result=max_pages // 5,
                max_depth=max_depth,
                output_dir=str(output_dir),
            )
        else:
            # Use mock implementation
            mock_urls = [
                f"https://example.com/search/{topic}/result1",
                f"https://example.com/search/{topic}/result2",
                f"https://example.com/search/{topic}/result3",
            ]
            all_results = []
            for url in mock_urls:
                all_results.extend(
                    mock_crawl_url(
                        url, max_pages=max_pages // len(mock_urls), max_depth=max_depth
                    )
                )
            return all_results

    return []


# Function to generate an answer
def generate_answer(query: str, context: str) -> str:
    """
    Generate an answer based on the query and context.

    Args:
        query: User query
        context: Retrieved context

    Returns:
        Generated answer
    """
    prompt = f"""
    Answer the following question based on the provided context from web pages.
    Include source attribution for key information in your answer.
    If the context doesn't contain enough information to answer the question,
    say so clearly and provide what information you can.

    Question: {query}

    Context:
    {context}
    """

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)
        return response if response else "Failed to generate an answer."
    except Exception as e:
        return f"Error generating answer: {e}"


# Main Streamlit app
def main():
    # Create header
    create_header(
        "WebQuestRAG: Dynamic Web RAG Agent",
        "A RAG system that dynamically builds knowledge bases from web content, powered by Gemini 2.5 Pro.",
        icon="🌐",
    )

    # Create sidebar
    create_sidebar(
        "WebQuestRAG Settings", "Configure how the web crawling and RAG system works."
    )

    # Sidebar settings
    with st.sidebar:
        st.header("Crawling Settings")

        crawl_type = st.selectbox(
            "Crawl Type:",
            options=["Single URL", "Multiple URLs", "Topic/Keywords"],
            index=0,
            format_func=lambda x: x,
        )

        max_pages = st.slider(
            "Maximum Pages:",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum number of pages to crawl",
        )

        max_depth = st.slider(
            "Maximum Depth:",
            min_value=1,
            max_value=5,
            value=2,
            help="Maximum depth of crawling",
        )

        st.header("Retrieval Settings")

        top_k = st.slider(
            "Number of chunks to retrieve:", min_value=1, max_value=10, value=5
        )

        show_process = st.checkbox("Show detailed process", value=True)

    # Main content
    if "GOOGLE_API_KEY" in os.environ:
        # Check dependencies
        has_crawl4ai = check_dependencies()

        # Initialize session state
        if "crawled_content" not in st.session_state:
            st.session_state.crawled_content = []

        if "crawling_completed" not in st.session_state:
            st.session_state.crawling_completed = False

        # Web Crawling Section
        st.subheader("1. Crawl Web Content")

        # Input fields based on crawl type
        if crawl_type == "Single URL":
            url = st.text_input(
                "Enter URL to crawl:", placeholder="https://example.com"
            )
            urls = [url] if url else []
            topic = None
            internal_crawl_type = "url"

        elif crawl_type == "Multiple URLs":
            urls_text = st.text_area(
                "Enter URLs to crawl (one per line):",
                placeholder="https://example.com\nhttps://another-example.com",
            )
            urls = [url.strip() for url in urls_text.split("\n") if url.strip()]
            topic = None
            internal_crawl_type = "multiple_urls"

        else:  # Topic/Keywords
            topic = st.text_input(
                "Enter topic or keywords:", placeholder="artificial intelligence"
            )
            urls = None
            internal_crawl_type = "topic"

        # Crawl button
        if st.button("Start Crawling"):
            if (internal_crawl_type in ["url", "multiple_urls"] and urls) or (
                internal_crawl_type == "topic" and topic
            ):
                with st.spinner("Crawling web content..."):
                    # Crawl web content
                    crawled_content = crawl_web_content(
                        crawl_type=internal_crawl_type,
                        urls=urls,
                        topic=topic,
                        max_pages=max_pages,
                        max_depth=max_depth,
                    )

                    # Save to session state
                    st.session_state.crawled_content = crawled_content
                    st.session_state.crawling_completed = True

                    # Display crawling results
                    st.success(f"Crawled {len(crawled_content)} pages successfully!")
            else:
                st.warning("Please enter a valid URL, URLs, or topic to crawl.")

        # Display crawled content
        if st.session_state.crawling_completed and st.session_state.crawled_content:
            with st.expander("View Crawled Content", expanded=False):
                for i, item in enumerate(st.session_state.crawled_content):
                    st.markdown(f"**Page {i + 1}: {item.get('title', 'Untitled')}**")
                    st.markdown(f"URL: {item.get('url', 'Unknown')}")
                    st.markdown("Content Preview:")
                    content = item.get("text", "")
                    st.markdown(
                        content[:500] + "..." if len(content) > 500 else content
                    )
                    st.markdown("---")

        # Query Section
        if st.session_state.crawling_completed and st.session_state.crawled_content:
            st.subheader("2. Ask Questions About Crawled Content")

            # Query input
            query, search_clicked = create_query_input(
                "Enter your question:", button_text="Search Crawled Content"
            )

            # Process query
            if search_clicked and query:
                # Initialize processing steps
                steps = []

                # Step 1: Retrieve relevant content
                with st.spinner("Retrieving relevant content..."):
                    start_time = time.time()

                    # Extract text from crawled content
                    documents = [
                        item.get("text", "")
                        for item in st.session_state.crawled_content
                    ]

                    # Retrieve relevant chunks
                    retrieved_docs = embedding_retrieval(query, documents, top_k=top_k)

                    retrieval_time = time.time() - start_time

                    # Add to steps
                    steps.append(
                        {
                            "title": "Content Retrieval",
                            "content": f"Retrieved {len(retrieved_docs)} chunks in {retrieval_time:.2f} seconds.",
                            "time": retrieval_time,
                        }
                    )

                # Step 2: Generate answer
                with st.spinner("Generating answer..."):
                    start_time = time.time()

                    # Prepare context
                    context = "\n\n".join([doc for doc, _ in retrieved_docs])

                    # Generate answer
                    answer = generate_answer(query, context)

                    generation_time = time.time() - start_time

                    # Add to steps
                    steps.append(
                        {
                            "title": "Answer Generation",
                            "content": "Generated answer based on retrieved content.",
                            "time": generation_time,
                        }
                    )

                # Display results
                st.subheader("Answer")
                st.markdown(answer)

                # Display retrieved chunks
                st.subheader("Retrieved Content")

                # Convert to format expected by display_documents
                docs_with_scores = []
                for i, (doc, score) in enumerate(retrieved_docs):
                    # Find the original crawled item
                    source_item = None
                    for item in st.session_state.crawled_content:
                        if doc in item.get("text", ""):
                            source_item = item
                            break

                    # Create document with score and source
                    doc_with_score = {
                        "text": doc,
                        "score": score,
                        "source": source_item.get("url", "Unknown")
                        if source_item
                        else "Unknown",
                        "title": source_item.get("title", "Untitled")
                        if source_item
                        else "Untitled",
                    }
                    docs_with_scores.append(doc_with_score)

                # Display documents
                for i, doc in enumerate(docs_with_scores):
                    with st.expander(
                        f"Chunk {i + 1} (Score: {doc['score']:.4f})", expanded=i == 0
                    ):
                        st.markdown(f"**Source:** [{doc['title']}]({doc['source']})")
                        st.markdown(doc["text"])

                # Display processing steps if enabled
                if show_process:
                    display_processing_steps(steps)

            elif search_clicked:
                st.warning("Please enter a question.")

        elif not st.session_state.crawling_completed:
            st.info("Please crawl web content first before asking questions.")

    # Add footer
    create_footer()


if __name__ == "__main__":
    main()
