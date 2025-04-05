"""
ResearchAssistantAgent: Web-Based Research Helper
A Streamlit application for gathering and synthesizing research information using Gemini 2.5 Pro.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from core.llm.gemini_utils import GeminiModelType, generate_content
from core.utils.retrieval_utils import embedding_retrieval, hybrid_retrieval
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
    crawl_topic,
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


# Function to crawl research topic
def crawl_research_topic(
    topic: str,
    subtopics: List[str] = None,
    num_results: int = 10,
    max_pages_per_result: int = 3,
    max_depth: int = 2,
    include_academic: bool = True,
    include_news: bool = True,
) -> List[Dict[str, Any]]:
    """
    Crawl web content related to a research topic.

    Args:
        topic: Main research topic
        subtopics: List of subtopics to explore
        num_results: Number of search results to crawl
        max_pages_per_result: Maximum number of pages to crawl per search result
        max_depth: Maximum depth of crawling
        include_academic: Whether to include academic sources
        include_news: Whether to include news sources

    Returns:
        List of dictionaries containing crawled content
    """
    # Check if crawl4ai is installed
    crawl4ai_available = check_crawl4ai_installed()

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "data" / "research_content"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare search queries
    search_queries = [topic]

    if subtopics:
        for subtopic in subtopics:
            search_queries.append(f"{topic} {subtopic}")

    # Add academic and news modifiers
    if include_academic:
        search_queries.append(f"{topic} research paper")
        search_queries.append(f"{topic} academic study")

    if include_news:
        search_queries.append(f"{topic} latest news")
        search_queries.append(f"{topic} recent developments")

    # Crawl each search query
    all_results = []

    for query in search_queries:
        with st.spinner(f"Researching: {query}..."):
            if crawl4ai_available:
                # Use crawl4ai
                results = crawl_topic(
                    topic=query,
                    num_results=num_results // len(search_queries),
                    max_pages_per_result=max_pages_per_result,
                    max_depth=max_depth,
                    output_dir=str(output_dir),
                )
            else:
                # Use mock implementation
                mock_urls = [
                    f"https://example.com/research/{query}/result1",
                    f"https://example.com/research/{query}/result2",
                    f"https://academic.example.edu/papers/{query}",
                ]

                results = []
                for url in mock_urls:
                    results.extend(
                        mock_crawl_url(
                            url=url, max_pages=max_pages_per_result, max_depth=max_depth
                        )
                    )

            # Add query to each result
            for result in results:
                result["query"] = query
                result["crawled_at"] = datetime.now().isoformat()

            all_results.extend(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"research_{timestamp}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return all_results


# Function to generate research summary
def generate_research_summary(research_data: List[Dict[str, Any]]) -> str:
    """
    Generate a summary of research findings.

    Args:
        research_data: List of dictionaries containing crawled research content

    Returns:
        Research summary as a string
    """
    # Prepare research information
    research_info = {"sources": len(research_data), "content": []}

    # Add content from each source (limit to avoid token limits)
    for i, item in enumerate(research_data[:20]):  # Limit to 20 sources
        source_info = {
            "title": item.get("title", f"Source {i + 1}"),
            "url": item.get("url", "Unknown URL"),
            "text": item.get("text", "")[:5000],  # Limit text length
            "query": item.get("query", "Unknown query"),
        }
        research_info["content"].append(source_info)

    # Create prompt
    prompt = f"""
    Generate a comprehensive research summary based on the following web content.
    
    Research Information:
    {json.dumps(research_info, indent=2)}
    
    Your task is to:
    1. Synthesize the key findings and information from these sources
    2. Organize the summary into clear sections with headings
    3. Highlight areas of consensus and any contradictions or debates
    4. Include proper citations for key information (use [Source X] format)
    5. Identify any gaps in the research or areas that need further exploration
    
    Format your summary as a well-structured research report with:
    - An executive summary
    - Key findings organized by theme
    - Methodology limitations (based on the sources used)
    - Recommendations for further research
    
    Use academic language and maintain a neutral, objective tone throughout.
    """

    # Generate summary
    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)
        return response if response else "Failed to generate research summary."
    except Exception as e:
        return f"Error generating research summary: {e}"


# Function to generate citations
def generate_citations(research_data: List[Dict[str, Any]], style: str = "APA") -> str:
    """
    Generate citations for research sources.

    Args:
        research_data: List of dictionaries containing crawled research content
        style: Citation style (APA, MLA, Chicago)

    Returns:
        Formatted citations as a string
    """
    # Prepare source information
    sources = []

    for item in research_data:
        source = {
            "title": item.get("title", "Unknown Title"),
            "url": item.get("url", ""),
            "date_accessed": datetime.now().strftime("%Y-%m-%d"),
            "date_published": item.get("date_published", "n.d."),
            "authors": item.get("authors", []),
            "website_name": item.get("website_name", ""),
        }

        # Extract website name from URL if not provided
        if not source["website_name"] and source["url"]:
            try:
                from urllib.parse import urlparse

                domain = urlparse(source["url"]).netloc
                source["website_name"] = domain.replace("www.", "")
            except:
                source["website_name"] = "Unknown Website"

        sources.append(source)

    # Create prompt
    prompt = f"""
    Generate citations for the following sources in {style} format.
    
    Sources:
    {json.dumps(sources, indent=2)}
    
    For each source, create a properly formatted citation according to {style} style guidelines.
    If information is missing (like author names or publication dates), follow {style} guidelines for handling missing information.
    
    Return the citations as a numbered list, sorted alphabetically according to {style} rules.
    """

    # Generate citations
    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)
        return response if response else "Failed to generate citations."
    except Exception as e:
        return f"Error generating citations: {e}"


# Function to answer research questions
def answer_research_question(
    query: str, research_data: List[Dict[str, Any]], top_k: int = 5
) -> str:
    """
    Answer a research question based on crawled content.

    Args:
        query: Research question
        research_data: List of dictionaries containing crawled research content
        top_k: Number of chunks to retrieve

    Returns:
        Answer to the research question
    """
    # Extract text from research data
    texts = [item.get("text", "") for item in research_data]

    # Retrieve relevant chunks using hybrid retrieval
    retrieved_docs = hybrid_retrieval(query, texts, top_k=top_k)

    # Map retrieved texts back to full documents
    retrieved_documents = []
    for text, score in retrieved_docs:
        for item in research_data:
            if item.get("text", "") == text:
                doc_with_score = {
                    "text": text,
                    "score": score,
                    "title": item.get("title", "Unknown Title"),
                    "url": item.get("url", "Unknown URL"),
                }
                retrieved_documents.append(doc_with_score)
                break

    # Prepare context
    context = ""
    for i, doc in enumerate(retrieved_documents):
        context += f"Source {i + 1}: {doc.get('title', 'Unknown Title')}\n"
        context += f"URL: {doc.get('url', 'Unknown URL')}\n"
        context += f"Content: {doc.get('text', '')}\n\n"

    # Create prompt
    prompt = f"""
    Answer the following research question based on the provided sources.
    
    Research Question: {query}
    
    Sources:
    {context}
    
    Your task is to:
    1. Provide a comprehensive answer that directly addresses the research question
    2. Base your answer strictly on the provided sources
    3. Include in-text citations for key information (use [Source X] format)
    4. Acknowledge any limitations or gaps in the available information
    5. Maintain an academic, objective tone
    
    Format your answer as a well-structured response with clear paragraphs and proper citations.
    Include a brief "Sources" section at the end listing the full URLs of cited sources.
    """

    # Generate answer
    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)
        return response if response else "Failed to answer research question."
    except Exception as e:
        return f"Error answering research question: {e}"


# Main Streamlit app
def main():
    # Create header
    create_header(
        "ResearchAssistantAgent: Web-Based Research Helper",
        "An advanced application for gathering and synthesizing research information from the web, powered by Gemini 2.5 Pro.",
        icon="📚",
    )

    # Create sidebar
    create_sidebar(
        "ResearchAssistantAgent Settings",
        "Configure how the research assistant system works.",
    )

    # Sidebar settings
    with st.sidebar:
        st.header("Research Settings")

        num_results = st.slider(
            "Number of Sources:",
            min_value=5,
            max_value=30,
            value=15,
            help="Number of sources to gather for research",
        )

        max_pages = st.slider(
            "Maximum Pages per Source:",
            min_value=1,
            max_value=5,
            value=3,
            help="Maximum number of pages to crawl per source",
        )

        max_depth = st.slider(
            "Maximum Depth:",
            min_value=1,
            max_value=3,
            value=2,
            help="Maximum depth of crawling",
        )

        st.header("Source Settings")

        include_academic = st.checkbox("Include Academic Sources", value=True)
        include_news = st.checkbox("Include News Sources", value=True)

        st.header("Citation Settings")

        citation_style = st.selectbox(
            "Citation Style:", options=["APA", "MLA", "Chicago"], index=0
        )

        show_process = st.checkbox("Show detailed process", value=True)

    # Main content
    if "GOOGLE_API_KEY" in os.environ:
        # Check dependencies
        has_crawl4ai = check_dependencies()

        # Initialize session state
        if "research_data" not in st.session_state:
            st.session_state.research_data = []

        if "research_completed" not in st.session_state:
            st.session_state.research_completed = False

        # Research Topic Section
        st.subheader("1. Specify Research Topic")

        # Input for research topic
        research_topic = st.text_input(
            "Enter your main research topic:",
            placeholder="Artificial Intelligence Ethics",
            help="Enter the main topic you want to research",
        )

        # Input for subtopics
        subtopics_text = st.text_area(
            "Enter subtopics (one per line, optional):",
            placeholder="Privacy concerns\nBias in algorithms\nRegulatory frameworks",
            help="Enter specific aspects of the main topic you want to explore",
        )

        # Parse subtopics
        subtopics = []
        if subtopics_text:
            subtopics = [
                topic.strip() for topic in subtopics_text.split("\n") if topic.strip()
            ]

        # Research button
        if st.button("Start Research"):
            if research_topic:
                # Crawl research topic
                research_data = crawl_research_topic(
                    topic=research_topic,
                    subtopics=subtopics,
                    num_results=num_results,
                    max_pages_per_result=max_pages,
                    max_depth=max_depth,
                    include_academic=include_academic,
                    include_news=include_news,
                )

                # Save to session state
                st.session_state.research_data = research_data
                st.session_state.research_completed = True

                # Display research results
                st.success(f"Gathered {len(research_data)} sources for your research!")
            else:
                st.warning("Please enter a research topic.")

        # Display research data
        if st.session_state.research_completed and st.session_state.research_data:
            # Display research sources
            st.subheader("2. Research Sources")

            with st.expander("View Research Sources", expanded=False):
                for i, item in enumerate(st.session_state.research_data):
                    st.markdown(f"**Source {i + 1}: {item.get('title', 'Untitled')}**")
                    st.markdown(f"URL: {item.get('url', 'Unknown')}")
                    st.markdown(f"Query: {item.get('query', 'Unknown query')}")
                    st.markdown("Content Preview:")
                    content = item.get("text", "")
                    st.markdown(
                        content[:300] + "..." if len(content) > 300 else content
                    )
                    st.markdown("---")

            # Research Summary Section
            st.subheader("3. Research Summary")

            if st.button("Generate Research Summary"):
                with st.spinner("Generating comprehensive research summary..."):
                    # Generate summary
                    summary = generate_research_summary(st.session_state.research_data)

                    # Display summary
                    st.markdown("### Research Summary")
                    st.markdown(summary)

            # Citations Section
            st.subheader("4. Citations")

            if st.button(f"Generate {citation_style} Citations"):
                with st.spinner(f"Generating {citation_style} citations..."):
                    # Generate citations
                    citations = generate_citations(
                        research_data=st.session_state.research_data,
                        style=citation_style,
                    )

                    # Display citations
                    st.markdown(f"### {citation_style} Citations")
                    st.markdown(citations)

            # Query Section
            st.subheader("5. Ask Research Questions")

            # Query input
            query, search_clicked = create_query_input(
                "Enter your research question:", button_text="Find Answer"
            )

            # Process query
            if search_clicked and query:
                with st.spinner("Researching your question..."):
                    # Generate answer
                    answer = answer_research_question(
                        query=query,
                        research_data=st.session_state.research_data,
                        top_k=5,
                    )

                    # Display answer
                    st.markdown("### Research Answer")
                    st.markdown(answer)

            elif search_clicked:
                st.warning("Please enter a research question.")

        elif not st.session_state.research_completed:
            st.info("Please specify a research topic and start the research process.")

    # Add footer
    create_footer()


if __name__ == "__main__":
    main()
