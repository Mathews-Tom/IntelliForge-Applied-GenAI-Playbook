"""
CompetitiveAnalysisAgent: Web-Based Competitor Intelligence
A Streamlit application for gathering and analyzing competitor information using Gemini 2.5 Pro.
"""

import json
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


# Function to extract company name from URL
def extract_company_name(url: str) -> str:
    """
    Extract company name from URL.

    Args:
        url: Website URL

    Returns:
        Company name
    """
    # Remove protocol and www
    domain = (
        url.lower().replace("https://", "").replace("http://", "").replace("www.", "")
    )

    # Get domain without TLD
    domain = domain.split("/")[0]
    company = domain.split(".")[0]

    # Handle special cases
    if "." in company:
        company = company.split(".")[-1]

    # Capitalize
    return company.capitalize()


# Function to crawl competitor websites
def crawl_competitor_websites(
    competitors: List[Dict[str, str]],
    max_pages_per_competitor: int = 10,
    max_depth: int = 2,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Crawl competitor websites.

    Args:
        competitors: List of competitor dictionaries with 'name' and 'url' keys
        max_pages_per_competitor: Maximum number of pages to crawl per competitor
        max_depth: Maximum depth of crawling

    Returns:
        Dictionary mapping competitor names to lists of crawled content
    """
    # Check if crawl4ai is installed
    crawl4ai_available = check_crawl4ai_installed()

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "data" / "crawled_content"
    os.makedirs(output_dir, exist_ok=True)

    # Crawl each competitor website
    competitor_data = {}

    for competitor in competitors:
        name = competitor["name"]
        url = competitor["url"]

        with st.spinner(f"Crawling {name}'s website ({url})..."):
            if crawl4ai_available:
                # Use crawl4ai
                results = crawl_url(
                    url=url,
                    max_pages=max_pages_per_competitor,
                    max_depth=max_depth,
                    output_dir=str(output_dir / name),
                )
            else:
                # Use mock implementation
                results = mock_crawl_url(
                    url=url, max_pages=max_pages_per_competitor, max_depth=max_depth
                )

            # Add competitor name to each result
            for result in results:
                result["competitor"] = name

            competitor_data[name] = results

            # Save results
            save_path = output_dir / f"{name}_data.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    return competitor_data


# Function to analyze competitor data
def analyze_competitor_data(
    competitor_data: Dict[str, List[Dict[str, Any]]], analysis_type: str
) -> str:
    """
    Analyze competitor data.

    Args:
        competitor_data: Dictionary mapping competitor names to lists of crawled content
        analysis_type: Type of analysis to perform

    Returns:
        Analysis results as a string
    """
    # Prepare competitor information
    competitor_info = {}

    for name, data in competitor_data.items():
        # Combine all text from this competitor
        all_text = "\n\n".join([item.get("text", "") for item in data])

        # Add to competitor info
        competitor_info[name] = {
            "text": all_text[:10000],  # Limit text length to avoid token limits
            "url_count": len(data),
            "urls": [item.get("url", "") for item in data][:5],  # Include up to 5 URLs
        }

    # Create prompt based on analysis type
    if analysis_type == "overview":
        prompt = f"""
        Provide a comprehensive overview of the following competitors based on their website content.
        For each competitor, include:
        1. Company overview and main business areas
        2. Key products or services
        3. Target audience/market
        4. Unique selling propositions or differentiators
        
        Competitor information:
        {json.dumps(competitor_info, indent=2)}
        
        Format your response as a well-structured report with clear headings and bullet points.
        """

    elif analysis_type == "product_comparison":
        prompt = f"""
        Compare the products/services of the following competitors based on their website content.
        Include:
        1. Key product/service categories for each competitor
        2. Feature comparison across similar products
        3. Pricing information (if available)
        4. Positioning and messaging differences
        
        Competitor information:
        {json.dumps(competitor_info, indent=2)}
        
        Format your response as a comparative analysis with tables or structured sections.
        """

    elif analysis_type == "marketing_analysis":
        prompt = f"""
        Analyze the marketing approach of the following competitors based on their website content.
        Include:
        1. Key messaging and value propositions
        2. Content strategy and types of content
        3. Visual branding elements
        4. Call-to-actions and conversion strategies
        
        Competitor information:
        {json.dumps(competitor_info, indent=2)}
        
        Format your response as a marketing analysis report with clear sections for each competitor and comparative insights.
        """

    else:  # custom query
        prompt = f"""
        Based on the website content of the following competitors, provide insights and analysis.
        
        Competitor information:
        {json.dumps(competitor_info, indent=2)}
        
        Provide a comprehensive analysis that covers key aspects of these competitors.
        Format your response as a well-structured report with clear headings and bullet points.
        """

    # Generate analysis
    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)
        return response if response else "Failed to generate analysis."
    except Exception as e:
        return f"Error generating analysis: {e}"


# Function to answer competitive intelligence questions
def answer_competitive_question(
    query: str, competitor_data: Dict[str, List[Dict[str, Any]]], top_k: int = 5
) -> str:
    """
    Answer a competitive intelligence question.

    Args:
        query: User query
        competitor_data: Dictionary mapping competitor names to lists of crawled content
        top_k: Number of chunks to retrieve per competitor

    Returns:
        Answer to the query
    """
    # Flatten all competitor data into a list of documents
    all_documents = []

    for name, data in competitor_data.items():
        for item in data:
            # Create a document with text and metadata
            document = {
                "text": item.get("text", ""),
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "competitor": name,
            }
            all_documents.append(document)

    # Extract text from documents
    texts = [doc["text"] for doc in all_documents]

    # Retrieve relevant chunks
    retrieved_docs = embedding_retrieval(query, texts, top_k=top_k)

    # Map retrieved texts back to full documents
    retrieved_documents = []
    for text, score in retrieved_docs:
        for doc in all_documents:
            if doc["text"] == text:
                doc_with_score = doc.copy()
                doc_with_score["score"] = score
                retrieved_documents.append(doc_with_score)
                break

    # Group retrieved documents by competitor
    documents_by_competitor = {}
    for doc in retrieved_documents:
        competitor = doc["competitor"]
        if competitor not in documents_by_competitor:
            documents_by_competitor[competitor] = []
        documents_by_competitor[competitor].append(doc)

    # Prepare context for each competitor
    competitor_contexts = {}
    for competitor, docs in documents_by_competitor.items():
        # Sort by score
        docs.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Create context
        context = ""
        for doc in docs:
            context += f"Source: {doc.get('url', 'Unknown URL')}\n"
            context += f"Title: {doc.get('title', 'Unknown Title')}\n"
            context += f"Content: {doc.get('text', '')}\n\n"

        competitor_contexts[competitor] = context

    # Create prompt
    prompt = f"""
    Answer the following competitive intelligence question based on the provided information from competitor websites.
    
    Question: {query}
    
    Competitor Information:
    {json.dumps(competitor_contexts, indent=2)}
    
    Your task is to:
    1. Provide a comprehensive answer that addresses the question directly
    2. Include specific information from each relevant competitor
    3. Make comparisons between competitors when appropriate
    4. Cite sources (URLs) for key information
    5. Acknowledge any information gaps or uncertainties
    
    Format your answer as a well-structured response with clear sections and bullet points where appropriate.
    """

    # Generate answer
    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)
        return response if response else "Failed to generate an answer."
    except Exception as e:
        return f"Error generating answer: {e}"


# Main Streamlit app
def main():
    # Create header
    create_header(
        "CompetitiveAnalysisAgent: Web-Based Competitor Intelligence",
        "An advanced application for gathering and analyzing competitor information from websites, powered by Gemini 2.5 Pro.",
        icon="🔍",
    )

    # Create sidebar
    create_sidebar(
        "CompetitiveAnalysisAgent Settings",
        "Configure how the competitive analysis system works.",
    )

    # Sidebar settings
    with st.sidebar:
        st.header("Crawling Settings")

        max_pages = st.slider(
            "Maximum Pages per Competitor:",
            min_value=1,
            max_value=30,
            value=10,
            help="Maximum number of pages to crawl per competitor",
        )

        max_depth = st.slider(
            "Maximum Depth:",
            min_value=1,
            max_value=3,
            value=2,
            help="Maximum depth of crawling",
        )

        st.header("Analysis Settings")

        top_k = st.slider(
            "Number of chunks to retrieve:",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of text chunks to retrieve per competitor",
        )

        show_process = st.checkbox("Show detailed process", value=True)

    # Main content
    if "GOOGLE_API_KEY" in os.environ:
        # Check dependencies
        has_crawl4ai = check_dependencies()

        # Initialize session state
        if "competitor_data" not in st.session_state:
            st.session_state.competitor_data = {}

        if "crawling_completed" not in st.session_state:
            st.session_state.crawling_completed = False

        # Competitor Specification Section
        st.subheader("1. Specify Competitors")

        # Input for competitor URLs
        competitor_urls = st.text_area(
            "Enter competitor website URLs (one per line):",
            placeholder="https://example-competitor.com\nhttps://another-competitor.com",
            help="Enter the main website URLs of competitors you want to analyze",
        )

        # Parse competitor URLs
        competitors = []
        if competitor_urls:
            for url in competitor_urls.strip().split("\n"):
                url = url.strip()
                if url:
                    name = extract_company_name(url)
                    competitors.append({"name": name, "url": url})

        # Display parsed competitors
        if competitors:
            st.write("Detected competitors:")
            cols = st.columns(min(3, len(competitors)))
            for i, competitor in enumerate(competitors):
                with cols[i % len(cols)]:
                    st.info(f"**{competitor['name']}**\n{competitor['url']}")

        # Crawl button
        if st.button("Start Competitor Analysis"):
            if competitors:
                # Crawl competitor websites
                competitor_data = crawl_competitor_websites(
                    competitors=competitors,
                    max_pages_per_competitor=max_pages,
                    max_depth=max_depth,
                )

                # Save to session state
                st.session_state.competitor_data = competitor_data
                st.session_state.crawling_completed = True

                # Display crawling results
                st.success(f"Analyzed {len(competitor_data)} competitors successfully!")
            else:
                st.warning("Please enter at least one competitor URL.")

        # Display crawled content
        if st.session_state.crawling_completed and st.session_state.competitor_data:
            # Display competitor data
            st.subheader("2. Competitor Data")

            tabs = st.tabs([name for name in st.session_state.competitor_data.keys()])

            for i, (name, data) in enumerate(st.session_state.competitor_data.items()):
                with tabs[i]:
                    st.write(f"Crawled {len(data)} pages from {name}'s website")

                    with st.expander("View Crawled Pages", expanded=False):
                        for j, item in enumerate(data):
                            st.markdown(
                                f"**Page {j + 1}: {item.get('title', 'Untitled')}**"
                            )
                            st.markdown(f"URL: {item.get('url', 'Unknown')}")
                            st.markdown("Content Preview:")
                            content = item.get("text", "")
                            st.markdown(
                                content[:300] + "..." if len(content) > 300 else content
                            )
                            st.markdown("---")

            # Analysis Section
            st.subheader("3. Competitive Analysis")

            analysis_type = st.selectbox(
                "Select Analysis Type:",
                options=[
                    "Overview",
                    "Product Comparison",
                    "Marketing Analysis",
                    "Custom Query",
                ],
                format_func=lambda x: x.replace("_", " ").title(),
            )

            if st.button("Generate Analysis"):
                with st.spinner("Generating competitive analysis..."):
                    # Generate analysis
                    analysis = analyze_competitor_data(
                        competitor_data=st.session_state.competitor_data,
                        analysis_type=analysis_type.lower(),
                    )

                    # Display analysis
                    st.markdown("### Analysis Results")
                    st.markdown(analysis)

            # Query Section
            st.subheader("4. Ask Competitive Intelligence Questions")

            # Query input
            query, search_clicked = create_query_input(
                "Enter your competitive intelligence question:",
                button_text="Get Competitive Insights",
            )

            # Process query
            if search_clicked and query:
                with st.spinner("Analyzing competitive intelligence..."):
                    # Generate answer
                    answer = answer_competitive_question(
                        query=query,
                        competitor_data=st.session_state.competitor_data,
                        top_k=top_k,
                    )

                    # Display answer
                    st.markdown("### Competitive Intelligence Answer")
                    st.markdown(answer)

            elif search_clicked:
                st.warning("Please enter a question.")

        elif not st.session_state.crawling_completed:
            st.info("Please specify competitors and start the analysis process.")

    # Add footer
    create_footer()


if __name__ == "__main__":
    main()
