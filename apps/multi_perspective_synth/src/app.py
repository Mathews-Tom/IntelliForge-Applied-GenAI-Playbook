"""
MultiPerspectiveSynth: Synthesizing Diverse Sources
A Streamlit application for synthesizing information from multiple, potentially conflicting sources using Gemini 2.5 Pro.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from core.llm.gemini_utils import GeminiModelType, generate_content
from core.utils.data_helpers import load_documents_from_csv
from core.utils.file_io import save_uploaded_file
from core.utils.retrieval_utils import embedding_retrieval
from core.utils.ui_helpers import (
    create_footer,
    create_header,
    create_query_input,
    create_sidebar,
    display_documents,
    display_processing_steps,
    file_uploader,
    timed_operation,
)


# Function to load documents from a CSV file
@st.cache_data
def load_documents(file_path, source_name=None):
    """
    Load documents from a CSV file.

    Args:
        file_path: Path to the CSV file
        source_name: Optional name to override the source

    Returns:
        DataFrame with documents
    """
    try:
        df = pd.read_csv(file_path)
        if "text" not in df.columns:
            st.error(f"CSV file {file_path} must contain a 'text' column.")
            return None

        # Add source name if provided
        if source_name:
            df["source"] = source_name

        # Ensure source column exists
        if "source" not in df.columns:
            df["source"] = os.path.basename(file_path)

        # Ensure perspective column exists
        if "perspective" not in df.columns:
            df["perspective"] = "unknown"

        return df
    except Exception as e:
        st.error(f"Error loading documents from {file_path}: {e}")
        return None


# Function to retrieve documents from multiple sources
def multi_source_retrieval(
    query: str,
    documents_by_source: Dict[str, List[Dict[str, Any]]],
    top_k_per_source: int = 2,
    top_k_total: int = 8,
) -> List[Dict[str, Any]]:
    """
    Retrieve documents from multiple sources, ensuring representation from each source.

    Args:
        query: The user query
        documents_by_source: Dictionary mapping source names to lists of document dictionaries
        top_k_per_source: Number of documents to retrieve from each source
        top_k_total: Maximum total number of documents to retrieve

    Returns:
        List of retrieved document dictionaries with scores
    """
    all_retrieved_docs = []

    # Retrieve from each source
    for source, docs in documents_by_source.items():
        # Extract text from documents
        texts = [doc["text"] for doc in docs]

        # Retrieve documents
        retrieved = embedding_retrieval(query, texts, top_k=top_k_per_source)

        # Convert to document dictionaries with scores
        for i, (text, score) in enumerate(retrieved):
            # Find the original document
            for doc in docs:
                if doc["text"] == text:
                    # Create a copy with the score
                    doc_with_score = doc.copy()
                    doc_with_score["score"] = score
                    all_retrieved_docs.append(doc_with_score)
                    break

    # Sort by score and limit to top_k_total
    all_retrieved_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_retrieved_docs[:top_k_total]


# Function to identify perspectives in retrieved documents
def identify_perspectives(
    query: str, retrieved_docs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Identify different perspectives in the retrieved documents.

    Args:
        query: The user query
        retrieved_docs: List of retrieved document dictionaries

    Returns:
        Dictionary with perspective analysis
    """
    # Prepare documents text with source and perspective information
    docs_text = ""
    for i, doc in enumerate(retrieved_docs):
        source = doc.get("source", "Unknown")
        perspective = doc.get("perspective", "Unknown")
        docs_text += f"Document {i + 1} (Source: {source}, Perspective: {perspective}):\n{doc['text']}\n\n"

    prompt = f"""
    Analyze the different perspectives present in these documents regarding the query.
    
    Query: {query}
    
    Documents:
    {docs_text}
    
    Identify the main perspectives or viewpoints represented across these documents.
    For each perspective, list the key points made and which documents support it.
    Then, identify areas of agreement and disagreement across the perspectives.
    
    Return your analysis as a JSON object with the following structure:
    {{
        "perspectives": [
            {{
                "name": "Name of perspective 1",
                "key_points": ["Point 1", "Point 2", ...],
                "supporting_documents": [1, 3, ...] // Document numbers
            }},
            ...
        ],
        "agreements": ["Area of agreement 1", "Area of agreement 2", ...],
        "disagreements": ["Area of disagreement 1", "Area of disagreement 2", ...]
    }}
    """

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)

        if response:
            # Try to parse the response as JSON
            try:
                import json

                result = json.loads(response.strip())
                return result
            except:
                # If parsing fails, return a simple result
                return {
                    "perspectives": [
                        {
                            "name": "Perspective from the documents",
                            "key_points": [
                                "The documents contain various perspectives on the topic."
                            ],
                            "supporting_documents": list(
                                range(1, len(retrieved_docs) + 1)
                            ),
                        }
                    ],
                    "agreements": ["Failed to parse perspective analysis."],
                    "disagreements": ["Failed to parse perspective analysis."],
                }
        else:
            return {
                "perspectives": [
                    {
                        "name": "Unknown",
                        "key_points": ["Failed to analyze perspectives."],
                        "supporting_documents": [],
                    }
                ],
                "agreements": ["Failed to analyze perspectives."],
                "disagreements": ["Failed to analyze perspectives."],
            }
    except Exception as e:
        st.error(f"Error identifying perspectives: {e}")
        return {
            "perspectives": [
                {
                    "name": "Error",
                    "key_points": [f"Error analyzing perspectives: {e}"],
                    "supporting_documents": [],
                }
            ],
            "agreements": ["Error in analysis."],
            "disagreements": ["Error in analysis."],
        }


# Function to generate a synthesized answer
def generate_synthesis(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    perspective_analysis: Dict[str, Any],
) -> str:
    """
    Generate a synthesized answer that incorporates multiple perspectives.

    Args:
        query: The user query
        retrieved_docs: List of retrieved document dictionaries
        perspective_analysis: Perspective analysis results

    Returns:
        Synthesized answer
    """
    # Prepare documents text with source information
    docs_text = ""
    for i, doc in enumerate(retrieved_docs):
        source = doc.get("source", "Unknown")
        perspective = doc.get("perspective", "Unknown")
        docs_text += f"Document {i + 1} (Source: {source}, Perspective: {perspective}):\n{doc['text']}\n\n"

    # Prepare perspectives text
    perspectives_text = ""
    for i, perspective in enumerate(perspective_analysis.get("perspectives", [])):
        name = perspective.get("name", f"Perspective {i + 1}")
        key_points = perspective.get("key_points", [])
        key_points_text = "\n".join([f"- {point}" for point in key_points])
        supporting_docs = perspective.get("supporting_documents", [])
        supporting_docs_text = ", ".join([str(doc) for doc in supporting_docs])

        perspectives_text += f"Perspective: {name}\nKey Points:\n{key_points_text}\nSupporting Documents: {supporting_docs_text}\n\n"

    # Prepare agreements and disagreements
    agreements = perspective_analysis.get("agreements", [])
    agreements_text = "\n".join([f"- {agreement}" for agreement in agreements])

    disagreements = perspective_analysis.get("disagreements", [])
    disagreements_text = "\n".join(
        [f"- {disagreement}" for disagreement in disagreements]
    )

    prompt = f"""
    Synthesize an answer to the following query based on multiple perspectives from the provided documents.
    
    Query: {query}
    
    Documents:
    {docs_text}
    
    Perspective Analysis:
    {perspectives_text}
    
    Areas of Agreement:
    {agreements_text}
    
    Areas of Disagreement:
    {disagreements_text}
    
    Your task is to synthesize a comprehensive answer that:
    1. Addresses the query directly
    2. Presents multiple perspectives fairly and accurately
    3. Clearly indicates areas of agreement and disagreement
    4. Attributes information to specific sources using inline citations (e.g., [Source: X])
    5. Avoids taking a side or presenting one perspective as superior
    6. Helps the reader understand the full range of viewpoints on this topic
    
    Structure your answer with:
    - An introduction that frames the topic and acknowledges multiple perspectives exist
    - A balanced presentation of different viewpoints with source attribution
    - A summary of areas of agreement and disagreement
    - A conclusion that synthesizes the information without favoring any perspective
    """

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)
        return response if response else "Failed to generate a synthesized answer."
    except Exception as e:
        return f"Error generating synthesized answer: {e}"


# Main Streamlit app
def main():
    # Create header
    create_header(
        "MultiPerspectiveSynth: Synthesizing Diverse Sources",
        "An advanced RAG system that retrieves and synthesizes information from multiple, potentially conflicting sources, powered by Gemini 2.5 Pro.",
        icon="🔍",
    )

    # Create sidebar
    create_sidebar(
        "MultiPerspectiveSynth Settings",
        "Configure how the multi-perspective synthesis system works.",
    )

    # Sidebar settings
    with st.sidebar:
        st.header("Retrieval Settings")

        top_k_per_source = st.slider(
            "Documents per source:",
            min_value=1,
            max_value=5,
            value=2,
            help="Number of documents to retrieve from each source",
        )

        top_k_total = st.slider(
            "Total documents:",
            min_value=3,
            max_value=15,
            value=8,
            help="Maximum total number of documents to retrieve",
        )

        show_process = st.checkbox("Show detailed process", value=True)

    # Main content
    if "GOOGLE_API_KEY" in os.environ:
        # File upload
        st.subheader("1. Upload Documents from Different Sources")

        # Multiple file upload
        uploaded_files = file_uploader(
            "Upload CSV files with documents from different sources (must contain 'text' and ideally 'source' columns)",
            types=["csv"],
            multiple=True,
            key="document_upload",
        )

        use_sample_data = st.checkbox("Use sample data instead", value=True)

        # Initialize documents by source
        documents_by_source = {}
        all_documents = []

        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Save the uploaded file
                file_path = save_uploaded_file(uploaded_file)

                # Load documents
                source_name = os.path.splitext(uploaded_file.name)[0]
                df = load_documents(file_path, source_name)

                if df is not None:
                    # Convert to list of dictionaries
                    docs = df.to_dict("records")
                    documents_by_source[source_name] = docs
                    all_documents.extend(docs)

            use_sample_data = False

        # Use sample data if requested
        if use_sample_data:
            sample_files = [
                "perspective_a.csv",
                "perspective_b.csv",
                "perspective_c.csv",
            ]

            for file_name in sample_files:
                file_path = Path(__file__).parent.parent / "data" / file_name
                source_name = os.path.splitext(file_name)[0]
                df = load_documents(file_path, source_name)

                if df is not None:
                    # Convert to list of dictionaries
                    docs = df.to_dict("records")
                    documents_by_source[source_name] = docs
                    all_documents.extend(docs)

        # Display loaded documents
        if documents_by_source:
            total_docs = len(all_documents)
            num_sources = len(documents_by_source)
            st.info(f"Loaded {total_docs} documents from {num_sources} sources.")

            # Allow users to view the documents by source
            for source, docs in documents_by_source.items():
                with st.expander(
                    f"View Documents from {source} ({len(docs)} documents)"
                ):
                    for i, doc in enumerate(docs):
                        title = doc.get("title", f"Document {i + 1}")
                        perspective = doc.get("perspective", "Unknown")
                        st.markdown(f"**{title}** (Perspective: {perspective})")
                        st.markdown(doc["text"])
                        st.markdown("---")

            # Query input
            st.subheader("2. Ask a Question")
            query, search_clicked = create_query_input(
                "Enter your question:", button_text="Search Across Perspectives"
            )

            # Process query
            if search_clicked and query:
                # Initialize processing steps
                steps = []

                # Step 1: Multi-source retrieval
                with st.spinner("Retrieving documents from multiple sources..."):
                    start_time = time.time()

                    # Retrieve documents
                    retrieved_docs = multi_source_retrieval(
                        query,
                        documents_by_source,
                        top_k_per_source=top_k_per_source,
                        top_k_total=top_k_total,
                    )

                    retrieval_time = time.time() - start_time

                    # Count documents by source
                    source_counts = {}
                    for doc in retrieved_docs:
                        source = doc.get("source", "Unknown")
                        source_counts[source] = source_counts.get(source, 0) + 1

                    # Add to steps
                    steps.append(
                        {
                            "title": "Multi-Source Retrieval",
                            "content": {
                                "Total Documents": len(retrieved_docs),
                                "Documents by Source": source_counts,
                            },
                            "time": retrieval_time,
                        }
                    )

                # Step 2: Perspective identification
                with st.spinner("Identifying perspectives..."):
                    start_time = time.time()

                    # Identify perspectives
                    perspective_analysis = identify_perspectives(query, retrieved_docs)

                    perspective_time = time.time() - start_time

                    # Add to steps
                    steps.append(
                        {
                            "title": "Perspective Identification",
                            "content": {
                                "Perspectives Identified": len(
                                    perspective_analysis.get("perspectives", [])
                                ),
                                "Areas of Agreement": len(
                                    perspective_analysis.get("agreements", [])
                                ),
                                "Areas of Disagreement": len(
                                    perspective_analysis.get("disagreements", [])
                                ),
                            },
                            "time": perspective_time,
                        }
                    )

                # Step 3: Generate synthesized answer
                with st.spinner("Generating synthesized answer..."):
                    start_time = time.time()

                    # Generate synthesis
                    synthesis = generate_synthesis(
                        query, retrieved_docs, perspective_analysis
                    )

                    synthesis_time = time.time() - start_time

                    # Add to steps
                    steps.append(
                        {
                            "title": "Synthesis Generation",
                            "content": "Generated a synthesized answer incorporating multiple perspectives.",
                            "time": synthesis_time,
                        }
                    )

                # Display results
                st.subheader("Synthesized Answer")
                st.markdown(synthesis)

                # Display perspective analysis
                st.subheader("Perspective Analysis")

                # Display perspectives
                perspectives = perspective_analysis.get("perspectives", [])
                if perspectives:
                    for i, perspective in enumerate(perspectives):
                        with st.expander(
                            f"Perspective: {perspective.get('name', f'Perspective {i + 1}')}"
                        ):
                            # Key points
                            st.markdown("**Key Points:**")
                            for point in perspective.get("key_points", []):
                                st.markdown(f"- {point}")

                            # Supporting documents
                            st.markdown("**Supporting Documents:**")
                            doc_indices = perspective.get("supporting_documents", [])
                            for idx in doc_indices:
                                if 1 <= idx <= len(retrieved_docs):
                                    doc = retrieved_docs[idx - 1]
                                    title = doc.get("title", f"Document {idx}")
                                    source = doc.get("source", "Unknown")
                                    st.markdown(f"- {title} (Source: {source})")

                # Display agreements and disagreements
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Areas of Agreement:**")
                    for agreement in perspective_analysis.get("agreements", []):
                        st.markdown(f"- {agreement}")

                with col2:
                    st.markdown("**Areas of Disagreement:**")
                    for disagreement in perspective_analysis.get("disagreements", []):
                        st.markdown(f"- {disagreement}")

                # Display retrieved documents
                st.subheader("Retrieved Documents")
                display_documents(retrieved_docs)

                # Display processing steps if enabled
                if show_process:
                    display_processing_steps(steps)

            elif search_clicked:
                st.warning("Please enter a question.")
        else:
            st.warning("Please upload CSV files with documents or use the sample data.")

    # Add footer
    create_footer()


if __name__ == "__main__":
    main()
