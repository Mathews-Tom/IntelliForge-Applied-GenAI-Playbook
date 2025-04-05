"""
AdaptiveQueryRAG: Contextual Strategy Selection
A Streamlit application for adaptive RAG with query-based strategy selection using Gemini 2.5 Pro.
"""

import os
import sys
import time
from enum import Enum, auto
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
from core.utils.retrieval_utils import (
    bm25_retrieval,
    embedding_retrieval,
    hybrid_retrieval,
    rerank_documents,
)
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


# Define query types
class QueryType(Enum):
    FACTUAL = auto()  # Factual lookup (who, what, when, where)
    SUMMARY = auto()  # Summarization (overview, explain, describe)
    COMPARISON = auto()  # Comparison (compare, contrast, differences)
    OPINION = auto()  # Opinion/analysis (why, how, implications)
    UNKNOWN = auto()  # Unknown or complex query type


# Function to load documents from a CSV file
@st.cache_data
def load_documents(file_path):
    """Load documents from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        if "text" not in df.columns:
            st.error("CSV file must contain a 'text' column.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return None


# Function to classify query type
def classify_query(query: str) -> Tuple[QueryType, Dict[str, float]]:
    """
    Classify the query type using Gemini.

    Args:
        query: The user query

    Returns:
        Tuple of (QueryType, confidence scores dictionary)
    """
    prompt = f"""
    Analyze this query and classify it into one of the following types:
    
    1. FACTUAL: Seeking specific facts or information (who, what, when, where)
    2. SUMMARY: Requesting an overview, explanation, or description
    3. COMPARISON: Asking to compare or contrast multiple items
    4. OPINION: Seeking analysis, implications, or reasoning (why, how)
    
    Query: "{query}"
    
    Return your classification as a JSON object with the following structure:
    {{
        "query_type": "FACTUAL|SUMMARY|COMPARISON|OPINION",
        "confidence": 0.0-1.0,
        "explanation": "Brief explanation of why this classification was chosen",
        "confidence_scores": {{
            "FACTUAL": 0.0-1.0,
            "SUMMARY": 0.0-1.0,
            "COMPARISON": 0.0-1.0,
            "OPINION": 0.0-1.0
        }}
    }}
    """

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)

        if response:
            # Try to parse the response as JSON
            try:
                import json

                result = json.loads(response.strip())

                # Extract query type and confidence scores
                query_type_str = result.get("query_type", "UNKNOWN")
                confidence_scores = result.get("confidence_scores", {})

                # Map string to enum
                query_type_map = {
                    "FACTUAL": QueryType.FACTUAL,
                    "SUMMARY": QueryType.SUMMARY,
                    "COMPARISON": QueryType.COMPARISON,
                    "OPINION": QueryType.OPINION,
                }

                query_type = query_type_map.get(query_type_str, QueryType.UNKNOWN)

                return query_type, confidence_scores
            except:
                # If parsing fails, return unknown type
                return QueryType.UNKNOWN, {
                    "FACTUAL": 0.25,
                    "SUMMARY": 0.25,
                    "COMPARISON": 0.25,
                    "OPINION": 0.25,
                }
        else:
            return QueryType.UNKNOWN, {
                "FACTUAL": 0.25,
                "SUMMARY": 0.25,
                "COMPARISON": 0.25,
                "OPINION": 0.25,
            }
    except Exception as e:
        st.error(f"Error classifying query: {e}")
        return QueryType.UNKNOWN, {
            "FACTUAL": 0.25,
            "SUMMARY": 0.25,
            "COMPARISON": 0.25,
            "OPINION": 0.25,
        }


# Function to get retrieval strategy based on query type
def get_retrieval_strategy(
    query_type: QueryType, confidence_scores: Dict[str, float]
) -> Dict[str, Any]:
    """
    Get the appropriate retrieval strategy based on query type.

    Args:
        query_type: The classified query type
        confidence_scores: Confidence scores for each query type

    Returns:
        Dictionary with strategy parameters
    """
    # Default strategy (balanced)
    default_strategy = {
        "method": "hybrid",
        "top_k": 5,
        "alpha": 0.5,  # Weight for BM25 vs embedding
        "description": "Balanced hybrid retrieval (default strategy)",
    }

    # Factual lookup strategy (precise, focused)
    factual_strategy = {
        "method": "embedding",
        "top_k": 3,
        "description": "Precise vector search for factual queries",
    }

    # Summary strategy (broader context)
    summary_strategy = {
        "method": "hybrid",
        "top_k": 7,
        "alpha": 0.3,  # Lower weight for BM25, higher for semantic
        "description": "Broader context retrieval for summary queries",
    }

    # Comparison strategy (targeted retrieval)
    comparison_strategy = {
        "method": "hybrid",
        "top_k": 8,
        "alpha": 0.5,
        "description": "Balanced retrieval with higher document count for comparison queries",
    }

    # Opinion/analysis strategy (diverse viewpoints)
    opinion_strategy = {
        "method": "bm25",
        "top_k": 6,
        "description": "Keyword-focused retrieval for opinion/analysis queries",
    }

    # Select strategy based on query type
    if query_type == QueryType.FACTUAL:
        return factual_strategy
    elif query_type == QueryType.SUMMARY:
        return summary_strategy
    elif query_type == QueryType.COMPARISON:
        return comparison_strategy
    elif query_type == QueryType.OPINION:
        return opinion_strategy
    else:
        # For unknown type, use weighted combination based on confidence scores
        # This creates a custom strategy that blends the different approaches

        # If confidence scores are missing or invalid, use default
        if not confidence_scores or sum(confidence_scores.values()) == 0:
            return default_strategy

        # Normalize confidence scores
        total = sum(confidence_scores.values())
        normalized_scores = {k: v / total for k, v in confidence_scores.items()}

        # Calculate weighted parameters
        top_k = int(
            normalized_scores.get("FACTUAL", 0) * factual_strategy["top_k"]
            + normalized_scores.get("SUMMARY", 0) * summary_strategy["top_k"]
            + normalized_scores.get("COMPARISON", 0) * comparison_strategy["top_k"]
            + normalized_scores.get("OPINION", 0) * opinion_strategy["top_k"]
        )

        # Ensure top_k is at least 3
        top_k = max(3, top_k)

        # Calculate alpha (BM25 weight)
        alpha = (
            normalized_scores.get("FACTUAL", 0)
            * 0.4  # Factual uses embedding but we blend
            + normalized_scores.get("SUMMARY", 0) * 0.3
            + normalized_scores.get("COMPARISON", 0) * 0.5
            + normalized_scores.get("OPINION", 0) * 0.7
        )

        return {
            "method": "hybrid",
            "top_k": top_k,
            "alpha": alpha,
            "description": "Custom blended strategy based on confidence scores",
        }


# Function to execute retrieval strategy
def execute_retrieval_strategy(
    query: str, documents: List[str], strategy: Dict[str, Any]
) -> List[Tuple[str, float]]:
    """
    Execute the selected retrieval strategy.

    Args:
        query: The user query
        documents: List of documents
        strategy: Retrieval strategy parameters

    Returns:
        List of (document, score) tuples
    """
    method = strategy.get("method", "hybrid")
    top_k = strategy.get("top_k", 5)

    if method == "bm25":
        return bm25_retrieval(query, documents, top_k=top_k)
    elif method == "embedding":
        return embedding_retrieval(query, documents, top_k=top_k)
    else:  # hybrid
        alpha = strategy.get("alpha", 0.5)
        return hybrid_retrieval(query, documents, top_k=top_k, alpha=alpha)


# Function to extract comparison items from query
def extract_comparison_items(query: str) -> List[str]:
    """
    Extract items being compared in a comparison query.

    Args:
        query: The comparison query

    Returns:
        List of items to compare
    """
    prompt = f"""
    Extract the specific items being compared in this query.
    
    Query: "{query}"
    
    Return a JSON array of the items being compared, like this:
    ["item1", "item2", ...]
    
    If no specific items are being compared, return an empty array.
    """

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)

        if response:
            # Try to parse the response as JSON
            try:
                import json

                items = json.loads(response.strip())
                if isinstance(items, list):
                    return items
            except:
                pass

        # Default case if parsing fails or no response
        return []
    except Exception as e:
        st.error(f"Error extracting comparison items: {e}")
        return []


# Function to handle comparison queries
def handle_comparison_query(
    query: str, documents: List[str], strategy: Dict[str, Any]
) -> List[Tuple[str, float]]:
    """
    Special handling for comparison queries.

    Args:
        query: The comparison query
        documents: List of documents
        strategy: Retrieval strategy parameters

    Returns:
        List of (document, score) tuples
    """
    # Extract items being compared
    items = extract_comparison_items(query)

    # If no specific items found, fall back to regular retrieval
    if not items:
        return execute_retrieval_strategy(query, documents, strategy)

    # Retrieve documents for each item
    all_retrieved_docs = []
    for item in items:
        # Create a query focused on this item
        item_query = f"{item} {query}"

        # Retrieve documents for this item
        item_docs = execute_retrieval_strategy(
            item_query,
            documents,
            {
                "method": strategy.get("method", "hybrid"),
                "top_k": 3,
            },  # Fewer docs per item
        )

        all_retrieved_docs.extend(item_docs)

    # Add some documents from the original query
    original_docs = execute_retrieval_strategy(
        query, documents, {"method": strategy.get("method", "hybrid"), "top_k": 3}
    )

    all_retrieved_docs.extend(original_docs)

    # Remove duplicates while preserving order
    seen = set()
    unique_docs = []
    for doc, score in all_retrieved_docs:
        if doc not in seen:
            seen.add(doc)
            unique_docs.append((doc, score))

    # Limit to top_k
    top_k = strategy.get("top_k", 5)
    return unique_docs[:top_k]


# Function to generate an answer
def generate_answer(query: str, context: str, query_type: QueryType) -> str:
    """
    Generate an answer based on the query, context, and query type.

    Args:
        query: User query
        context: Retrieved context
        query_type: The classified query type

    Returns:
        Generated answer
    """
    # Base prompt template
    base_prompt = f"""
    Answer the following question based on the provided context.
    If the context doesn't contain enough information to answer the question,
    say so clearly and provide what information you can.

    Question: {query}

    Context:
    {context}
    """

    # Customize prompt based on query type
    if query_type == QueryType.FACTUAL:
        prompt = (
            base_prompt
            + """
        
        Since this is a factual query, focus on providing precise, accurate information.
        Cite specific details from the context and be concise.
        """
        )
    elif query_type == QueryType.SUMMARY:
        prompt = (
            base_prompt
            + """
        
        Since this is a summary query, provide a comprehensive overview.
        Organize the information in a structured way and cover the main points.
        """
        )
    elif query_type == QueryType.COMPARISON:
        prompt = (
            base_prompt
            + """
        
        Since this is a comparison query, clearly identify the items being compared.
        Highlight similarities and differences in a structured format.
        Consider using a table or bullet points to organize the comparison.
        """
        )
    elif query_type == QueryType.OPINION:
        prompt = (
            base_prompt
            + """
        
        Since this is an opinion/analysis query, focus on explaining reasoning and implications.
        Consider different perspectives and provide a balanced analysis.
        Make it clear when you're presenting an interpretation versus a fact.
        """
        )
    else:
        prompt = base_prompt

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)
        return response if response else "Failed to generate an answer."
    except Exception as e:
        return f"Error generating answer: {e}"


# Main Streamlit app
def main():
    # Create header
    create_header(
        "AdaptiveQueryRAG: Contextual Strategy Selection",
        "An advanced RAG system that adapts its retrieval strategy based on query analysis, powered by Gemini 2.5 Pro.",
        icon="🎯",
    )

    # Create sidebar
    create_sidebar(
        "AdaptiveQueryRAG Settings",
        "Configure how the adaptive retrieval system works.",
    )

    # Sidebar settings
    with st.sidebar:
        st.header("Retrieval Settings")

        # Allow manual override of strategy
        override_strategy = st.checkbox(
            "Override automatic strategy selection", value=False
        )

        if override_strategy:
            retrieval_method = st.selectbox(
                "Retrieval Method:",
                options=["Hybrid", "BM25 Only", "Embedding Only"],
                index=0,
            )

            top_k = st.slider(
                "Number of documents to retrieve:", min_value=1, max_value=10, value=5
            )

            if retrieval_method == "Hybrid":
                alpha = st.slider(
                    "BM25 weight (α):",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
            else:
                alpha = 1.0 if retrieval_method == "BM25 Only" else 0.0

            manual_strategy = {
                "method": retrieval_method.lower().split()[0],
                "top_k": top_k,
                "alpha": alpha if retrieval_method == "Hybrid" else None,
                "description": f"Manual override: {retrieval_method} with top_k={top_k}",
            }

        show_process = st.checkbox("Show detailed process", value=True)

    # Main content
    if "GOOGLE_API_KEY" in os.environ:
        # File upload
        st.subheader("1. Upload Documents")

        uploaded_file = file_uploader(
            "Upload a CSV file with documents (must contain a 'text' column)",
            types=["csv"],
            key="document_upload",
        )

        use_sample_data = st.checkbox("Use sample data instead", value=True)

        # Load documents
        documents_df = None

        if uploaded_file:
            # Save the uploaded file
            file_path = save_uploaded_file(uploaded_file)
            documents_df = load_documents(file_path)
            use_sample_data = False
        elif use_sample_data:
            sample_path = Path(__file__).parent.parent / "data" / "sample_docs.csv"
            documents_df = load_documents(sample_path)

        if documents_df is not None:
            # Extract documents
            documents = documents_df["text"].tolist()

            # Display the number of documents
            st.info(f"Loaded {len(documents)} documents.")

            # Allow users to view the documents
            with st.expander("View Documents"):
                for i, doc in enumerate(documents):
                    title = (
                        documents_df["title"][i]
                        if "title" in documents_df.columns
                        else f"Document {i + 1}"
                    )
                    source = (
                        documents_df["source"][i]
                        if "source" in documents_df.columns
                        else "Unknown"
                    )
                    st.markdown(f"**{title}** (Source: {source})")
                    st.markdown(doc)
                    st.markdown("---")

            # Query input
            st.subheader("2. Ask a Question")
            query, search_clicked = create_query_input(
                "Enter your question:", button_text="Search with Adaptive Strategy"
            )

            # Process query
            if search_clicked and query:
                # Initialize processing steps
                steps = []

                # Step 1: Classify query
                with st.spinner("Analyzing query type..."):
                    start_time = time.time()

                    # Classify query
                    query_type, confidence_scores = classify_query(query)

                    classification_time = time.time() - start_time

                    # Add to steps
                    steps.append(
                        {
                            "title": "Query Classification",
                            "content": {
                                "Query Type": query_type.name,
                                "Confidence Scores": {
                                    k: f"{v:.2f}" for k, v in confidence_scores.items()
                                },
                                "Description": get_query_type_description(query_type),
                            },
                            "time": classification_time,
                        }
                    )

                # Step 2: Select retrieval strategy
                with st.spinner("Selecting retrieval strategy..."):
                    start_time = time.time()

                    # Get strategy
                    if override_strategy:
                        strategy = manual_strategy
                    else:
                        strategy = get_retrieval_strategy(query_type, confidence_scores)

                    strategy_time = time.time() - start_time

                    # Add to steps
                    steps.append(
                        {
                            "title": "Strategy Selection",
                            "content": {
                                "Method": strategy["method"].capitalize(),
                                "Top K": strategy["top_k"],
                                "Alpha (BM25 Weight)": f"{strategy.get('alpha', 'N/A')}",
                                "Description": strategy["description"],
                            },
                            "time": strategy_time,
                        }
                    )

                # Step 3: Execute retrieval
                with st.spinner("Retrieving documents..."):
                    start_time = time.time()

                    # Special handling for comparison queries
                    if query_type == QueryType.COMPARISON and not override_strategy:
                        retrieved_docs = handle_comparison_query(
                            query, documents, strategy
                        )
                        retrieval_method = "Comparison-specific retrieval"
                    else:
                        # Standard retrieval
                        retrieved_docs = execute_retrieval_strategy(
                            query, documents, strategy
                        )
                        retrieval_method = (
                            f"{strategy['method'].capitalize()} retrieval"
                        )

                    retrieval_time = time.time() - start_time

                    # Add to steps
                    steps.append(
                        {
                            "title": "Document Retrieval",
                            "content": f"Retrieved {len(retrieved_docs)} documents using {retrieval_method} in {retrieval_time:.2f} seconds.",
                            "time": retrieval_time,
                        }
                    )

                # Step 4: Generate answer
                with st.spinner("Generating answer..."):
                    start_time = time.time()

                    # Prepare context
                    context_docs = [doc for doc, _ in retrieved_docs]
                    if isinstance(context_docs[0], dict):
                        context = "\n\n".join(
                            [doc.get("text", "") for doc in context_docs]
                        )
                    else:
                        context = "\n\n".join(context_docs)

                    # Generate answer
                    answer = generate_answer(query, context, query_type)

                    generation_time = time.time() - start_time

                    # Add to steps
                    steps.append(
                        {
                            "title": "Answer Generation",
                            "content": "Generated answer using query-specific prompt template.",
                            "time": generation_time,
                        }
                    )

                # Display results
                st.subheader("Answer")
                st.markdown(answer)

                # Display retrieved documents
                st.subheader("Retrieved Documents")

                # Convert to format expected by display_documents
                docs_with_scores = []
                for doc, score in retrieved_docs:
                    if isinstance(doc, dict):
                        doc_with_score = doc.copy()
                        doc_with_score["score"] = score
                        docs_with_scores.append(doc_with_score)
                    else:
                        docs_with_scores.append({"text": doc, "score": score})

                scores = [score for _, score in retrieved_docs]
                display_documents(docs_with_scores, scores=scores)

                # Display processing steps if enabled
                if show_process:
                    display_processing_steps(steps)

            elif search_clicked:
                st.warning("Please enter a question.")
        else:
            st.warning(
                "Please upload a CSV file with documents or use the sample data."
            )

    # Add footer
    create_footer()


def get_query_type_description(query_type: QueryType) -> str:
    """Get a description of the query type."""
    if query_type == QueryType.FACTUAL:
        return "Factual queries seek specific information (who, what, when, where)."
    elif query_type == QueryType.SUMMARY:
        return "Summary queries request overviews, explanations, or descriptions."
    elif query_type == QueryType.COMPARISON:
        return "Comparison queries ask to compare or contrast multiple items."
    elif query_type == QueryType.OPINION:
        return "Opinion queries seek analysis, implications, or reasoning (why, how)."
    else:
        return "Complex or ambiguous query that doesn't fit a single category."


if __name__ == "__main__":
    main()
