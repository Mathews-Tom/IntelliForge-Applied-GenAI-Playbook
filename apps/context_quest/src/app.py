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
)

# Initialize session state for embeddings cache
if "embeddings_cache" not in st.session_state:
    st.session_state.embeddings_cache = {}


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
    from a knowledge base, combining keyword-based and semantic search methods.
    """)

# Main content
if "GOOGLE_API_KEY" in os.environ:
    # Load context data
    context_data = load_context_data(
        Path(__file__).parent.parent / "data" / "context.csv"
    )

    if context_data is not None:
        # Extract documents
        documents = context_data["text"].tolist()

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
                        retrieved_docs = bm25_retrieval(query, documents, top_k=top_k)
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
