"""
ContextQuest: Hybrid Retrieval
A Streamlit application for hybrid retrieval-augmented generation using Gemini 2.5 Pro.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from core.llm.gemini_utils import GeminiModelType, generate_content

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


# Function to tokenize text for BM25
def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 retrieval.

    Args:
        text: Text to tokenize.

    Returns:
        List of tokens.
    """
    return text.lower().split()


# Function to get embeddings using Gemini
def get_embeddings(text: str, cache_key: str | None = None) -> np.ndarray | None:
    """
    Get embeddings for text using Gemini.

    Args:
        text: Text to embed.
        cache_key: Optional key for caching.

    Returns:
        Embedding vector as numpy array or None if generation fails.
    """
    # Check cache first if a cache_key is provided
    if cache_key and cache_key in st.session_state.embeddings_cache:
        return st.session_state.embeddings_cache[cache_key]

    # Create a prompt for Gemini to generate an embedding representation
    prompt = f"""
    Generate a numerical embedding representation for the following text.
    The embedding should capture the semantic meaning of the text.
    Return only the embedding as a comma-separated list of 768 floating-point numbers.

    Text: {text}
    """

    try:
        # This is a simplified approach - in a real application, you would use
        # a dedicated embedding API rather than generating text and parsing it
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)

        if response:
            # Extract numbers from the response
            # This is a simplified parsing approach
            numbers_text = response.strip().split(",")
            embedding = np.array([float(num) for num in numbers_text])

            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)

            # Cache the result if a cache_key is provided
            if cache_key:
                st.session_state.embeddings_cache[cache_key] = embedding

            return embedding
        else:
            st.error("Failed to generate embedding.")
            return None
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None


# Function for BM25 retrieval
def bm25_retrieval(
    query: str, documents: list[str], top_k: int = 3
) -> list[tuple[str, float]]:
    """
    Retrieve documents using BM25 algorithm.

    Args:
        query: Query text.
        documents: List of documents.
        top_k: Number of documents to retrieve.

    Returns:
        List of (document, score) tuples.
    """
    # Tokenize documents
    tokenized_docs = [tokenize(doc) for doc in documents]

    # Create BM25 model
    bm25 = BM25Okapi(tokenized_docs)

    # Tokenize query
    tokenized_query = tokenize(query)

    # Get scores
    scores = bm25.get_scores(tokenized_query)

    # Get top-k documents
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Return documents and scores
    return [(documents[i], scores[i]) for i in top_indices]


# Function for embedding-based retrieval
def embedding_retrieval(
    query: str,
    documents: list[str],
    document_embeddings: list[np.ndarray] | None = None,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    Retrieve documents using embedding-based similarity.

    Args:
        query: Query text.
        documents: List of documents.
        document_embeddings: Pre-computed document embeddings (optional).
        top_k: Number of documents to retrieve.

    Returns:
        List of (document, score) tuples.
    """
    # Get query embedding
    query_embedding = get_embeddings(query)

    if query_embedding is None:
        return []

    # If document embeddings are not provided, compute them
    if document_embeddings is None:
        document_embeddings = []
        for i, doc in enumerate(documents):
            doc_embedding = get_embeddings(doc, cache_key=f"doc_{i}")
            if doc_embedding is not None:
                document_embeddings.append(doc_embedding)
            else:
                document_embeddings.append(
                    np.zeros(768)
                )  # Placeholder for failed embeddings

    # Compute similarities
    similarities = [
        cosine_similarity(query_embedding.reshape(1, -1), doc_embedding.reshape(1, -1))[
            0
        ][0]
        for doc_embedding in document_embeddings
    ]

    # Get top-k documents
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Return documents and scores
    return [(documents[i], similarities[i]) for i in top_indices]


# Function for hybrid retrieval
def hybrid_retrieval(
    query: str,
    documents: list[str],
    document_embeddings: list[np.ndarray] | None = None,
    top_k: int = 3,
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Retrieve documents using a hybrid of BM25 and embedding-based similarity.

    Args:
        query: Query text.
        documents: List of documents.
        document_embeddings: Pre-computed document embeddings (optional).
        top_k: Number of documents to retrieve.
        alpha: Weight for BM25 scores (1-alpha for embedding scores).

    Returns:
        List of (document, score) tuples.
    """
    # Get BM25 results
    bm25_results = bm25_retrieval(query, documents, top_k=len(documents))

    # Get embedding results
    embedding_results = embedding_retrieval(
        query, documents, document_embeddings, top_k=len(documents)
    )

    if not embedding_results:
        return bm25_results[:top_k]

    # Create dictionaries for easy lookup
    bm25_dict = {doc: score for doc, score in bm25_results}
    embedding_dict = {doc: score for doc, score in embedding_results}

    # Normalize scores
    max_bm25 = max(bm25_dict.values()) if bm25_dict else 1.0
    max_embedding = max(embedding_dict.values()) if embedding_dict else 1.0

    # Combine scores
    combined_scores = {}
    for doc in documents:
        bm25_score = bm25_dict.get(doc, 0) / max_bm25 if max_bm25 > 0 else 0
        embedding_score = (
            embedding_dict.get(doc, 0) / max_embedding if max_embedding > 0 else 0
        )
        combined_scores[doc] = alpha * bm25_score + (1 - alpha) * embedding_score

    # Sort by combined score
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    # Return top-k results
    return sorted_results[:top_k]


# Function to evaluate retrieval relevance
def evaluate_relevance(query: str, retrieved_docs: list[tuple[str, float]]) -> str:
    """
    Evaluate the relevance of retrieved documents to the query.

    Args:
        query: Query text.
        retrieved_docs: List of retrieved documents with scores as (document, score) tuples.

    Returns:
        String containing the evaluation results formatted as markdown.
    """
    # Create a prompt for Gemini to evaluate relevance
    docs_text = "\n\n".join(
        [f"Document {i + 1}: {doc}" for i, (doc, _) in enumerate(retrieved_docs)]
    )

    prompt = f"""
    Evaluate the relevance of the following documents to the query.

    Query: {query}

    {docs_text}

    For each document, provide a relevance score from 0 to 10, where:
    - 0-3: Not relevant
    - 4-6: Somewhat relevant
    - 7-10: Highly relevant

    Also provide a brief explanation for each score.

    Format your response as a table with columns: Document Number, Relevance Score, Explanation
    """

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)

        if response:
            # In a real application, you would parse the response into a structured format
            return response
        else:
            return "Failed to evaluate relevance."
    except Exception as e:
        return f"Error evaluating relevance: {e}"


# Streamlit app
st.title("🔍 ContextQuest: Hybrid Retrieval")
st.markdown("""
This application demonstrates hybrid retrieval-augmented generation using a combination of
keyword-based (BM25) and semantic (embedding-based) retrieval methods, powered by Gemini 2.5 Pro.
""")

# Sidebar for API keys and settings
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Google API key:", type="password")

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("API key saved!")
    else:
        st.warning("Please enter your Google API key to proceed.")

    st.markdown("---")
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
        query = st.text_area("Enter your query:", height=100)

        if st.button("Search"):
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

                    for i, (doc, score) in enumerate(retrieved_docs):
                        st.markdown(f"**Document {i + 1}** (Score: {score:.4f})")
                        st.markdown(doc)
                        st.markdown("---")

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
                            evaluation = evaluate_relevance(query, retrieved_docs)
                            st.markdown(evaluation)
            else:
                st.warning("Please enter a query.")
    else:
        st.error("Failed to load context data. Please check the file path.")

# Add footer
st.markdown("---")
st.markdown("ContextQuest: Hybrid Retrieval | Powered by Gemini 2.5 Pro")
