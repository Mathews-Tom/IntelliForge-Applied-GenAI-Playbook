"""
Retrieval utility functions for IntelliForge applications.

This module contains functions for document retrieval, embedding generation,
and ranking that are used across multiple applications.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

import google.generativeai as genai  # Added for embeddings
import numpy as np
import streamlit as st  # Added for caching
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry  # Added for retries
from tenacity import RetryError, stop_after_attempt, wait_exponential

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Assuming generate_content might still be needed for reranking
from core.llm.gemini_utils import GeminiModelType, generate_content

# Constants
EMBEDDING_MODEL = "models/text-embedding-004"  # Or the specific model being used
EMBEDDING_BATCH_SIZE = 100  # Check API limits for the chosen model

# --- Embedding Function (Async with Batching & Retries) ---


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _embed_batch(batch: List[str]) -> List[np.ndarray]:
    """Helper function to embed a single batch with retries."""
    try:
        # Use embed_content for batch embedding
        # Task type can be adjusted based on use case (e.g., RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY)
        result = await genai.embed_content_async(
            model=EMBEDDING_MODEL,
            content=batch,
            task_type="RETRIEVAL_DOCUMENT",  # Adjust if needed
        )
        # Ensure the result structure matches expectations
        if "embedding" in result and isinstance(result["embedding"], list):
            # Convert list of lists to list of numpy arrays
            return [np.array(emb) for emb in result["embedding"]]
        else:
            # Handle unexpected response format
            print(f"Warning: Unexpected embedding response format: {result}")
            # Return list of zero vectors matching batch size and expected dimension (e.g., 768)
            # TODO: Determine actual embedding dimension dynamically or from config
            return [np.zeros(768) for _ in batch]
    except Exception as e:
        print(f"Error embedding batch: {e}")
        raise  # Reraise to trigger tenacity retry


async def get_embeddings(
    texts: List[str], cache: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, np.ndarray]:
    """
    Generate embeddings asynchronously for a list of texts using Google's
    text-embedding model, with batching, caching, and retries.

    Args:
        texts: List of unique texts to embed.
        cache: Optional cache dictionary to store/retrieve embeddings.

    Returns:
        Dictionary mapping text to its embedding vector (np.ndarray).
        Returns empty dict if input texts is empty.
    """
    if not texts:
        return {}

    if cache is None:
        cache = {}

    results = {}
    texts_to_embed = []

    # 1. Check cache
    for text in texts:
        if text in cache:
            results[text] = cache[text]
        else:
            # Avoid adding duplicates to the embedding list if input `texts` had duplicates
            if text not in texts_to_embed:
                texts_to_embed.append(text)

    # 2. Process texts needing embedding in batches
    if texts_to_embed:
        print(f"Found {len(texts_to_embed)} unique texts requiring embedding.")
        all_embeddings = []
        for i in range(0, len(texts_to_embed), EMBEDDING_BATCH_SIZE):
            batch = texts_to_embed[i : i + EMBEDDING_BATCH_SIZE]
            print(
                f"Embedding batch {i // EMBEDDING_BATCH_SIZE + 1} ({len(batch)} texts)..."
            )
            try:
                batch_embeddings = await _embed_batch(batch)
                if len(batch_embeddings) == len(batch):
                    all_embeddings.extend(batch_embeddings)
                else:
                    print(
                        f"Warning: Mismatch in batch embedding results count for batch starting at index {i}."
                    )
                    # Handle mismatch - e.g., add zero vectors for missing ones
                    # TODO: Determine actual embedding dimension
                    all_embeddings.extend([np.zeros(768)] * len(batch))

            except RetryError as e:
                print(f"Embedding batch failed after retries: {e}. Skipping batch.")
                # Handle failed batch - e.g., add zero vectors
                # TODO: Determine actual embedding dimension
                all_embeddings.extend([np.zeros(768)] * len(batch))
            except Exception as e:
                print(f"Unexpected error during batch embedding: {e}. Skipping batch.")
                # TODO: Determine actual embedding dimension
                all_embeddings.extend([np.zeros(768)] * len(batch))

        # 3. Store results and update cache
        if len(all_embeddings) == len(texts_to_embed):
            for text, embedding in zip(texts_to_embed, all_embeddings):
                results[text] = embedding
                if cache is not None:
                    cache[text] = embedding
        else:
            print(
                f"Error: Final embedding count ({len(all_embeddings)}) does not match texts to embed ({len(texts_to_embed)})."
            )

    return results


# Retrieval functions
@st.cache_data(show_spinner="Calculating BM25 scores...")
def bm25_retrieval(
    query: str, documents: List[Union[str, Dict[str, Any]]], top_k: int = 5
) -> List[Tuple[Union[str, Dict[str, Any]], float]]:
    """
    Retrieve documents using BM25 keyword-based retrieval (cached).

    Args:
        query: Query string.
        documents: List of documents (strings or dictionaries with 'text' key).
                   Note: For caching to work effectively, this list should be
                   treated as immutable within a Streamlit session run.
        top_k: Number of documents to retrieve.

    Returns:
        List of (document, score) tuples.
    """
    print(
        f"Cache miss: Calculating BM25 for query '{query[:50]}...'"
    )  # Log cache misses

    if not documents:
        return []

    # Extract text from documents if they are dictionaries
    doc_texts = []
    original_docs_map = {}  # Keep track of original doc structure
    for i, doc in enumerate(documents):
        if isinstance(doc, dict):
            text = doc.get("text", "")
            doc_texts.append(text)
            original_docs_map[i] = doc
        else:
            # Ensure doc is a string before lowercasing/splitting
            text = str(doc) if doc is not None else ""
            doc_texts.append(text)
            original_docs_map[i] = doc

    # Basic tokenization (consider more advanced tokenizers if needed)
    try:
        tokenized_docs = [doc.lower().split() for doc in doc_texts]
        tokenized_query = query.lower().split()
    except AttributeError as e:
        st.error(
            f"Error tokenizing documents/query for BM25: {e}. Ensure all documents are text."
        )
        print(f"BM25 Tokenization Error: {e}")
        return []

    # Create BM25 model
    # Handle potential errors during BM25 initialization (e.g., empty docs)
    try:
        if not any(tokenized_docs):  # Check if all docs are empty after tokenization
            print("Warning: All documents are empty after tokenization for BM25.")
            return []
        bm25 = BM25Okapi(tokenized_docs)
    except ValueError as e:
        # rank_bm25 might raise ValueError for certain inputs (e.g., zero length docs list)
        st.error(f"Error initializing BM25 model: {e}")
        print(f"BM25 Initialization Error: {e}")
        return []

    # Get scores
    try:
        scores = bm25.get_scores(tokenized_query)
    except Exception as e:
        # Catch potential errors during scoring
        st.error(f"Error getting BM25 scores: {e}")
        print(f"BM25 Scoring Error: {e}")
        return []

    # Get top-k documents
    # Ensure scores array is not empty before argsort
    if len(scores) == 0:
        return []
    # Ensure top_k is not greater than the number of documents
    actual_top_k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:actual_top_k]  # Use actual_top_k

    # Return documents and scores using the original map
    results = []
    for i in top_indices:
        # Ensure index i is valid for original_docs_map
        if i in original_docs_map:
            results.append((original_docs_map[i], scores[i]))
        else:
            print(f"Warning: BM25 index {i} out of bounds for original documents map.")

    return results


async def embedding_retrieval(
    query: str,
    documents: List[Union[str, Dict[str, Any]]],
    embeddings_cache: Optional[Dict[str, np.ndarray]] = None,  # Renamed for clarity
    top_k: int = 5,
) -> List[Tuple[Union[str, Dict[str, Any]], float]]:
    """
    Retrieve documents asynchronously using embedding-based retrieval.

    Args:
        query: Query string
        documents: List of documents (strings or dictionaries with 'text' key)
        embeddings_cache: Optional pre-computed embeddings cache. This function will
                          fetch missing embeddings and update the cache if provided.
        top_k: Number of documents to retrieve

    Returns:
        List of (document, score) tuples
    """
    if not documents:
        return []

    # Ensure cache exists if provided, otherwise create temporary one
    local_cache = embeddings_cache if embeddings_cache is not None else {}

    # Extract text from documents if they are dictionaries
    doc_texts = []
    original_docs_map = {}  # Keep track of original doc structure
    for i, doc in enumerate(documents):
        if isinstance(doc, dict):
            text = doc.get("text", "")
            doc_texts.append(text)
            original_docs_map[i] = doc
        else:
            doc_texts.append(doc)
            original_docs_map[i] = doc

    # Identify texts needing embeddings (query + docs not in cache)
    texts_to_check = [query] + doc_texts
    unique_texts_to_embed = list(set(t for t in texts_to_check if t not in local_cache))

    # Generate embeddings asynchronously if needed
    if unique_texts_to_embed:
        print(f"Fetching embeddings for {len(unique_texts_to_embed)} texts...")
        fetched_embeddings = await get_embeddings(unique_texts_to_embed, local_cache)
        # Update the main cache if it was provided
        if embeddings_cache is not None:
            embeddings_cache.update(fetched_embeddings)
        # Update the local cache regardless
        local_cache.update(fetched_embeddings)

    # Get query embedding (should be in local_cache now)
    if query not in local_cache:
        print(f"Warning: Query embedding for '{query}' not found after fetch attempt.")
        return []  # Cannot proceed without query embedding
    query_embedding = local_cache[query].reshape(1, -1)

    # Get document embeddings
    doc_embeddings_list = []
    valid_indices = []
    for i, doc_text in enumerate(doc_texts):
        if doc_text in local_cache:
            doc_embeddings_list.append(local_cache[doc_text])
            valid_indices.append(i)  # Keep track of docs with embeddings
        else:
            # This shouldn't happen if get_embeddings worked, but handle defensively
            print(
                f"Warning: Document embedding for text '{doc_text[:50]}...' not found."
            )
            # Optionally: append zero vector and keep index, or skip the document
            # Skipping for now:
            # doc_embeddings_list.append(np.zeros_like(query_embedding[0]))
            # valid_indices.append(i)
            pass

    if not doc_embeddings_list:
        print("Warning: No valid document embeddings found.")
        return []

    # Calculate similarity scores only for documents with embeddings
    doc_embeddings_array = np.array(doc_embeddings_list)
    scores = cosine_similarity(query_embedding, doc_embeddings_array)[0]

    # Map scores back to original document indices and combine
    doc_score_pairs = []
    for score_idx, original_doc_idx in enumerate(valid_indices):
        doc_score_pairs.append((original_docs_map[original_doc_idx], scores[score_idx]))

    # Sort by score descending
    doc_score_pairs.sort(key=lambda item: item[1], reverse=True)

    # Return top-k documents
    return doc_score_pairs[:top_k]


async def hybrid_retrieval(
    query: str,
    documents: List[Union[str, Dict[str, Any]]],
    embeddings_cache: Optional[Dict[str, np.ndarray]] = None,  # Pass cache along
    top_k: int = 5,
    alpha: float = 0.5,  # Weight for BM25
) -> List[Tuple[Union[str, Dict[str, Any]], float]]:
    """
    Retrieve documents asynchronously using a hybrid of BM25 and embedding-based retrieval.
    Uses Reciprocal Rank Fusion (RRF) for combining results.

    Args:
        query: Query string
        documents: List of documents (strings or dictionaries with 'text' key)
        embeddings_cache: Optional pre-computed embeddings cache for embedding retrieval.
        top_k: Number of documents to retrieve.
        alpha: Weight for BM25 (Not used with RRF, kept for potential future use/compatibility).

    Returns:
        List of (document, score) tuples, sorted by RRF score.
    """
    if not documents:
        return []

    # --- Run Retrievers ---
    # BM25 (synchronous) - Retrieve more results initially for better fusion
    k_for_bm25 = max(top_k * 2, 10)  # Retrieve more for BM25
    bm25_results = bm25_retrieval(query, documents, k_for_bm25)

    # Embedding (asynchronous) - Retrieve more results initially
    k_for_embedding = max(top_k * 2, 10)  # Retrieve more for embedding
    embedding_results = await embedding_retrieval(
        query, documents, embeddings_cache, k_for_embedding
    )

    # --- Reciprocal Rank Fusion (RRF) ---
    # Create maps for easy lookup: doc -> score
    # Using document text as key assumes unique text content for simplicity.
    # A more robust approach might use document IDs if available.
    def get_doc_key(doc):
        return doc.get("text", doc) if isinstance(doc, dict) else doc

    bm25_rank_map = {
        get_doc_key(doc): rank + 1 for rank, (doc, _) in enumerate(bm25_results)
    }
    embedding_rank_map = {
        get_doc_key(doc): rank + 1 for rank, (doc, _) in enumerate(embedding_results)
    }

    # Calculate RRF scores (k=60 is a common default for RRF)
    rrf_k = 60
    combined_scores = {}
    all_docs = {get_doc_key(doc): doc for doc, _ in bm25_results + embedding_results}

    for doc_key, doc_obj in all_docs.items():
        score = 0.0
        if doc_key in bm25_rank_map:
            score += 1.0 / (rrf_k + bm25_rank_map[doc_key])
        if doc_key in embedding_rank_map:
            score += 1.0 / (rrf_k + embedding_rank_map[doc_key])
        if score > 0:
            combined_scores[doc_key] = score

    # Sort documents by combined RRF score
    sorted_doc_keys = sorted(combined_scores, key=combined_scores.get, reverse=True)

    # Return top-k documents with their RRF scores
    results = []
    for doc_key in sorted_doc_keys[:top_k]:
        results.append((all_docs[doc_key], combined_scores[doc_key]))

    return results


async def rerank_documents(
    query: str,
    documents: List[Union[str, Dict[str, Any]]],
    # Assuming any provided reranker will also be async
    reranker: Optional[Callable] = None,
) -> List[Tuple[Union[str, Dict[str, Any]], float]]:
    """
    Rerank documents asynchronously using a reranker function.

    Args:
        query: Query string
        documents: List of documents (strings or dictionaries with 'text' key)
        reranker: Async function that takes a query and documents and returns scores

    Returns:
        List of (document, score) tuples
    """
    if reranker is None:
        # Use Gemini to rerank documents (now async)
        return await gemini_rerank(query, documents)

    # Use the provided async reranker
    # If a synchronous reranker could be passed, error handling/wrapping would be needed
    return await reranker(query, documents)


async def gemini_rerank(
    query: str, documents: List[Union[str, Dict[str, Any]]], top_k: int = 5
) -> List[Tuple[Union[str, Dict[str, Any]], float]]:
    """
    Rerank documents asynchronously using Gemini to evaluate relevance.

    Args:
        query: Query string
        documents: List of documents (strings or dictionaries with 'text' key)
        top_k: Number of documents to return

    Returns:
        List of (document, score) tuples
    """
    if not documents:
        return []

    # Extract text from documents if they are dictionaries
    doc_texts = []
    original_docs_map = {}  # Keep track of original doc structure
    for i, doc in enumerate(documents):
        if isinstance(doc, dict):
            text = doc.get("text", "")
            doc_texts.append(text)
            original_docs_map[i] = doc
        else:
            doc_texts.append(doc)
            original_docs_map[i] = doc

    # Create prompt for Gemini
    docs_prompt_part = "\n\n".join(
        [f"Document {i + 1}:\n{doc_text}" for i, doc_text in enumerate(doc_texts)]
    )

    prompt = f"""Evaluate the relevance of the following documents to the query.

Query: {query}

{docs_prompt_part}

For each document, provide a relevance score from 0 to 10, where 0 is not relevant and 10 is highly relevant.
Return ONLY a JSON array of scores, one score per document in the original order, like this: [7, 3, 9, 5, 2]"""

    try:
        # generate_content is async, needs await
        response = await generate_content(
            GeminiModelType.GEMINI_2_5_PRO, prompt
        )  # Consider a faster/cheaper model?

        if response:
            # Try to parse the response as a JSON array
            try:
                import json

                # Clean potential markdown code fences
                cleaned_response = (
                    response.strip().strip("```json").strip("```").strip()
                )
                scores = json.loads(cleaned_response)

                if isinstance(scores, list) and len(scores) == len(documents):
                    # Normalize scores to 0-1 range (optional, but good practice)
                    scores_array = np.array(scores, dtype=float) / 10.0

                    # Combine original documents with scores
                    doc_score_pairs = []
                    for i, score in enumerate(scores_array):
                        doc_score_pairs.append((original_docs_map[i], score))

                    # Sort by score descending
                    doc_score_pairs.sort(key=lambda item: item[1], reverse=True)

                    # Return top-k
                    return doc_score_pairs[:top_k]
                else:
                    print(
                        f"Warning: Gemini rerank response parsing error - score count mismatch or not a list. Response: {response}"
                    )

            except json.JSONDecodeError as json_err:
                print(
                    f"Warning: Gemini rerank response JSON parsing failed: {json_err}. Response: {response}"
                )
            except Exception as parse_err:
                print(
                    f"Warning: Gemini rerank response parsing failed unexpectedly: {parse_err}. Response: {response}"
                )

        # Fallback: Return original documents (up to top_k) with default score if reranking fails
        print(
            "Warning: Falling back to original document order due to reranking failure."
        )
        return [
            (original_docs_map[i], 0.5) for i in range(min(top_k, len(documents)))
        ]  # Use 0.5 to indicate fallback?

    except Exception as e:
        print(f"Error during Gemini reranking API call: {e}")
        # Fallback on API error
        return [(original_docs_map[i], 0.5) for i in range(min(top_k, len(documents)))]
        # Duplicate code block removed
        print(f"Error during Gemini reranking API call: {e}")
        # Fallback on API error
        return [(original_docs_map[i], 0.5) for i in range(min(top_k, len(documents)))]
        # Duplicate code block removed
