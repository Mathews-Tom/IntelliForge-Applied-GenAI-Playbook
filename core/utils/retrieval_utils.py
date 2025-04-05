"""
Retrieval utility functions for IntelliForge applications.

This module contains functions for document retrieval, embedding generation,
and ranking that are used across multiple applications.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
from pathlib import Path

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from core.llm.gemini_utils import GeminiModelType, generate_content

# Embedding functions
def get_embeddings(texts: List[str], 
                  cache: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for a list of texts using Google's text-embedding model.
    
    Args:
        texts: List of texts to embed
        cache: Optional cache dictionary to store embeddings
        
    Returns:
        Dictionary mapping text to embedding vector
    """
    # This is a placeholder. In a real implementation, you would use
    # Google's text-embedding model or another embedding model.
    # For now, we'll use a simple mock implementation.
    
    if cache is None:
        cache = {}
        
    result = {}
    for text in texts:
        if text in cache:
            result[text] = cache[text]
        else:
            # Mock embedding - in a real implementation, call the embedding API
            # This creates a random vector of length 768 (common embedding size)
            embedding = np.random.randn(768)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
            result[text] = embedding
            if cache is not None:
                cache[text] = embedding
                
    return result

# Retrieval functions
def bm25_retrieval(query: str, 
                  documents: List[Union[str, Dict[str, Any]]], 
                  top_k: int = 5) -> List[Tuple[Union[str, Dict[str, Any]], float]]:
    """
    Retrieve documents using BM25 keyword-based retrieval.
    
    Args:
        query: Query string
        documents: List of documents (strings or dictionaries with 'text' key)
        top_k: Number of documents to retrieve
        
    Returns:
        List of (document, score) tuples
    """
    # Extract text from documents if they are dictionaries
    doc_texts = []
    for doc in documents:
        if isinstance(doc, dict):
            doc_texts.append(doc.get('text', ''))
        else:
            doc_texts.append(doc)
    
    # Tokenize documents
    tokenized_docs = [doc.lower().split() for doc in doc_texts]
    
    # Create BM25 model
    bm25 = BM25Okapi(tokenized_docs)
    
    # Tokenize query
    tokenized_query = query.lower().split()
    
    # Get scores
    scores = bm25.get_scores(tokenized_query)
    
    # Get top-k documents
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # Return documents and scores
    results = []
    for i in top_indices:
        results.append((documents[i], scores[i]))
        
    return results

def embedding_retrieval(query: str, 
                       documents: List[Union[str, Dict[str, Any]]], 
                       embeddings: Optional[Dict[str, np.ndarray]] = None,
                       top_k: int = 5) -> List[Tuple[Union[str, Dict[str, Any]], float]]:
    """
    Retrieve documents using embedding-based retrieval.
    
    Args:
        query: Query string
        documents: List of documents (strings or dictionaries with 'text' key)
        embeddings: Optional pre-computed embeddings
        top_k: Number of documents to retrieve
        
    Returns:
        List of (document, score) tuples
    """
    # Extract text from documents if they are dictionaries
    doc_texts = []
    for doc in documents:
        if isinstance(doc, dict):
            doc_texts.append(doc.get('text', ''))
        else:
            doc_texts.append(doc)
    
    # Generate embeddings if not provided
    if embeddings is None:
        embeddings = get_embeddings(doc_texts + [query])
    elif query not in embeddings:
        query_embedding = get_embeddings([query])
        embeddings.update(query_embedding)
    
    # Get query embedding
    query_embedding = embeddings[query].reshape(1, -1)
    
    # Get document embeddings
    doc_embeddings = []
    for doc_text in doc_texts:
        if doc_text in embeddings:
            doc_embeddings.append(embeddings[doc_text])
        else:
            # If embedding not found, use a zero vector
            doc_embeddings.append(np.zeros_like(query_embedding[0]))
    
    # Calculate similarity scores
    doc_embeddings = np.array(doc_embeddings)
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Get top-k documents
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # Return documents and scores
    results = []
    for i in top_indices:
        results.append((documents[i], scores[i]))
        
    return results

def hybrid_retrieval(query: str, 
                    documents: List[Union[str, Dict[str, Any]]], 
                    embeddings: Optional[Dict[str, np.ndarray]] = None,
                    top_k: int = 5, 
                    alpha: float = 0.5) -> List[Tuple[Union[str, Dict[str, Any]], float]]:
    """
    Retrieve documents using a hybrid of BM25 and embedding-based retrieval.
    
    Args:
        query: Query string
        documents: List of documents (strings or dictionaries with 'text' key)
        embeddings: Optional pre-computed embeddings
        top_k: Number of documents to retrieve
        alpha: Weight for BM25 scores (1-alpha for embedding scores)
        
    Returns:
        List of (document, score) tuples
    """
    # Get BM25 results
    bm25_results = bm25_retrieval(query, documents, len(documents))
    
    # Get embedding results
    embedding_results = embedding_retrieval(query, documents, embeddings, len(documents))
    
    # Normalize scores
    bm25_scores = np.array([score for _, score in bm25_results])
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()
    
    embedding_scores = np.array([score for _, score in embedding_results])
    if embedding_scores.max() > 0:
        embedding_scores = embedding_scores / embedding_scores.max()
    
    # Combine scores
    combined_scores = alpha * bm25_scores + (1 - alpha) * embedding_scores
    
    # Get top-k documents
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    # Return documents and scores
    results = []
    for i in top_indices:
        results.append((documents[i], combined_scores[i]))
        
    return results

def rerank_documents(query: str, 
                    documents: List[Union[str, Dict[str, Any]]],
                    reranker: Optional[Callable] = None) -> List[Tuple[Union[str, Dict[str, Any]], float]]:
    """
    Rerank documents using a reranker function.
    
    Args:
        query: Query string
        documents: List of documents (strings or dictionaries with 'text' key)
        reranker: Function that takes a query and documents and returns scores
        
    Returns:
        List of (document, score) tuples
    """
    if reranker is None:
        # Use Gemini to rerank documents
        return gemini_rerank(query, documents)
    
    # Use the provided reranker
    return reranker(query, documents)

def gemini_rerank(query: str, 
                 documents: List[Union[str, Dict[str, Any]]],
                 top_k: int = 5) -> List[Tuple[Union[str, Dict[str, Any]], float]]:
    """
    Rerank documents using Gemini to evaluate relevance.
    
    Args:
        query: Query string
        documents: List of documents (strings or dictionaries with 'text' key)
        top_k: Number of documents to return
        
    Returns:
        List of (document, score) tuples
    """
    # Extract text from documents if they are dictionaries
    doc_texts = []
    for doc in documents:
        if isinstance(doc, dict):
            doc_texts.append(doc.get('text', ''))
        else:
            doc_texts.append(doc)
    
    # Create prompt for Gemini
    docs_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(doc_texts)])
    
    prompt = f"""
    Evaluate the relevance of the following documents to the query.

    Query: {query}

    {docs_text}

    For each document, provide a relevance score from 0 to 10, where:
    - 0-3: Not relevant
    - 4-6: Somewhat relevant
    - 7-10: Highly relevant

    Return ONLY a JSON array of scores, one for each document, like this:
    [7, 3, 9, 5, 2]
    """

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)
        
        if response:
            # Try to parse the response as a JSON array
            try:
                import json
                scores = json.loads(response.strip())
                
                if isinstance(scores, list) and len(scores) == len(documents):
                    # Normalize scores to 0-1 range
                    scores = np.array(scores) / 10.0
                    
                    # Get top-k documents
                    top_indices = np.argsort(scores)[::-1][:top_k]
                    
                    # Return documents and scores
                    results = []
                    for i in top_indices:
                        results.append((documents[i], scores[i]))
                        
                    return results
            except:
                # If parsing fails, fall back to a simple approach
                pass
        
        # Fall back to returning the original documents with equal scores
        return [(doc, 1.0) for doc in documents[:top_k]]
    except Exception as e:
        print(f"Error in Gemini reranking: {e}")
        return [(doc, 1.0) for doc in documents[:top_k]]
