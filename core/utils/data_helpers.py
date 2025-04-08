"""
Data helper functions for IntelliForge applications.

This module contains functions for loading, processing, and manipulating data
that are used across multiple applications.
"""

import csv
import io
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st  # Added for caching decorators


# Document loading functions
@st.cache_data(show_spinner="Loading documents from CSV...")
def load_documents_from_csv(
    file_path: Union[str, Path],
    text_column: str = "text",
    id_column: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load documents from a CSV file (cached).

    Args:
        file_path: Path to the CSV file
        text_column: Name of the column containing the document text
        id_column: Name of the column containing document IDs (optional)

    Returns:
        List of document dictionaries with 'text' and optional 'id' keys
    """
    print(f"Cache miss: Loading CSV from {file_path}")  # Log cache misses
    try:
        # Ensure file_path is usable by pandas (e.g., handle Path objects)
        df = pd.read_csv(str(file_path))
        documents = []

        required_cols = [text_column]
        if id_column:
            required_cols.append(id_column)
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns in CSV: {missing}")

        for _, row in df.iterrows():
            # Handle potential missing text data gracefully
            text_content = row.get(text_column, "")
            doc = {"text": str(text_content) if pd.notna(text_content) else ""}

            if id_column and id_column in df.columns:
                doc["id"] = str(row[id_column])  # Ensure ID is string
            else:
                doc["id"] = str(len(documents))  # Use index as fallback ID

            # Add all other columns as metadata, handling potential NaN values
            for col in df.columns:
                if col != text_column and col != id_column:
                    value = row[col]
                    doc[col] = (
                        str(value) if pd.notna(value) else None
                    )  # Convert non-null to string

            documents.append(doc)

        return documents
    except FileNotFoundError:
        st.error(f"Error: CSV file not found at {file_path}")
        return []
    except Exception as e:
        st.error(f"Error loading documents from CSV '{file_path}': {e}")
        print(f"Error loading documents from CSV: {e}")  # Also print to console log
        return []


@st.cache_data(show_spinner="Loading documents from text file...")
def load_documents_from_text_file(
    file_path: Union[str, Path], chunk_size: Optional[int] = None, overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Load documents from a text file (cached), optionally chunking the content.

    Args:
        file_path: Path to the text file.
        chunk_size: Number of characters per chunk (if None, the entire file is one document).
        overlap: Number of characters to overlap between chunks (used if chunk_size is set).

    Returns:
        List of document dictionaries with 'text', 'id', and 'source' keys.
    """
    print(f"Cache miss: Loading text file from {file_path}")  # Log cache misses
    try:
        file_path_str = str(file_path)  # Ensure string path for open()
        with open(file_path_str, "r", encoding="utf-8") as f:
            text = f.read()

        if not text:
            st.warning(f"Text file '{file_path_str}' is empty.")
            return []

        if chunk_size is None:
            # Return entire file as one document
            return [{"id": "0", "text": text, "source": file_path_str}]
        else:
            # Chunk the text using the utility function
            chunks = chunk_text(text, chunk_size, overlap)
            if not chunks:
                st.warning(f"Text file '{file_path_str}' resulted in zero chunks.")
                return []
            # Create document dictionaries for each chunk
            return [
                {"id": str(i), "text": chunk, "source": file_path_str, "chunk_index": i}
                for i, chunk in enumerate(chunks)
            ]
    except FileNotFoundError:
        st.error(f"Error: Text file not found at {file_path_str}")
        return []
    except Exception as e:
        st.error(f"Error loading documents from text file '{file_path_str}': {e}")
        print(
            f"Error loading documents from text file: {e}"
        )  # Also print to console log
        return []


# Text processing functions
def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Split text into chunks of specified size with optional overlap.

    Args:
        text: Text to split into chunks
        chunk_size: Maximum number of characters per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end of the text, try to find a good break point
        if end < len(text):
            # Try to break at paragraph, then sentence, then word boundary
            paragraph_break = text.rfind("\n\n", start, end)
            sentence_break = text.rfind(". ", start, end)
            space_break = text.rfind(" ", start, end)

            if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                end = paragraph_break + 2  # Include the paragraph break
            elif sentence_break != -1 and sentence_break > start + chunk_size // 2:
                end = sentence_break + 2  # Include the period and space
            elif space_break != -1:
                end = space_break + 1  # Include the space

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks


def create_document_metadata(
    document: str, source: str, chunk_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create metadata for a document.

    Args:
        document: Document text
        source: Source of the document (e.g., file name)
        chunk_id: ID of the chunk if the document is chunked

    Returns:
        Dictionary containing metadata
    """
    metadata = {
        "source": source,
        "length": len(document),
        "created_at": pd.Timestamp.now().isoformat(),
    }

    if chunk_id is not None:
        metadata["chunk_id"] = chunk_id

    return metadata


# Data transformation functions
def documents_to_dataframe(documents: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of document dictionaries to a pandas DataFrame.

    Args:
        documents: List of document dictionaries

    Returns:
        DataFrame containing the documents
    """
    return pd.DataFrame(documents)


def dataframe_to_documents(
    df: pd.DataFrame, text_column: str = "text", id_column: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convert a pandas DataFrame to a list of document dictionaries.

    Args:
        df: DataFrame containing documents
        text_column: Name of the column containing the document text
        id_column: Name of the column containing document IDs (optional)

    Returns:
        List of document dictionaries
    """
    documents = []

    for i, row in df.iterrows():
        doc = {"text": row[text_column]}
        if id_column and id_column in df.columns:
            doc["id"] = row[id_column]
        else:
            doc["id"] = str(i)

        # Add all other columns as metadata
        for col in df.columns:
            if col != text_column and col != id_column:
                doc[col] = row[col]

        documents.append(doc)

    return documents

    return documents
    return documents
