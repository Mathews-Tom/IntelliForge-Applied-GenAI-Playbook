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


# Document loading functions
def load_documents_from_csv(
    file_path: Union[str, Path],
    text_column: str = "text",
    id_column: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load documents from a CSV file.

    Args:
        file_path: Path to the CSV file
        text_column: Name of the column containing the document text
        id_column: Name of the column containing document IDs (optional)

    Returns:
        List of document dictionaries with 'text' and optional 'id' keys
    """
    try:
        df = pd.read_csv(file_path)
        documents = []

        for _, row in df.iterrows():
            doc = {"text": row[text_column]}
            if id_column and id_column in df.columns:
                doc["id"] = row[id_column]
            else:
                doc["id"] = str(len(documents))

            # Add all other columns as metadata
            for col in df.columns:
                if col != text_column and col != id_column:
                    doc[col] = row[col]

            documents.append(doc)

        return documents
    except Exception as e:
        print(f"Error loading documents from CSV: {e}")
        return []


def load_documents_from_text_file(
    file_path: Union[str, Path], chunk_size: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load documents from a text file, optionally chunking the content.

    Args:
        file_path: Path to the text file
        chunk_size: Number of characters per chunk (if None, the entire file is one document)

    Returns:
        List of document dictionaries with 'text' and 'id' keys
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if chunk_size is None:
            return [{"id": "0", "text": text, "source": str(file_path)}]

        # Chunk the text
        chunks = chunk_text(text, chunk_size)
        return [
            {"id": str(i), "text": chunk, "source": str(file_path)}
            for i, chunk in enumerate(chunks)
        ]
    except Exception as e:
        print(f"Error loading documents from text file: {e}")
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
