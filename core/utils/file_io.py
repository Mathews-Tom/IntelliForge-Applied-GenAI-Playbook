"""
File I/O helper functions for IntelliForge applications.

This module contains functions for handling file operations
that are used across multiple applications.
"""

import os
import tempfile
import pandas as pd
import csv
import json
from typing import List, Dict, Any, Tuple, Optional, Union, BinaryIO
from pathlib import Path
import io

def save_uploaded_file(uploaded_file: BinaryIO, 
                      directory: Optional[Union[str, Path]] = None) -> str:
    """
    Save an uploaded file to disk.
    
    Args:
        uploaded_file: File object from st.file_uploader
        directory: Directory to save the file (if None, uses a temp directory)
        
    Returns:
        Path to the saved file
    """
    if directory is None:
        directory = tempfile.mkdtemp()
    else:
        directory = Path(directory)
        os.makedirs(directory, exist_ok=True)
        
    file_path = os.path.join(directory, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return file_path

def read_csv_file(file_path: Union[str, Path, BinaryIO]) -> pd.DataFrame:
    """
    Read and parse a CSV file.
    
    Args:
        file_path: Path to the CSV file or file-like object
        
    Returns:
        DataFrame containing the CSV data
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()

def read_text_file(file_path: Union[str, Path, BinaryIO]) -> str:
    """
    Read a text file.
    
    Args:
        file_path: Path to the text file or file-like object
        
    Returns:
        String containing the file contents
    """
    try:
        if isinstance(file_path, (str, Path)):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Assume file-like object
            return file_path.read().decode('utf-8')
    except Exception as e:
        print(f"Error reading text file: {e}")
        return ""

def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get the extension of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (lowercase, without the dot)
    """
    return os.path.splitext(str(file_path))[1].lower().lstrip('.')

def detect_file_type(file_path: Union[str, Path]) -> str:
    """
    Detect the type of a file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File type ('csv', 'txt', 'pdf', etc.)
    """
    return get_file_extension(file_path)

def save_dataframe_to_csv(df: pd.DataFrame, 
                         file_path: Union[str, Path]) -> None:
    """
    Save a DataFrame to a CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the CSV file
    """
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")

def save_json(data: Any, file_path: Union[str, Path]) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving JSON file: {e}")

def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)

def ensure_directory(directory: Union[str, Path]) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)
