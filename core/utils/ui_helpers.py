"""
UI helper functions for IntelliForge applications.

This module contains functions for creating consistent UI components
across multiple Streamlit applications.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st


def create_sidebar(title: str, description: str, api_key_field: bool = True) -> None:
    """
    Create a standard sidebar with configuration options.

    Args:
        title: Title of the sidebar
        description: Description text
        api_key_field: Whether to include an API key input field
    """
    with st.sidebar:
        st.header(title)
        st.markdown(description)

        if api_key_field:
            st.header("Configuration")
            api_key = st.text_input("Enter your Google API key:", type="password")

            if api_key:
                st.session_state.api_key = api_key
                st.success("API key saved!")
            else:
                st.warning("Please enter your Google API key to proceed.")

        st.markdown("---")


def create_header(title: str, description: str, icon: Optional[str] = None) -> None:
    """
    Create a standard header for the application.

    Args:
        title: Title of the application
        description: Description text
        icon: Optional emoji icon to display before the title
    """
    if icon:
        st.title(f"{icon} {title}")
    else:
        st.title(title)

    st.markdown(description)
    st.markdown("---")


def create_footer() -> None:
    """Create a standard footer for the application."""
    st.markdown("---")
    st.markdown("IntelliForge: Applied GenAI Playbook | Powered by Gemini 2.5 Pro")


def display_documents(
    documents: List[Dict[str, Any]],
    scores: Optional[List[float]] = None,
    max_docs: int = 10,
    expand: bool = True,
) -> None:
    """
    Display retrieved documents in a standardized format.

    Args:
        documents: List of document dictionaries
        scores: Optional list of relevance scores
        max_docs: Maximum number of documents to display
        expand: Whether to put documents in an expander
    """
    if not documents:
        st.info("No documents to display.")
        return

    display_content = (
        st.expander("Retrieved Documents", expanded=expand) if expand else st
    )

    with display_content:
        st.info(
            f"Displaying {min(len(documents), max_docs)} of {len(documents)} documents."
        )

        for i, doc in enumerate(documents[:max_docs]):
            if scores and i < len(scores):
                st.markdown(f"**Document {i + 1}** (Score: {scores[i]:.4f})")
            else:
                st.markdown(f"**Document {i + 1}**")

            # Display text
            st.markdown(doc.get("text", "No text available"))

            # Display metadata if available
            if "source" in doc:
                st.caption(f"Source: {doc['source']}")

            st.markdown("---")


def file_uploader(
    label: str = "Upload your documents",
    types: List[str] = ["csv", "txt", "pdf"],
    multiple: bool = False,
    key: Optional[str] = None,
) -> Any:
    """
    Create a standardized file uploader.

    Args:
        label: Label for the file uploader
        types: List of allowed file types
        multiple: Whether to allow multiple files
        key: Optional key for the Streamlit component

    Returns:
        Uploaded file(s)
    """
    return st.file_uploader(label, types, multiple=multiple, key=key)


def create_query_input(
    label: str = "Enter your query:",
    height: int = 100,
    button_text: str = "Search",
    key: Optional[str] = None,
) -> Tuple[str, bool]:
    """
    Create a standardized query input with a button.

    Args:
        label: Label for the text input
        height: Height of the text area
        button_text: Text for the button
        key: Optional key for the Streamlit component

    Returns:
        Tuple of (query_text, button_clicked)
    """
    query = st.text_area(label, height=height, key=f"query_{key}" if key else None)
    button_clicked = st.button(button_text, key=f"button_{key}" if key else None)

    return query, button_clicked


def display_processing_steps(steps: List[Dict[str, Any]], expand: bool = True) -> None:
    """
    Display processing steps in a standardized format.

    Args:
        steps: List of step dictionaries with 'title', 'content', and optional 'time' keys
        expand: Whether to put steps in an expander
    """
    display_content = st.expander("Processing Steps", expanded=expand) if expand else st

    with display_content:
        for i, step in enumerate(steps):
            with st.container():
                header = step.get("title", f"Step {i + 1}")
                if "time" in step:
                    header += f" ({step['time']:.2f}s)"

                st.subheader(header)

                if "content" in step:
                    if isinstance(step["content"], str):
                        st.markdown(step["content"])
                    elif isinstance(step["content"], pd.DataFrame):
                        st.dataframe(step["content"])
                    elif isinstance(step["content"], dict):
                        for k, v in step["content"].items():
                            st.markdown(f"**{k}:** {v}")
                    else:
                        st.write(step["content"])

                st.markdown("---")


def timed_operation(operation: Callable, description: str) -> Tuple[Any, float]:
    """
    Execute an operation and time it.

    Args:
        operation: Function to execute
        description: Description of the operation for the spinner

    Returns:
        Tuple of (operation_result, execution_time)
    """
    with st.spinner(description):
        start_time = time.time()
        result = operation()
        execution_time = time.time() - start_time

    return result, execution_time
