"""
Web crawling utility functions for IntelliForge applications.

This module contains functions for web crawling and data extraction
using the crawl4ai library, optimized for RAG applications.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import streamlit as st  # Added for caching

try:
    from crawl4ai import Crawler

    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def check_crawl4ai_installed() -> bool:
    """
    Check if crawl4ai is installed.

    Returns:
        Boolean indicating if crawl4ai is available
    """
    return CRAWL4AI_AVAILABLE


@st.cache_data(show_spinner="Crawling URL...")
def crawl_url(
    url: str, max_pages: int = 10, max_depth: int = 2, output_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Crawl a URL and extract content optimized for RAG (cached).

    Args:
        url: The URL to crawl.
        max_pages: Maximum number of pages to crawl.
        max_depth: Maximum depth of crawling.
        output_dir: Directory to save crawled content (optional).

    Returns:
        List of dictionaries containing crawled content.
    """
    print(f"Cache miss: Crawling URL '{url}'")  # Log cache misses
    if not check_crawl4ai_installed():
        st.error("crawl4ai is not installed. Please install it: pip install crawl4ai")
        return []
        # Or raise ImportError if preferred

    try:
        # Initialize crawler
        # Note: Crawler instance itself might not be cacheable with st.cache_data
        # We are caching the *result* of the crawl operation.
        crawler = Crawler(
            urls=[url],
            max_pages=max_pages,
            max_depth=max_depth,
            output_dir=output_dir,
            verbose=True,  # Consider making this False or configurable for less console noise
        )

        # Start crawling
        results = crawler.crawl()  # This is the potentially slow part

        # Process results into a standardized format
        processed_results = []
        if results:  # Check if results is not None or empty
            for result in results:
                # Ensure result is a dict before accessing keys
                if isinstance(result, dict):
                    processed_result = {
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        # Prioritize markdown, fallback to text, ensure string
                        "text": str(result.get("markdown", result.get("text", ""))),
                        "source": url,  # Original starting URL
                        "crawled_at": time.time(),
                    }
                    processed_results.append(processed_result)
                else:
                    print(
                        f"Warning: Unexpected item type in crawl results for {url}: {type(result)}"
                    )

        if not processed_results:
            st.warning(
                f"Crawling URL '{url}' completed but yielded no processable content."
            )

        return processed_results
    except Exception as e:
        st.error(f"Error crawling URL {url}: {e}")
        print(f"Error crawling URL {url}: {e}")  # Also log to console
        return []


@st.cache_data(show_spinner="Crawling multiple URLs...")
def crawl_multiple_urls(
    urls: List[str],
    max_pages_per_url: int = 5,
    max_depth: int = 2,
    output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Crawl multiple URLs and extract content optimized for RAG (cached).

    Args:
        urls: List of URLs to crawl.
        max_pages_per_url: Maximum number of pages to crawl per URL.
        max_depth: Maximum depth of crawling.
        output_dir: Directory to save crawled content (optional).

    Returns:
        List of dictionaries containing crawled content from all URLs.
    """
    # Convert list to tuple for hashing by st.cache_data
    urls_tuple = tuple(sorted(urls))  # Sort to ensure cache hits regardless of order
    print(f"Cache miss: Crawling multiple URLs: {urls_tuple}")  # Log cache misses

    if not check_crawl4ai_installed():
        st.error("crawl4ai is not installed. Please install it: pip install crawl4ai")
        return []

    all_results = []
    # Note: This loop calls the cached crawl_url function.
    # Caching here provides an additional layer if the exact same *list* of URLs
    # is requested again, avoiding the loop entirely.
    for url in urls_tuple:  # Iterate over the sorted tuple used for caching
        try:
            # Call the potentially cached crawl_url function
            results = crawl_url(
                url=url,
                max_pages=max_pages_per_url,
                max_depth=max_depth,
                output_dir=output_dir,  # Pass output_dir if needed by individual crawls
            )
            all_results.extend(results)
        except Exception as e:
            # Error should ideally be handled within crawl_url, but catch here too
            st.error(f"Unexpected error processing crawl result for {url}: {e}")
            print(f"Error processing crawl result for {url}: {e}")

    if not all_results:
        st.warning(
            f"Crawling multiple URLs {urls_tuple} yielded no processable content."
        )

    return all_results


@st.cache_data(show_spinner="Crawling topic...")
def crawl_topic(
    topic: str,
    num_results: int = 5,
    max_pages_per_result: int = 3,
    max_depth: int = 1,
    output_dir: Optional[str] = None,
    # TODO: Add parameter for actual search engine integration later
) -> List[Dict[str, Any]]:
    """
    Crawl web pages related to a topic using (mock) search results (cached).

    Args:
        topic: The topic to search for.
        num_results: Number of search results to crawl.
        max_pages_per_result: Maximum number of pages to crawl per search result.
        max_depth: Maximum depth of crawling.
        output_dir: Directory to save crawled content (optional).

    Returns:
        List of dictionaries containing crawled content.
    """
    print(f"Cache miss: Crawling topic '{topic}'")  # Log cache misses
    if not check_crawl4ai_installed():
        st.error("crawl4ai is not installed. Please install it: pip install crawl4ai")
        return []

    try:
        # --- Mock Search Results ---
        # Replace this with actual search API call in a real implementation
        print(f"Note: Using MOCK search results for topic '{topic}'")
        base_url = "https://example-search.com"  # Use a more distinct mock domain
        search_results = [
            f"{base_url}/mock/{topic}/result{i + 1}" for i in range(num_results)
        ]
        # --------------------------

        if not search_results:
            st.warning(f"No search results found (or mocked) for topic '{topic}'.")
            return []

        # Limit to requested number of results (already done in mock generation)
        # search_results = search_results[:num_results]

        # Call the potentially cached crawl_multiple_urls function
        return crawl_multiple_urls(
            urls=search_results,  # Pass the generated/mocked URLs
            max_pages_per_url=max_pages_per_result,
            max_depth=max_depth,
            output_dir=output_dir,
        )
    except Exception as e:
        st.error(f"Error crawling topic {topic}: {e}")
        print(f"Error crawling topic {topic}: {e}")  # Also log to console
        return []


def save_crawled_content(content: List[Dict[str, Any]], output_file: str) -> bool:
    """
    Save crawled content to a file.

    Args:
        content: List of dictionaries containing crawled content
        output_file: Path to save the content

    Returns:
        Boolean indicating success
    """
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save content
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"Error saving crawled content: {e}")
        return False


@st.cache_data(show_spinner="Loading crawled content...")
def load_crawled_content(input_file: str) -> List[Dict[str, Any]]:
    """
    Load crawled content from a file (cached).

    Args:
        input_file: Path to the content file.

    Returns:
        List of dictionaries containing crawled content.
    """
    print(
        f"Cache miss: Loading crawled content from '{input_file}'"
    )  # Log cache misses
    try:
        # Ensure file exists before trying to open
        if not os.path.exists(input_file):
            st.error(f"Error: Crawled content file not found at {input_file}")
            return []

        with open(input_file, "r", encoding="utf-8") as f:
            content = json.load(f)

        # Basic validation: Check if it's a list
        if not isinstance(content, list):
            st.error(
                f"Error: Crawled content file '{input_file}' does not contain a valid JSON list."
            )
            return []

        # Optional: Validate structure of items in the list
        # for item in content:
        #     if not isinstance(item, dict) or "text" not in item:
        #         st.warning(f"Warning: Invalid item structure found in {input_file}")
        #         # Decide how to handle: skip item, return [], etc.
        #         pass

        return content
    except json.JSONDecodeError as json_err:
        st.error(
            f"Error decoding JSON from crawled content file '{input_file}': {json_err}"
        )
        print(f"Error decoding JSON from {input_file}: {json_err}")
        return []
    except Exception as e:
        st.error(f"Error loading crawled content from '{input_file}': {e}")
        print(f"Error loading crawled content from {input_file}: {e}")
        return []


def extract_text_from_crawled_content(content: List[Dict[str, Any]]) -> List[str]:
    """
    Extract text from crawled content.

    Args:
        content: List of dictionaries containing crawled content

    Returns:
        List of text strings
    """
    return [item.get("text", "") for item in content]


def mock_crawl_url(
    url: str, max_pages: int = 10, max_depth: int = 2
) -> List[Dict[str, Any]]:
    """
    Mock implementation of crawl_url for testing or when crawl4ai is not available.

    Args:
        url: The URL to crawl
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum depth of crawling

    Returns:
        List of dictionaries containing mock crawled content
    """
    # Generate mock content
    domain = url.split("//")[-1].split("/")[0]
    path = "/".join(url.split("//")[-1].split("/")[1:])

    results = []
    for i in range(min(3, max_pages)):
        results.append(
            {
                "url": f"{url}/page{i}" if i > 0 else url,
                "title": f"Mock Page {i} for {domain}/{path}",
                "text": f"This is mock content for {url}/page{i if i > 0 else ''}. This would normally contain Markdown-formatted content extracted from the web page, optimized for RAG applications. The content would be cleaned, structured, and ready for embedding and retrieval.",
                "source": url,
                "crawled_at": time.time(),
            }
        )

    return results
    return results
