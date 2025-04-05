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


def crawl_url(
    url: str, max_pages: int = 10, max_depth: int = 2, output_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Crawl a URL and extract content optimized for RAG.

    Args:
        url: The URL to crawl
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum depth of crawling
        output_dir: Directory to save crawled content (optional)

    Returns:
        List of dictionaries containing crawled content
    """
    if not check_crawl4ai_installed():
        raise ImportError(
            "crawl4ai is not installed. Please install it with 'pip install crawl4ai'."
        )

    try:
        # Initialize crawler
        crawler = Crawler(
            urls=[url],
            max_pages=max_pages,
            max_depth=max_depth,
            output_dir=output_dir,
            verbose=True,
        )

        # Start crawling
        results = crawler.crawl()

        # Process results into a standardized format
        processed_results = []
        for result in results:
            processed_result = {
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "text": result.get("markdown", result.get("text", "")),
                "source": url,
                "crawled_at": time.time(),
            }
            processed_results.append(processed_result)

        return processed_results
    except Exception as e:
        print(f"Error crawling URL {url}: {e}")
        return []


def crawl_multiple_urls(
    urls: List[str],
    max_pages_per_url: int = 5,
    max_depth: int = 2,
    output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Crawl multiple URLs and extract content optimized for RAG.

    Args:
        urls: List of URLs to crawl
        max_pages_per_url: Maximum number of pages to crawl per URL
        max_depth: Maximum depth of crawling
        output_dir: Directory to save crawled content (optional)

    Returns:
        List of dictionaries containing crawled content
    """
    if not check_crawl4ai_installed():
        raise ImportError(
            "crawl4ai is not installed. Please install it with 'pip install crawl4ai'."
        )

    all_results = []

    for url in urls:
        try:
            # Crawl each URL
            results = crawl_url(
                url=url,
                max_pages=max_pages_per_url,
                max_depth=max_depth,
                output_dir=output_dir,
            )

            all_results.extend(results)
        except Exception as e:
            print(f"Error crawling URL {url}: {e}")

    return all_results


def crawl_topic(
    topic: str,
    num_results: int = 5,
    max_pages_per_result: int = 3,
    max_depth: int = 1,
    output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Crawl web pages related to a topic using search results.

    Args:
        topic: The topic to search for
        num_results: Number of search results to crawl
        max_pages_per_result: Maximum number of pages to crawl per search result
        max_depth: Maximum depth of crawling
        output_dir: Directory to save crawled content (optional)

    Returns:
        List of dictionaries containing crawled content
    """
    if not check_crawl4ai_installed():
        raise ImportError(
            "crawl4ai is not installed. Please install it with 'pip install crawl4ai'."
        )

    try:
        # This is a simplified implementation
        # In a real application, you would use a search API to get URLs for the topic
        # For now, we'll use a mock implementation

        # Mock search results (in a real app, this would come from a search API)
        search_results = [
            f"https://example.com/search/{topic}/result1",
            f"https://example.com/search/{topic}/result2",
            f"https://example.com/search/{topic}/result3",
            f"https://example.com/search/{topic}/result4",
            f"https://example.com/search/{topic}/result5",
        ]

        # Limit to requested number of results
        search_results = search_results[:num_results]

        # Crawl each search result
        return crawl_multiple_urls(
            urls=search_results,
            max_pages_per_url=max_pages_per_result,
            max_depth=max_depth,
            output_dir=output_dir,
        )
    except Exception as e:
        print(f"Error crawling topic {topic}: {e}")
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


def load_crawled_content(input_file: str) -> List[Dict[str, Any]]:
    """
    Load crawled content from a file.

    Args:
        input_file: Path to the content file

    Returns:
        List of dictionaries containing crawled content
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = json.load(f)

        return content
    except Exception as e:
        print(f"Error loading crawled content: {e}")
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
