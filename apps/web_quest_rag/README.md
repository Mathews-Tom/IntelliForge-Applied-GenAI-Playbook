# WebQuestRAG: Dynamic Web RAG Agent

## Overview

WebQuestRAG is an advanced Retrieval-Augmented Generation (RAG) application that dynamically builds knowledge bases from web content. Unlike traditional RAG systems that rely on pre-loaded documents, WebQuestRAG crawls the web in real-time to gather information on topics specified by the user, creating a fresh, up-to-date knowledge base for each query session.

This application showcases the integration of web crawling with RAG techniques, allowing users to ask questions about web content without having to manually download or process web pages.

## Features

- **Dynamic Web Crawling**: Crawl websites or topics in real-time using crawl4ai
- **RAG-Optimized Content**: Extract clean, structured content from web pages optimized for RAG
- **On-the-Fly Knowledge Base**: Create temporary or persistent knowledge bases from crawled content
- **Multi-Source Synthesis**: Combine information from multiple web sources to answer questions
- **Source Attribution**: Clearly attribute information to specific web sources

## How It Works

1. **Web Content Acquisition**: Users provide a starting URL, a list of URLs, or a topic/set of keywords
2. **Dynamic Crawling**: The system uses crawl4ai to crawl the relevant web pages and extract RAG-optimized content
3. **Knowledge Base Creation**: The extracted content is indexed into a vector database
4. **Query Processing**: Users ask questions about the crawled content
5. **RAG-Based Answering**: The system retrieves relevant information from the dynamically created knowledge base and generates comprehensive answers

## Technical Details

- **Web Crawling**: Uses crawl4ai for efficient, RAG-optimized web content extraction
- **Content Processing**: Handles Markdown output from crawl4ai, preserving structure and context
- **Vector Storage**: Indexes crawled content for efficient retrieval
- **LLM Integration**: Uses Gemini 2.5 Pro for answer generation
- **Streamlit UI**: Provides a user-friendly interface for crawling and querying

## Usage

1. Enter a starting URL, a list of URLs, or a topic to crawl
2. Configure crawling parameters (depth, max pages, etc.)
3. Start the crawling process
4. Once crawling is complete, ask questions about the crawled content
5. View answers with source attribution to specific web pages

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI Python SDK
- crawl4ai
- Other dependencies listed in requirements.txt
