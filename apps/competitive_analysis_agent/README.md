# CompetitiveAnalysisAgent: Web-Based Competitor Intelligence

## Overview

CompetitiveAnalysisAgent is an advanced application that gathers, analyzes, and synthesizes information about competitors from their websites and other online sources. The system uses web crawling to build comprehensive knowledge bases about specified competitors, enabling users to ask comparative questions and gain competitive intelligence insights.

This application showcases the integration of targeted web crawling with advanced RAG techniques to create a powerful competitive analysis tool.

## Features

- **Competitor Website Crawling**: Automatically extract information from competitor websites
- **Multi-Source Intelligence**: Gather data from product pages, about pages, news sections, and more
- **Comparative Analysis**: Ask questions that compare multiple competitors
- **Feature Extraction**: Identify and compare product features, pricing, and positioning
- **News Monitoring**: Track recent mentions and news about competitors
- **Structured Insights**: Generate structured reports and comparisons

## How It Works

1. **Competitor Specification**: Users provide competitor website URLs or company names
2. **Targeted Crawling**: The system uses crawl4ai to crawl competitor websites, focusing on key sections
3. **Knowledge Base Creation**: The extracted content is processed and indexed into a vector database
4. **Comparative Querying**: Users ask questions about competitors individually or comparatively
5. **Intelligence Synthesis**: The system retrieves relevant information and generates comprehensive answers

## Technical Details

- **Focused Web Crawling**: Uses crawl4ai for efficient, targeted extraction of competitor information
- **Structured Data Extraction**: Identifies and extracts product features, pricing, and other key data points
- **Comparative Analysis**: Specialized prompting for comparing information across multiple sources
- **LLM Integration**: Uses Gemini 2.5 Pro for sophisticated synthesis of competitive intelligence
- **Streamlit UI**: Provides a user-friendly interface for competitor analysis

## Usage

1. Enter competitor website URLs or company names
2. Configure crawling parameters (depth, focus areas, etc.)
3. Start the crawling process
4. Once crawling is complete, ask questions about competitors:
   - "What are Competitor A's main product features?"
   - "Compare pricing between Competitor A and Competitor B"
   - "What are recent news mentions of Competitor C?"
   - "How does Competitor A position their products compared to Competitor B?"

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI Python SDK
- crawl4ai
- Other dependencies listed in requirements.txt
