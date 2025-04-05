# ResearchAssistantAgent: Web-Based Research Helper

## Overview

ResearchAssistantAgent is an advanced application that helps users gather, organize, and synthesize information on specific research topics from the web. The system uses targeted web crawling to build comprehensive knowledge bases on user-specified topics, enabling deep exploration and question answering on research subjects.

This application showcases the integration of focused web crawling with advanced RAG techniques to create a powerful research assistant tool.

## Features

- **Topic-Based Web Crawling**: Automatically gather information on specific research topics
- **Multi-Source Research**: Collect data from academic sources, news articles, and specialized websites
- **Research Question Answering**: Ask specific questions about your research topic
- **Citation Management**: Track sources and generate proper citations
- **Research Summaries**: Generate comprehensive summaries of research findings
- **Knowledge Organization**: Categorize and structure research information

## How It Works

1. **Research Topic Specification**: Users provide a research topic or specific research questions
2. **Targeted Crawling**: The system uses crawl4ai to gather relevant information from the web
3. **Knowledge Base Creation**: The extracted content is processed and indexed into a vector database
4. **Research Querying**: Users ask specific questions about their research topic
5. **Information Synthesis**: The system retrieves relevant information and generates comprehensive answers with citations

## Technical Details

- **Focused Web Crawling**: Uses crawl4ai for efficient, targeted extraction of research information
- **Source Credibility Assessment**: Prioritizes information from credible sources
- **Citation Generation**: Automatically formats source information as proper citations
- **LLM Integration**: Uses Gemini 2.5 Pro for sophisticated synthesis of research information
- **Streamlit UI**: Provides a user-friendly interface for research exploration

## Usage

1. Enter a research topic or specific research questions
2. Configure crawling parameters (depth, source types, etc.)
3. Start the research gathering process
4. Once crawling is complete, explore your research topic:
   - Ask specific questions about the topic
   - Generate summaries of key findings
   - Get properly formatted citations for sources
   - Identify gaps in the current research

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI Python SDK
- crawl4ai
- Other dependencies listed in requirements.txt
