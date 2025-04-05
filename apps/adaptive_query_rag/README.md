# AdaptiveQueryRAG: Contextual Strategy Selection

## Overview

AdaptiveQueryRAG demonstrates an advanced Retrieval-Augmented Generation (RAG) system that adapts its retrieval strategy based on query analysis. The system analyzes the user's query to determine the type of information needed and dynamically selects the most appropriate retrieval method, improving relevance and accuracy.

This application showcases a more intelligent approach to RAG that can handle different types of queries with specialized retrieval strategies, rather than using a one-size-fits-all approach.

## Features

- **Query Classification**: Automatically determines query type (factual lookup, summary, comparison, opinion)
- **Dynamic Strategy Selection**: Chooses the most appropriate retrieval method based on query type
- **Strategy-Specific Parameters**: Adjusts retrieval parameters for each strategy
- **Method Weighting**: Dynamically weights different retrieval methods
- **Transparent Process**: Visualizes the query classification and strategy selection process

## How It Works

1. **Query Analysis**: When a user asks a question, the system uses Gemini 2.5 Pro to analyze the query and determine its type
2. **Strategy Selection**: Based on the query type, the system selects the most appropriate retrieval strategy:
   - **Factual Lookup**: Prioritizes dense vector search for specific snippets
   - **Summary**: Uses broader keyword search or retrieves larger chunks
   - **Comparison**: Performs targeted retrievals for each item being compared
   - **Opinion/Analysis**: Uses diverse retrieval to get multiple viewpoints
3. **Parameter Tuning**: Adjusts retrieval parameters (chunk size, similarity threshold, etc.) based on the selected strategy
4. **Retrieval Execution**: Executes the selected retrieval strategy to find the most relevant documents
5. **Answer Generation**: Generates an answer based on the retrieved context

## Technical Details

- **Query Classification**: Uses Gemini 2.5 Pro to analyze query intent and information needs
- **Multiple Retrieval Methods**: Implements various retrieval approaches (vector search, keyword search, hybrid search)
- **Strategy-Specific Logic**: Contains specialized logic for each query type
- **Streamlit UI**: Provides a user-friendly interface with visualization of the strategy selection process

## Usage

1. Upload documents to create your knowledge base
2. Enter your query in the text box
3. View the entire process, including:
   - Query classification results
   - Selected retrieval strategy
   - Retrieved documents
   - Generated answer

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI Python SDK
- Other dependencies listed in requirements.txt
