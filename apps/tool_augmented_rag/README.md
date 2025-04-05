# ToolAugmentedRAG: Retrieval + Live Data Integration

## Overview

ToolAugmentedRAG demonstrates an advanced Retrieval-Augmented Generation (RAG) system that combines static document retrieval with dynamic data from external tools and APIs. The system analyzes queries to determine when live data is needed and integrates this information with retrieved context to provide comprehensive, up-to-date answers.

This application showcases a more powerful approach to RAG that can handle queries requiring both background knowledge and current information, bridging the gap between static knowledge bases and real-time data.

## Features

- **Tool Detection**: Automatically identifies when external tools are needed
- **API Integration**: Connects to external data sources for real-time information
- **Data Synthesis**: Combines static and dynamic information in a coherent answer
- **Tool Selection**: Chooses appropriate tools based on the query
- **Transparent Process**: Visualizes the tool selection and data integration process

## How It Works

1. **Document Retrieval**: The system first retrieves relevant background information from static documents
2. **Tool Detection**: Gemini 2.5 Pro analyzes the query to determine if it requires live data
3. **Tool Selection**: If live data is needed, the system selects the appropriate tool(s):
   - **Stock Tool**: For queries about current stock prices or financial data
   - **Weather Tool**: For queries about current weather conditions
   - **Web Search Tool**: For queries requiring recent information from the web
4. **Tool Execution**: The selected tool is executed to retrieve the live data
5. **Data Integration**: The system combines the static context with the live data
6. **Answer Generation**: Gemini 2.5 Pro generates a comprehensive answer using both information sources

## Technical Details

- **Tool Framework**: Modular design for easy addition of new tools
- **API Integrations**: Connections to financial, weather, and search APIs
- **Context Merging**: Techniques for combining static and dynamic information
- **Tool-Aware Prompting**: Specialized prompting to generate answers that incorporate tool outputs
- **Streamlit UI**: Provides a user-friendly interface with clear presentation of tool usage

## Usage

1. Upload documents to create your static knowledge base
2. Enter your query in the text box
3. The system automatically detects if tools are needed
4. View the answer that combines static knowledge with live data
5. Explore the "Tool Usage" section to see which tools were used and their outputs

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI Python SDK
- yfinance (for stock data)
- requests (for API calls)
- Other dependencies listed in requirements.txt
