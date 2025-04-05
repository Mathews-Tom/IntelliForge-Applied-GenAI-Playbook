# MultiPerspectiveSynth: Synthesizing Diverse Sources

## Overview

MultiPerspectiveSynth demonstrates an advanced Retrieval-Augmented Generation (RAG) system that retrieves and synthesizes information from multiple, potentially conflicting sources. The system identifies different perspectives on a topic and generates balanced answers that highlight areas of agreement and disagreement.

This application showcases a more sophisticated approach to RAG that can handle diverse viewpoints and present a comprehensive synthesis rather than a single perspective.

## Features

- **Multi-Source Document Handling**: Process and track documents from different sources
- **Perspective Identification**: Detect different viewpoints and stances on topics
- **Agreement Analysis**: Identify consensus and disagreement points across sources
- **Balanced Synthesis**: Generate answers that represent diverse perspectives
- **Source Attribution**: Clearly attribute information to specific sources

## How It Works

1. **Document Upload**: Users upload multiple documents that might discuss the same topic from different viewpoints
2. **Source Tracking**: The system maintains metadata about each document's source
3. **Multi-Source Retrieval**: When a query is made, the system retrieves relevant chunks from all applicable sources
4. **Perspective Analysis**: The system identifies different perspectives on the topic across the retrieved documents
5. **Synthesis Generation**: Gemini 2.5 Pro synthesizes the information, explicitly highlighting areas of agreement, disagreement, or differing perspectives
6. **Balanced Presentation**: The UI displays the answer with inline citations and a separate section summarizing key perspectives

## Technical Details

- **Source Metadata**: Tracks and preserves document sources throughout the retrieval process
- **Multi-Source Retrieval**: Ensures representation from different sources in the retrieved context
- **Perspective Detection**: Uses Gemini 2.5 Pro to identify and categorize different viewpoints
- **Synthesis Prompting**: Specialized prompting to generate balanced, multi-perspective answers
- **Streamlit UI**: Provides a user-friendly interface with clear presentation of different perspectives

## Usage

1. Upload multiple documents from different sources
2. Enter your query in the text box
3. View the synthesized answer that incorporates multiple perspectives
4. Explore the "Perspectives Analysis" section to see identified viewpoints
5. Check source attributions to understand where information comes from

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI Python SDK
- Other dependencies listed in requirements.txt
