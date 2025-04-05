# ReflectiveRAG: Self-Correcting Retrieval

## Overview

ReflectiveRAG demonstrates an enhanced Retrieval-Augmented Generation (RAG) system that incorporates self-correction and reflection. The system not only retrieves relevant context and generates an answer but also evaluates the quality of its own retrieval and generation process, potentially triggering re-retrieval or answer refinement.

This application showcases a more sophisticated RAG approach that can detect and correct issues like low relevance, contradictions, or hallucinations before presenting the final answer to the user.

## Features

- **Document Upload**: Upload your own documents to create a knowledge base
- **Self-Correcting Retrieval**: The system evaluates the relevance of retrieved documents and can trigger re-retrieval if needed
- **Answer Faithfulness Check**: Evaluates if the generated answer is faithful to the retrieved context
- **Transparent Process**: Visualizes the entire process from initial query to final answer
- **Reflection Insights**: Provides insights into the system's self-correction process

## How It Works

1. **Initial Retrieval**: When a user asks a question, the system performs an initial retrieval using vector search to find relevant documents
2. **Draft Answer Generation**: The system generates a preliminary answer based on the retrieved context
3. **Self-Correction**: Before showing the answer to the user, the system:
   - Evaluates the relevance of retrieved documents to the query
   - Checks the faithfulness of the answer to the retrieved context
   - Identifies potential issues like hallucinations or contradictions
4. **Re-Retrieval (if needed)**: If issues are detected, the system can:
   - Modify the query to improve retrieval
   - Adjust retrieval parameters
   - Retrieve additional context
5. **Final Answer**: The system generates a refined answer based on the improved context

## Technical Details

- **Vector Search**: Uses embedding-based retrieval for finding semantically relevant documents
- **Self-Evaluation**: Leverages Gemini 2.5 Pro to evaluate retrieval quality and answer faithfulness
- **Reflection Mechanism**: Implements a reflection step to identify and address issues
- **Streamlit UI**: Provides a user-friendly interface with visualization of the entire process

## Usage

1. Upload documents to create your knowledge base
2. Enter your query in the text box
3. View the entire process, including:
   - Initial retrieval results
   - Draft answer
   - Self-correction evaluation
   - Re-retrieval (if performed)
   - Final answer

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI Python SDK
- Other dependencies listed in requirements.txt
