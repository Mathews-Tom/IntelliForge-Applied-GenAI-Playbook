# ContextQuest: Hybrid Retrieval Application

## Overview

ContextQuest demonstrates a Retrieval-Augmented Generation (RAG) system that utilizes a hybrid approach, combining semantic (vector) search with keyword-based search to retrieve relevant context before generating an answer using Google's Gemini 2.5 Pro model.

This application aims to provide more robust and relevant information retrieval compared to using only one search method, leveraging the strengths of both dense vector representations and traditional sparse keyword matching.

## Features

* **Hybrid Search:** Implements both semantic search (using vector embeddings) and keyword search (e.g., BM25).
* **Vector Embeddings:** Ingests documents, chunks them, and creates vector embeddings (specific model TBD - e.g., Google's text-embedding models or alternatives like BGE).
* **Vector Store:** Uses a vector database (e.g., FAISS, ChromaDB - TBD) for efficient similarity search.
* **LLM Integration:** Leverages the **Gemini 2.5 Pro** model via the central `core.llm.gemini_utils` module for final answer generation based on retrieved context.
* **Simple Interface:** (Planned) A basic user interface (e.g., Streamlit/Gradio) for interacting with the application.

## Setup

*(Instructions to be added)*

1. **Prerequisites:**
    * Python 3.x
    * Google API Key (set as `GOOGLE_API_KEY` environment variable)
    * ... (Other dependencies)
2. **Installation:**

    ```bash
    pip install -r apps/context_quest/requirements.txt
    # or pip install -r requirements.txt if using a central file
    ```

3. **Data Preparation:**
    * Place your source documents in the `apps/context_quest/data/` directory.
    * Use the sample data provided or add your own context data in CSV format
4. **Running the Application:**
    * `streamlit run apps/context_quest/src/app.py` (Command TBD based on interface choice)

## Usage

*(Instructions and examples to be added)*

Enter your query into the application interface. The system will:

1. Perform hybrid search over the indexed documents.
2. Retrieve the most relevant text chunks.
3. Pass the query and retrieved context to Gemini 2.5 Pro.
4. Display the generated answer.

## Technology Stack

* **Core LLM:** Google Gemini 2.5 Pro
* **Language:** Python
* **Key Libraries:**
  * `google-generativeai`
  * `python-dotenv`
  * Vector Database (TBD: FAISS, ChromaDB, etc.)
  * Embedding Model Library (TBD: SentenceTransformers, Google AI SDK, etc.)
  * Keyword Search Library (TBD: Rank-BM25, Scikit-learn, etc.)
  * Web Framework (TBD: Streamlit, Gradio, Flask)
  * LangChain (Optional, for structuring RAG pipeline)
