# IntelliForge: Applied GenAI Playbook - Project Overview

- **Version:** 1.1 (Added Web Crawling Apps)
- **GitHub Repository:** [Mathews-Tom/IntelliForge-Applied-GenAI-Playbook](https://github.com/Mathews-Tom/IntelliForge-Applied-GenAI-Playbook)
- **License:** [MIT License](https://opensource.org/licenses/MIT)

## 1. Introduction

Welcome to IntelliForge, a curated collection of practical demonstration applications designed to showcase the power and versatility of Google's **Gemini 2.5 Pro** large language model. This repository serves as a hands-on playbook for developers, researchers, and enthusiasts interested in exploring applied Generative AI (GenAI) and Agentic AI concepts.

The primary goal of IntelliForge is to move beyond theoretical discussions and provide concrete, runnable examples of how state-of-the-art AI can be integrated into functional applications across various domains, including data analysis, financial insights, knowledge management, and **advanced information retrieval incorporating both static documents and dynamic web content**. This version now includes **eleven** distinct applications.

## 2. Core Concepts Demonstrated

IntelliForge focuses particularly on illustrating sophisticated **Retrieval-Augmented Generation (RAG)** techniques, alongside other AI patterns:

- **Standard RAG:** Basic retrieval from a knowledge base to ground LLM responses (ContextQuest, GraphQuery).
- **Hybrid Retrieval:** Combining keyword (BM25) and semantic (vector) search for improved relevance (ContextQuest).
- **Graph RAG:** Using knowledge graphs (NetworkX) for structured information retrieval (GraphQuery).
- **Agentic Behavior:** Employing LLMs to orchestrate tasks and use external tools (FiscalAgent, ToolAugmentedRAG, CompetitiveAnalysisAgent).
- **Self-Correction/Reflection (Enhanced RAG):** Systems that evaluate and refine their own retrieval/generation process (ReflectiveRAG).
- **Adaptive/Conditional RAG:** Dynamically adjusting the retrieval strategy based on query analysis (AdaptiveQueryRAG).
- **Multi-Source Synthesis (Enhanced RAG):** Handling and synthesizing information from multiple, potentially conflicting sources (MultiPerspectiveSynth).
- **Tool Augmentation (Agentic RAG):** Integrating real-time API data into the RAG workflow (ToolAugmentedRAG).
- **Dynamic Web RAG:** Building knowledge bases on-the-fly from dynamically crawled web content using **`crawl4ai`** (WebQuestRAG, CompetitiveAnalysisAgent, ResearchAssistantAgent, enhanced ContextQuest).
- **Targeted Web Crawling:** Focused information gathering from specific websites or topics using **`crawl4ai`** (CompetitiveAnalysisAgent, ResearchAssistantAgent).
- **Natural Language Interfaces:** Using Streamlit to create intuitive conversational interfaces for complex backend processes.

## 3. Overall System Architecture

IntelliForge employs a modular architecture where individual applications reside in the `apps/` directory. These applications operate independently but leverage shared components located in the `core/` directory, primarily for Gemini API interaction and common utility functions. The recent addition of web crawling capabilities utilizes the `crawl4ai` library, integrated via a shared utility (`core/utils/web_crawler.py`).

```bash
IntelliForge-Applied-GenAI-Playbook/
│
├── apps/                   # Directory for individual applications
│   │
│   ├── context_quest/      # ContextQuest: Hybrid Retrieval with optional web source
│   │   ├── src/
│   │   ├── data/
│   │   └── README.md
│   ├── fiscal_agent/       # FiscalAgent: Financial Insights
│   │   ├── src/
│   │   ├── data/
│   │   └── README.md
│   ├── graph_query/        # GraphQuery: Knowledge Navigator
│   │   ├── src/
│   │   ├── data/
│   │   └── README.md
│   ├── insight_agent/      # InsightAgent: Data Analysis
│   │   ├── src/
│   │   ├── data/
│   │   └── README.md
│   ├── reflective_rag/     # ReflectiveRAG: Self-Correcting Retrieval
│   │   ├── src/
│   │   ├── data/
│   │   └── README.md
│   ├── adaptive_query_rag/ # AdaptiveQueryRAG: Contextual Strategy Selection
│   │   ├── src/
│   │   ├── data/
│   │   └── README.md
│   ├── multi_perspective_synth/ # MultiPerspectiveSynth: Synthesizing Diverse Sources
│   │   ├── src/
│   │   ├── data/
│   │   └── README.md
│   ├── tool_augmented_rag/ # ToolAugmentedRAG: Retrieval + Live Data Integration
│   │   ├── src/
│   │   ├── data/
│   │   └── README.md
│   ├── web_quest_rag/      # WebQuestRAG: Dynamic Web RAG Agent
│   │   ├── src/
│   │   ├── data/ (Optional: for saving crawl results?)
│   │   └── README.md
│   │
│   ├── competitive_analysis_agent/ # CompetitiveAnalysisAgent: Web Competitor Intelligence
│   │   ├── src/
│   │   ├── data/ (Optional: for saving crawl results?)
│   │   └── README.md
│   │
│   └── research_assistant_agent/ # ResearchAssistantAgent: Web Research Helper
│       ├── src/
│       ├── data/ (Optional: for saving crawl results?)
│       └── README.md
│
├── core/                   # Shared core components
│   ├── llm/                # LLM integration logic
│   │   └── gemini_utils.py # Gemini API utilities
│   └── utils/              # Shared utility functions
│       ├── __init__.py     # Package initialization
│       ├── data_helpers.py # Data loading and processing (documents)
│       ├── ui_helpers.py   # Streamlit UI components
│       ├── file_io.py      # File handling operations
│       ├── retrieval_utils.py # Retrieval and embedding functions (for indexed data)
│       ├── evaluation.py   # Evaluation metrics and functions
│       └── web_crawler.py  # NEW: Web crawling using crawl4ai
│
├── docs/                   # Documentation
│   ├── apps/               # Detailed app docs
│   │   └── ...             # <<< Files for each app
│   ├── figures/            # Diagrams (if any)
│   ├── architecture.md     # (Legacy/Source) Architecture notes
│   └── overview.md         # <<< This file
│
├── .env.example            # Example environment file
├── .gitignore              # Git ignore file
├── LICENSE                 # Repository license file (Should contain MIT License text)
└── README.md               # Main repository README (links here and to apps)

```

*(Note: Corrected the `docs/` structure in the diagram based on your project)*

## 4. Core Components

### 4.1. Gemini Integration (`core/llm/gemini_utils.py`)

This module is the heart of the LLM interaction layer. It provides a standardized and reusable way for all applications to interact with the Google Gemini API.

- **API Key Configuration:** Securely loads the `GOOGLE_API_KEY` using `python-dotenv`.
- **Model Selection:** Uses a Python `Enum` (`GeminiModelType`) to allow selection of specific Gemini models (defaulting to **Gemini 2.5 Pro**).
- **Content Generation Function:** Offers a robust `generate_content` function with basic error handling and retry logic (if implemented).
- **Type Hinting:** Ensures code quality and developer experience.

### 4.2. Shared Utilities (`core/utils/`)

This package encapsulates common functionalities reused across multiple applications, promoting code reuse and consistency.

- **`data_helpers.py`:** Functions for loading documents (PDF, TXT, CSV), text splitting/chunking strategies, cleaning text, and creating document metadata for *static* files.
- **`ui_helpers.py`:** Reusable Streamlit components (e.g., standard page layout templates, custom widgets, functions for displaying data or chat messages consistently).
- **`file_io.py`:** Generic functions for reading/writing common file formats, handling uploads/downloads within Streamlit.
- **`retrieval_utils.py`:** Core logic for vector embedding generation (interfacing with models like `text-embedding-004`), managing vector stores (e.g., ChromaDB), implementing keyword search (BM25), combining search results (hybrid ranking/RRF), and document reranking *for indexed data*.
- **`evaluation.py`:** Functions for evaluating RAG performance, such as calculating retrieval metrics (Precision, Recall) or implementing LLM-based checks for answer faithfulness and context relevance (used heavily by ReflectiveRAG).
- **`web_crawler.py` (New):** Provides functions leveraging the **`crawl4ai`** library to perform web crawling tasks. This includes fetching web pages based on provided URLs or topics, extracting clean, RAG-optimized Markdown content, and handling basic crawl configurations (like depth or page limits). This utility powers the dynamic web capabilities of the newer applications and the enhanced ContextQuest.

## 5. Design Principles

The development of IntelliForge adheres to the following principles:

1. **Modularity:** Each application in `apps/` is self-contained and can, in principle, be run independently, minimizing inter-application dependencies.
2. **Reusability:** Common functionalities (LLM access, data handling, retrieval, web crawling) are centralized in `core/` to avoid code duplication.
3. **Consistency:** Applications strive for similar project structures (`src/`, `data/`, `README.md`) and UI interaction patterns where applicable.
4. **Extensibility:** The modular design makes it relatively straightforward to add new applications showcasing different AI concepts or Gemini features.
5. **Clarity & Documentation:** Emphasis on clear code, type hinting, and comprehensive documentation (like this file, the main README, and individual app READMEs/docs).
6. **Focus on Concepts:** Each app is designed primarily to illustrate specific AI techniques rather than being a production-ready tool.

## 6. Technology Stack (Overall)

- **Primary Language:** Python (3.8+)
- **Core AI Model:** Google Gemini 2.5 Pro
- **AI SDK:** `google-generativeai`
- **Web UI Framework:** Streamlit
- **Web Crawling:** **`crawl4ai`**
- **Data Handling:** Pandas, NumPy
- **Configuration:** `python-dotenv`
- **Keyword Search:** `rank_bm25`
- **Vector Search:** (Example: `chromadb-client`)
- **PDF Processing:** `PyPDF2`
- **Graph Handling:** `NetworkX`
- **Visualization:** Plotly
- **Financial Data:** `yfinance`

*(Refer to individual application documentation for specifics)*

## 7. Installation and Setup

Please refer to the main [README.md](../README.md#installation) for the general steps to clone the repository and set up your Google API Key via the `.env` file.

Detailed installation steps, including specific Python dependencies (`requirements.txt`) and environment setup for each application, can be found in the documentation for the respective application within the `docs/apps/` directory. **Note:** Applications using web crawling features will require installing `crawl4ai` as part of their dependencies.

## 8. Security Considerations

- **API Keys:** Never hardcode API keys. Use the `.env` file mechanism. Ensure `.env` is included in your `.gitignore`.
- **Data Handling:** Applications are designed to process user-uploaded data locally during runtime. No persistent storage of user data is implemented unless explicitly mentioned (e.g., conversation history in FiscalAgent's SQLite DB). Be mindful of data privacy if adapting these demos.
- **Web Crawling Ethics:** When using web crawling features (`crawl4ai`), ensure your usage respects `robots.txt` policies and website terms of service. Be a polite and responsible crawler; avoid overwhelming websites with rapid or excessive requests. `crawl4ai` likely incorporates politeness delays, but responsible configuration and usage are essential.
- **Dependencies:** Keep dependencies up-to-date to mitigate security vulnerabilities. Use virtual environments. The addition of `crawl4ai` introduces web-related dependencies; keep these updated as well.
- **LLM Safety:** Rely on Google's built-in safety settings for the Gemini API. Implementations here do not add extensive custom input/output filtering beyond basic error handling.

## 9. Future Enhancements (Potential Roadmap)

- **Containerization:** Provide Dockerfiles for easier setup and deployment of each application.
- **Testing Framework:** Implement unit and integration tests for core components and application logic.
- **API Layer:** Expose functionality via REST APIs (e.g., using FastAPI) for headless operation or integration with other systems.
- **Monitoring/Logging:** Add more structured logging and potentially basic telemetry.
- **Deployment Examples:** Provide configurations or guides for deploying apps to cloud platforms (e.g., Google Cloud Run, Streamlit Community Cloud).
- **Advanced Error Handling:** Implement more robust error handling and user feedback mechanisms.
- **User Authentication:** Add options for user accounts and access control if deploying in a multi-user setting.
- **Enhanced Crawler Configuration:** More UI options for `crawl4ai` settings (depth, exclusions, selectors, user-agent, javascript rendering toggle).
- **Hybrid Dynamic/Static RAG:** More sophisticated blending strategies for combining dynamically crawled content with pre-indexed static documents within a single query.
- **Crawler State Management:** Implementing ways to save/resume large crawls or persistently store crawled data for specific projects.

## 10. Contributing

Contributions are highly encouraged! Please refer to the main [README.md](../README.md#contributing) for basic guidelines. Consider opening an issue first to discuss proposed changes or features.
