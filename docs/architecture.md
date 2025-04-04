# IntelliForge: Applied GenAI Playbook - Architecture

This document provides a high-level overview of the architecture and design principles used in the IntelliForge: Applied GenAI Playbook repository.

## System Architecture

The repository is organized as a collection of independent applications that share a common core for Gemini API integration. This modular approach allows each application to focus on its specific domain while leveraging shared utilities for LLM interaction.

```bash
IntelliForge-Applied-GenAI-Playbook/
│
├── apps/                   # Directory for individual applications
│   │
│   ├── context_quest/      # ContextQuest: Hybrid Retrieval
│   │   ├── src/            # Source code
│   │   ├── data/           # Sample data
│   │   ├── notebooks/      # Jupyter notebooks for experimentation/demos
│   │   └── README.md       # App-specific README
│   │
│   ├── fiscal_agent/       # FiscalAgent: Financial Insights
│   │   ├── src/            # Source code
│   │   ├── data/           # Sample data
│   │   ├── notebooks/      # Jupyter notebooks for experimentation/demos
│   │   └── README.md       # App-specific README
│   │
│   ├── graph_query/        # GraphQuery: Knowledge Navigator
│   │   ├── src/            # Source code
│   │   ├── data/           # Sample data
│   │   ├── notebooks/      # Jupyter notebooks for experimentation/demos
│   │   └── README.md       # App-specific README
│   │
│   └── insight_agent/      # InsightAgent: Data Analysis
│       ├── src/            # Source code
│       ├── data/           # Sample data
│       ├── notebooks/      # Jupyter notebooks for experimentation/demos
│       └── README.md       # App-specific README
│
├── core/                   # Shared core components
│   └── llm/                # LLM integration logic
│       └── gemini_utils.py # Gemini API utilities
│
├── docs/                   # Documentation
│   ├── figures/            # IntelliForge application architecture diagrams
│   └── architecture.md     # IntelliForge applications architecture documentations
│
├── .gitignore              # Git ignore file
├── LICENSE                 # Repository license file
└── README.md               # Main repository README
```

## Core Components

### Gemini Integration (`core/llm/gemini_utils.py`)

The central component of the architecture is the Gemini API integration module. This provides:

1. **API Configuration**: Functions to configure the Gemini API client with authentication
2. **Model Selection**: Enum-based model selection with support for different Gemini models
3. **Content Generation**: Standardized functions for generating content with error handling
4. **Type Hinting**: Proper type annotations for better IDE support and code quality

All applications use this shared module to interact with the Gemini API, ensuring consistency in how the LLM is accessed and used.

## Application Architecture

Each application follows a similar structure while implementing domain-specific functionality:

### ContextQuest: Hybrid Retrieval

- **Retrieval Methods**: BM25 and embedding-based retrieval
- **Hybrid Ranking**: Weighted combination of retrieval scores
- **Evaluation**: Relevance assessment of retrieved documents
- **UI Components**: Interactive controls for retrieval parameters
- **Description:** This RAG application allows users to query documents via a Streamlit UI. The backend performs hybrid retrieval: searching a ChromaDB vector store (using embeddings like `text-embedding-004` or `gemini-embedding-exp`) and simultaneously using a keyword-based method (BM25). Results are ranked, combined, and fed as context along with the original query to Gemini 2.5 Pro. The final, context-aware answer is displayed in the UI. (Note: Embedding/Indexing is an offline step).
- **Mermaid Flowchart (Online Query Flow):**

    ```mermaid
    %%{init: {'theme': 'base', 'themeVariables': { 'titleColor': '#333', 'titleFontSize': '20px'}}}%%
    graph LR
        %% Title at the top
        classDef titleClass fill:none,stroke:none,color:#333,font-size:18px,font-weight:bold;
        title["ContextQuest: Hybrid Retrieval Architecture"]:::titleClass;

        subgraph Offline Preparation
            direction TB
            Prep1[Documents] --> Prep2{Text Processing/Chunking};
            Prep2 --> Prep3["Embedding Model <br> (e.g., text-embedding-004)"];
            Prep3 --> Prep4[(ChromaDB Vector Store)];
            Prep2 --> Prep5["Keyword Index <br> (BM25)"];
        end

        subgraph Online Query
            direction LR
            A[User] --> B(Streamlit UI);
            B -- Query --> C{ContextQuest Backend};
            C -- Query --> D[Vector Search];
            D -- Vector Results --> F{Hybrid Ranker};
            C -- Query --> E["Keyword Search <br> (BM25)"];
            E -- Keyword Results --> F;
            F -- Ranked Context --> C;
            C -- Query + Context --> G["core/llm/gemini_utils.py <br> (Gemini 2.5 Pro)"];
            G -- Generated Answer --> C;
            C -- Final Answer --> B;
        end

       %% Link offline stores to online retrieval components
       Prep4 --- D;
       Prep5 --- E;

       %% Position title at the top
       
    ```

### FiscalAgent: Financial Insights

- **Data Sources**: Integration with financial APIs (yfinance)
- **Multi-agent System**: Web search and financial data agents
- **Conversation Storage**: SQLite database for conversation history
- **UI Components**: Tabbed interface for different functionalities
- **Description:** The user interacts via a Streamlit UI (possibly tabbed). Queries are handled by a multi-agent backend. An orchestrator (likely leveraging Gemini logic) routes tasks to specialized agents: a Web Search agent and a Financial Data agent (using yfinance). Gemini 2.5 Pro synthesizes information from these agents and conversation history (stored in SQLite) to provide comprehensive answers, which are displayed in the UI.
- **Mermaid Flowchart:**

    ```mermaid
    %%{init: {'theme': 'base', 'themeVariables': { 'titleColor': '#333', 'titleFontSize': '20px'}}}%%
    graph LR
        %% Title at the top
        classDef titleClass fill:none,stroke:none,color:#333,font-size:18px,font-weight:bold;
        title["FiscalAgent: Financial Insights Architecture"]:::titleClass;

        A[User] --> B(Streamlit UI <br> w/ Tabs);
        B -- NL Query --> C{"FiscalAgent Backend <br> (Multi-Agent System)"};
        C -- Orchestration Logic --> D["core/llm/gemini_utils.py <br> (Gemini 2.5 Pro)"];
        D -- Agent Tasking --> C;
        C -- Web Search Task --> E[Web Search Agent];
        E -- Web Results --> C;
        C -- Financial Data Task --> F["Financial Data Agent <br> (uses yfinance)"];
        F -- Financial Data --> C;
        C -- Store/Retrieve History --> G[(SQLite DB <br> Conversation History)];
        C -- Synthesized Answer --> B;

        %% Position title at the top
        title ~~~ A;
    ```

### GraphQuery: Knowledge Navigator

- **Document Processing**: PDF text extraction and chunking
- **Entity Extraction**: Identification of entities and relationships
- **Graph Construction**: Building and querying knowledge graphs
- **Visualization**: 3D interactive graph visualization
- **Description:** Users upload PDF documents via a Streamlit interface. The backend extracts text (PyPDF2), uses Gemini 2.5 Pro to identify entities and relationships, and builds a knowledge graph (using NetworkX). When the user queries, the backend interprets the query (potentially using Gemini again), queries the NetworkX graph, and sends the retrieved graph context to Gemini 2.5 Pro for answer synthesis. The answer and an interactive 3D visualization of the relevant graph portion are displayed in Streamlit. (Note: Graph building is an initial step).
- **Mermaid Flowchart (Online Query Flow):**

    ```mermaid
    %%{init: {'theme': 'base', 'themeVariables': { 'titleColor': '#333', 'titleFontSize': '20px'}}}%%
    graph TD
       %% Title at the top
       classDef titleClass fill:none,stroke:none,color:#333,font-size:18px,font-weight:bold;
       title["GraphQuery: Knowledge Navigator Architecture"]:::titleClass;

       subgraph Graph Building Phase
           direction LR
           Prep1[PDF Document] --> Prep2[PyPDF2 Text Extraction];
           Prep2 --> Prep3{Text Chunking};
           Prep3 --> Prep4["core/llm/gemini_utils.py <br> (Gemini 2.5 Pro for Entity/Rel Extraction)"];
           Prep4 -- Entities & Relations --> Prep5{Graph Construction};
           Prep5 --> Prep6[(Knowledge Graph <br> NetworkX)];
       end

       subgraph Online Query Phase
           direction LR
            A[User] --> B(Streamlit UI);
            B -- NL Query --> C{GraphQuery Backend};
            C -- Interpret Query / Generate Graph Query --> D["core/llm/gemini_utils.py <br> (Gemini 2.5 Pro)"];
            D -- Graph Query Logic --> C;
            C -- Query Graph --> E[(Knowledge Graph <br> NetworkX)];
            E -- Retrieved Graph Context --> C;
            C -- Query + Graph Context --> D;
            D -- Synthesized Answer --> C;
            C -- Answer + Graph Data --> F[3D Graph Visualization Engine];
            C -- Synthesized Answer --> B;
            F -- Interactive Graph --> B;
       end

       %% Connect Graph Store
       Prep6 --- E;
       %% Visualization might also query the graph directly
       Prep6 ---- F;

       %% Position title at the top
       title ~~~ Graph_Building_Phase[Graph Building Phase];
    ```

### InsightAgent: Data Analysis

- **Data Processing**: Functions to handle CSV and Excel file uploads
- **Query Processing**: Natural language to SQL conversion using Gemini
- **UI Components**: Streamlit interface with data visualization

- **Description:** The user interacts via a Streamlit UI, uploading data (CSV/Excel) and entering natural language queries. The backend uses Gemini 2.5 Pro to interpret the query, potentially converting it to SQL or generating data analysis steps. Pandas is used to process the data according to the generated plan. Results and visualizations (using Plotly) are displayed back in the Streamlit UI.
- **Mermaid Flowchart:**

    ```mermaid
    %%{init: {'theme': 'base', 'themeVariables': { 'titleColor': '#333', 'titleFontSize': '20px'}}}%%
    graph LR
        %% Title at the top
        classDef titleClass fill:none,stroke:none,color:#333,font-size:18px,font-weight:bold;
        title["InsightAgent: Data Analysis Architecture"]:::titleClass;

        A[User] --> B(Streamlit UI);
        B -- Upload Data (CSV/Excel) & NL Query --> C{InsightAgent Backend};
        C -- NL Query --> D["core/llm/gemini_utils.py <br> (Gemini 2.5 Pro)"];
        D -- Analysis Plan / SQL Query --> C;
        C -- Process Data --> E[Pandas Engine];
        E -- Processed Data --> C;
        C -- Generate Visuals --> F[Plotly];
        F -- Visualization Data --> C;
        C -- Results & Visuals --> B;

        %% Position title at the top
        title ~~~ A;
    ```

## Design Principles

The architecture follows these key design principles:

1. **Modularity**: Each application is independent and can be run separately
2. **Reusability**: Common functionality is extracted to shared modules
3. **Consistency**: Similar patterns and approaches across applications
4. **Extensibility**: Easy to add new applications or enhance existing ones
5. **Documentation**: Comprehensive documentation at both repository and application levels

## Technology Stack

- **Language**: Python 3.8+
- **LLM**: Google Gemini 2.5 Pro
- **UI Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Storage**: SQLite, CSV
- **Domain-specific**: yfinance, NetworkX, rank_bm25, PyPDF2

## Security Considerations

- API keys are managed through environment variables or .env files
- No sensitive data is hardcoded in the source code
- User-uploaded data is processed locally and not stored permanently

## Future Architecture Enhancements

Potential enhancements to the architecture include:

1. **Containerization**: Docker containers for each application
2. **API Layer**: RESTful API endpoints for headless operation
3. **Testing Framework**: Comprehensive unit and integration tests
4. **Monitoring**: Telemetry and usage statistics
5. **Authentication**: User authentication and access control
6. **Deployment**: Cloud deployment configurations
