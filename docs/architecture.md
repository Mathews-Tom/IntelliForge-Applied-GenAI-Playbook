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
│   ├── insight_agent/      # InsightAgent: Data Analysis
│   │   ├── src/            # Source code
│   │   ├── data/           # Sample data
│   │   ├── notebooks/      # Jupyter notebooks for experimentation/demos
│   │   └── README.md       # App-specific README
│   │
│   ├── reflective_rag/     # ReflectiveRAG: Self-Correcting Retrieval
│   │   ├── src/            # Source code
│   │   ├── data/           # Sample data
│   │   └── README.md       # App-specific README
│   │
│   ├── adaptive_query_rag/ # AdaptiveQueryRAG: Contextual Strategy Selection
│   │   ├── src/            # Source code
│   │   ├── data/           # Sample data
│   │   └── README.md       # App-specific README
│   │
│   ├── multi_perspective_synth/ # MultiPerspectiveSynth: Synthesizing Diverse Sources
│   │   ├── src/            # Source code
│   │   ├── data/           # Sample data
│   │   └── README.md       # App-specific README
│   │
│   └── tool_augmented_rag/ # ToolAugmentedRAG: Retrieval + Live Data Integration
│       ├── src/            # Source code
│       ├── data/           # Sample data
│       └── README.md       # App-specific README
│
├── core/                   # Shared core components
│   ├── llm/                # LLM integration logic
│   │   └── gemini_utils.py # Gemini API utilities
│   └── utils/              # Shared utility functions
│       ├── __init__.py     # Package initialization
│       ├── data_helpers.py # Data loading and processing
│       ├── ui_helpers.py   # Streamlit UI components
│       ├── file_io.py      # File handling operations
│       ├── retrieval_utils.py # Retrieval and embedding functions
│       └── evaluation.py   # Evaluation metrics and functions
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

### Shared Utilities (`core/utils/`)

The shared utilities package provides common functionality used across multiple applications:

1. **Data Helpers (`data_helpers.py`)**: Functions for loading, processing, and manipulating data
   - Document loading from various formats
   - Text chunking and cleaning
   - Document metadata creation

2. **UI Helpers (`ui_helpers.py`)**: Streamlit UI components and patterns
   - Standard headers, footers, and sidebars
   - Document display functions
   - Query input components
   - Processing step visualization

3. **File I/O (`file_io.py`)**: File handling operations
   - File upload and saving
   - CSV, JSON, and text file reading/writing
   - File type detection

4. **Retrieval Utilities (`retrieval_utils.py`)**: Document retrieval functions
   - Embedding generation
   - BM25 keyword retrieval
   - Vector-based semantic retrieval
   - Hybrid retrieval combining multiple methods
   - Document reranking

5. **Evaluation (`evaluation.py`)**: Metrics and evaluation functions
   - Retrieval relevance evaluation
   - Answer faithfulness checking
   - Precision, recall, and F1 calculation

## Application Architecture

Each application follows a similar structure while implementing domain-specific functionality:

### ContextQuest: Hybrid Retrieval

- **Description:** This RAG application allows users to query documents via a Streamlit UI. The backend performs hybrid retrieval: searching a ChromaDB vector store (using embeddings like `text-embedding-004` or `gemini-embedding-exp`) and simultaneously using a keyword-based method (BM25). Results are ranked, combined, and fed as context along with the original query to Gemini 2.5 Pro. The final, context-aware answer is displayed in the UI. (Note: Embedding/Indexing is an offline step).
- **Retrieval Methods**: BM25 and embedding-based retrieval
- **Hybrid Ranking**: Weighted combination of retrieval scores
- **Evaluation**: Relevance assessment of retrieved documents
- **UI Components**: Interactive controls for retrieval parameters
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

- **Description:** The user interacts via a Streamlit UI (possibly tabbed). Queries are handled by a multi-agent backend. An orchestrator (likely leveraging Gemini logic) routes tasks to specialized agents: a Web Search agent and a Financial Data agent (using yfinance). Gemini 2.5 Pro synthesizes information from these agents and conversation history (stored in SQLite) to provide comprehensive answers, which are displayed in the UI.
- **Data Sources**: Integration with financial APIs (yfinance)
- **Multi-agent System**: Web search and financial data agents
- **Conversation Storage**: SQLite database for conversation history
- **UI Components**: Tabbed interface for different functionalities
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

- **Description:** Users upload PDF documents via a Streamlit interface. The backend extracts text (PyPDF2), uses Gemini 2.5 Pro to identify entities and relationships, and builds a knowledge graph (using NetworkX). When the user queries, the backend interprets the query (potentially using Gemini again), queries the NetworkX graph, and sends the retrieved graph context to Gemini 2.5 Pro for answer synthesis. The answer and an interactive 3D visualization of the relevant graph portion are displayed in Streamlit. (Note: Graph building is an initial step).
- **Document Processing**: PDF text extraction and chunking
- **Entity Extraction**: Identification of entities and relationships
- **Graph Construction**: Building and querying knowledge graphs
- **Visualization**: 3D interactive graph visualization
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

- **Description:** The user interacts via a Streamlit UI, uploading data (CSV/Excel) and entering natural language queries. The backend uses Gemini 2.5 Pro to interpret the query, potentially converting it to SQL or generating data analysis steps. Pandas is used to process the data according to the generated plan. Results and visualizations (using Plotly) are displayed back in the Streamlit UI.
- **Mermaid Flowchart:**
- **Data Processing**: Functions to handle CSV and Excel file uploads
- **Query Processing**: Natural language to SQL conversion using Gemini
- **UI Components**: Streamlit interface with data visualization

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

### ReflectiveRAG: Self-Correcting Retrieval

- **Description:** This enhanced RAG application adds a self-correction layer to the retrieval process. After initial retrieval and draft answer generation, the system uses Gemini 2.5 Pro to evaluate the relevance of retrieved documents and the faithfulness of the generated answer. If issues are detected, it can reformulate the query, perform re-retrieval, and refine the answer. The entire process is visualized in the UI, making the system's "thinking" transparent to the user.
- **Self-Correction**: Evaluation and improvement of retrieval and generation
- **Query Reformulation**: Automatic query refinement based on evaluation
- **Answer Faithfulness**: Checking and refining answers for factual accuracy
- **Process Visualization**: Transparent display of the reflection process
- **Mermaid Flowchart:**

    ```mermaid
    %%{init: {'theme': 'base', 'themeVariables': { 'titleColor': '#333', 'titleFontSize': '20px'}}}%%
    graph TD
        %% Title at the top
        classDef titleClass fill:none,stroke:none,color:#333,font-size:18px,font-weight:bold;
        title["ReflectiveRAG: Self-Correcting Retrieval Architecture"]:::titleClass;

        A[User] --> B(Streamlit UI);
        B -- Query --> C{ReflectiveRAG Backend};

        %% Initial Retrieval
        C --> D[Initial Retrieval];
        D --> E[Vector Search];
        E -- Retrieved Documents --> F[Draft Answer Generation];

        %% Self-Correction Loop
        F -- Draft Answer --> G{Self-Correction};
        G -- Retrieval Evaluation --> H["core/llm/gemini_utils.py <br> (Gemini 2.5 Pro)"];
        H -- Evaluation Results --> G;
        G -- Answer Faithfulness Check --> H;

        %% Conditional Re-Retrieval
        G -- Issues Detected --> I[Query Reformulation];
        I --> J[Re-Retrieval];
        J -- New Documents --> K[Answer Refinement];

        %% Final Output
        G -- No Issues --> L[Final Answer];
        K -- Refined Answer --> L;
        L --> B;

        %% Position title at the top
        title ~~~ A;
    ```

### AdaptiveQueryRAG: Contextual Strategy Selection

- **Query Classification**: Determining query type and information needs
- **Strategy Selection**: Choosing appropriate retrieval methods based on query type
- **Parameter Tuning**: Adjusting retrieval parameters for each strategy
- **Method Weighting**: Dynamic weighting of different retrieval methods

- **Description:** This application analyzes the user's query to determine the type of information needed (factual lookup, summary, comparison, opinion) and dynamically selects the most appropriate retrieval strategy. For example, factual queries might prioritize dense vector search, while summary requests might use broader keyword search. The system adapts its retrieval approach to match the specific information need, improving relevance and accuracy.
- **Mermaid Flowchart:**

    ```mermaid
    %%{init: {'theme': 'base', 'themeVariables': { 'titleColor': '#333', 'titleFontSize': '20px'}}}%%
    graph LR
        %% Title at the top
        classDef titleClass fill:none,stroke:none,color:#333,font-size:18px,font-weight:bold;
        title["AdaptiveQueryRAG: Contextual Strategy Selection Architecture"]:::titleClass;

        A[User] --> B(Streamlit UI);
        B -- Query --> C{AdaptiveQueryRAG Backend};

        %% Query Analysis
        C --> D[Query Analyzer];
        D --> E["core/llm/gemini_utils.py <br> (Gemini 2.5 Pro)"];
        E -- Query Classification --> D;

        %% Strategy Selection
        D -- Query Type --> F{Strategy Selector};
        F -- Factual Query --> G[Dense Vector Search];
        F -- Summary Query --> H[Keyword Search];
        F -- Comparison Query --> I[Multi-Query Retrieval];
        F -- Opinion Query --> J[Diverse Retrieval];

        %% Results Combination
        G --> K{Results Combiner};
        H --> K;
        I --> K;
        J --> K;

        %% Answer Generation
        K -- Retrieved Context --> L[Answer Generator];
        L --> E;
        E -- Generated Answer --> C;
        C -- Final Answer --> B;

        %% Position title at the top
        title ~~~ A;
    ```

### MultiPerspectiveSynth: Synthesizing Diverse Sources

- **Description:** This application allows users to upload multiple documents that might discuss the same topic from different viewpoints. When a query is made, the system retrieves relevant chunks from all applicable sources and uses Gemini 2.5 Pro to synthesize the information, explicitly highlighting areas of agreement, disagreement, or differing perspectives. The UI displays the answer with inline citations and a separate section summarizing key perspectives.
- **Multi-Source Handling**: Processing documents from different sources
- **Perspective Identification**: Detecting viewpoints and stances
- **Agreement Analysis**: Identifying consensus and disagreement points
- **Balanced Synthesis**: Generating balanced answers from diverse perspectives
- **Mermaid Flowchart:**

    ```mermaid
    %%{init: {'theme': 'base', 'themeVariables': { 'titleColor': '#333', 'titleFontSize': '20px'}}}%%
    graph TD
        %% Title at the top
        classDef titleClass fill:none,stroke:none,color:#333,font-size:18px,font-weight:bold;
        title["MultiPerspectiveSynth: Synthesizing Diverse Sources Architecture"]:::titleClass;

        A[User] --> B(Streamlit UI);
        B -- Upload Multiple Documents --> C{Document Processor};
        C -- Process & Index --> D[(Multi-Source Document Store)];

        B -- Query --> E{MultiPerspectiveSynth Backend};
        E -- Query --> F[Multi-Source Retrieval];
        F -- Retrieve From --> D;

        F -- Retrieved Documents --> G[Perspective Analyzer];
        G --> H["core/llm/gemini_utils.py <br> (Gemini 2.5 Pro)"];
        H -- Identified Perspectives --> G;

        G -- Analyzed Perspectives --> I[Synthesis Generator];
        I --> H;
        H -- Synthesized Answer --> I;

        I -- Final Answer with Perspectives --> E;
        E -- Answer + Perspective Summary --> B;

        %% Position title at the top
        title ~~~ A;
    ```

### ToolAugmentedRAG: Retrieval + Live Data Integration

- **Description:** This application extends basic RAG by combining static document retrieval with dynamic data from external tools and APIs. When a user asks a question, the system first retrieves relevant background information from static documents. It then analyzes the query to see if it requires live data (e.g., current stock prices, weather) and triggers the appropriate tool if needed. The final answer synthesizes both the static context and the live data.
- **Tool Detection**: Identifying when external tools are needed
- **API Integration**: Connecting to external data sources
- **Data Synthesis**: Combining static and dynamic information
- **Tool Selection**: Choosing appropriate tools based on the query
- **Mermaid Flowchart:**

    ```mermaid
    %%{init: {'theme': 'base', 'themeVariables': { 'titleColor': '#333', 'titleFontSize': '20px'}}}%%
    graph TD
        %% Title at the top
        classDef titleClass fill:none,stroke:none,color:#333,font-size:18px,font-weight:bold;
        title["ToolAugmentedRAG: Retrieval + Live Data Integration Architecture"]:::titleClass;

        A[User] --> B(Streamlit UI);
        B -- Query --> C{ToolAugmentedRAG Backend};

        %% Static Retrieval
        C --> D[Static Document Retrieval];
        D -- Retrieve From --> E[(Document Store)];
        D -- Static Context --> F{Data Integrator};

        %% Tool Detection
        C --> G[Tool Detector];
        G --> H["core/llm/gemini_utils.py <br> (Gemini 2.5 Pro)"];
        H -- Tool Detection Result --> G;

        %% Tool Execution
        G -- Tool Needed --> I{Tool Router};
        I -- Stock Data --> J[Stock Price API];
        I -- Weather Data --> K[Weather API];
        I -- Web Search --> L[Search API];

        J -- Live Data --> F;
        K -- Live Data --> F;
        L -- Live Data --> F;

        %% Answer Generation
        F -- Combined Context --> M[Answer Generator];
        M --> H;
        H -- Generated Answer --> C;
        C -- Final Answer --> B;

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
- **Retrieval**: rank_bm25, scikit-learn (cosine similarity)
- **Visualization**: Plotly
- **Storage**: SQLite, CSV
- **Domain-specific**: yfinance, NetworkX, PyPDF2

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
 