# IntelliForge: Applied GenAI Playbook - Architecture

This document provides a high-level overview of the architecture and design principles used in the IntelliForge: Applied GenAI Playbook repository.

## System Architecture

The repository is organized as a collection of independent applications that share a common core for Gemini API integration. This modular approach allows each application to focus on its specific domain while leveraging shared utilities for LLM interaction.

```bash
IntelliForge-Applied-GenAI-Playbook/
│
├── apps/                   # Directory for individual applications
│   │
│   ├── insight_agent/      # InsightAgent: Data Analysis
│   │   ├── src/            # Source code
│   │   ├── data/           # Sample data
│   │   ├── notebooks/      # Jupyter notebooks for experimentation/demos
│   │   └── README.md       # App-specific README
│   │
│   ├── fiscal_agent/       # FiscalAgent: Financial Insights
│   │   ├── src/
│   │   ├── data/
│   │   ├── notebooks/
│   │   └── README.md
│   │
│   ├── context_quest/      # ContextQuest: Hybrid Retrieval
│   │   ├── src/
│   │   ├── data/
│   │   ├── notebooks/
│   │   └── README.md
│   │
│   └── graph_query/        # GraphQuery: Knowledge Navigator
│       ├── src/
│       ├── data/
│       ├── notebooks/
│       └── README.md
│
├── core/                   # Shared core components
│   └── llm/                # LLM integration logic
│       └── gemini_utils.py # Gemini API utilities
│
├── docs/                   # Overall documentation
│   └── architecture.md     # This file
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

### 1. InsightAgent: Data Analysis

- **Data Processing**: Functions to handle CSV and Excel file uploads
- **Query Processing**: Natural language to SQL conversion using Gemini
- **UI Components**: Streamlit interface with data visualization

### 2. FiscalAgent: Financial Insights

- **Data Sources**: Integration with financial APIs (yfinance)
- **Multi-agent System**: Web search and financial data agents
- **Conversation Storage**: SQLite database for conversation history
- **UI Components**: Tabbed interface for different functionalities

### 3. ContextQuest: Hybrid Retrieval

- **Retrieval Methods**: BM25 and embedding-based retrieval
- **Hybrid Ranking**: Weighted combination of retrieval scores
- **Evaluation**: Relevance assessment of retrieved documents
- **UI Components**: Interactive controls for retrieval parameters

### 4. GraphQuery: Knowledge Navigator

- **Document Processing**: PDF text extraction and chunking
- **Entity Extraction**: Identification of entities and relationships
- **Graph Construction**: Building and querying knowledge graphs
- **Visualization**: 3D interactive graph visualization

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
