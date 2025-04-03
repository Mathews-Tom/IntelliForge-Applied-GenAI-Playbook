# IntelliForge: Applied GenAI Playbook

A collection of demo applications showcasing applied Generative AI use cases powered by Google's Gemini 2.5 Pro.

## 🚀 Overview

IntelliForge is a comprehensive playbook of practical Generative AI applications, designed to demonstrate the capabilities of Google's Gemini 2.5 Pro model across various domains. This repository contains four specialized applications, each focusing on a different aspect of AI-powered data interaction and analysis.

## 📊 Applications

### [InsightAgent: Data Analysis](apps/insight_agent/)

A data analysis tool that allows users to upload datasets and ask questions in natural language. The application generates SQL queries and provides comprehensive analyses of the data.

**Key Features:**

- File upload (CSV, Excel)
- Natural language querying
- SQL query generation
- Interactive data tables

### [FiscalAgent: Financial Insights](apps/fiscal_agent/)

A financial analysis tool that provides insights on stocks, financial markets, and economic trends. The application integrates real-time financial data with AI-powered analysis.

**Key Features:**

- Stock price data and visualization
- Company information and analyst recommendations
- Financial news integration
- Web search capabilities

### [ContextQuest: Hybrid Retrieval](apps/context_quest/)

A hybrid retrieval-augmented generation (RAG) system that combines keyword-based and semantic search to provide more accurate and relevant information retrieval.

**Key Features:**

- BM25 keyword-based retrieval
- Embedding-based semantic search
- Hybrid retrieval with adjustable weights
- Retrieval evaluation and relevance scoring

### [GraphQuery: Knowledge Navigator](apps/graph_query/)

A knowledge graph-based system that extracts entities and relationships from documents, builds an interactive graph, and allows natural language querying of the graph.

**Key Features:**

- PDF document processing
- Entity and relationship extraction
- 3D graph visualization
- Relevant subgraph identification

## 🛠️ Core Components

All applications are built on a shared foundation:

- **Gemini 2.5 Pro Integration**: Standardized access to Google's advanced language model
- **Streamlit UI**: Clean, interactive user interfaces
- **Modular Design**: Reusable components and utilities
- **Comprehensive Documentation**: Detailed READMEs for each application

## 🔧 Installation

1. Clone the repository:

   ```bash
   gh repo clone Mathews-Tom/IntelliForge-Applied-GenAI-Playbook
   cd IntelliForge-Applied-GenAI-Playbook
   ```

2. Set up your Google API key:
   - Create a `.env` file in the project root with:

     ```
     GOOGLE_API_KEY=your_google_api_key_here
     ```

3. Install the requirements for the specific application you want to run:

   ```bash
   pip install -r apps/[app_name]/requirements.txt
   ```

4. Run the application:

   ```bash
   cd apps/[app_name]
   streamlit run src/app.py
   ```

## 📝 License

This project is licensed under the CC0 License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
