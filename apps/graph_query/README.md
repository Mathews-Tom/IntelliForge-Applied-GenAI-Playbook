# GraphQuery: Knowledge Navigator

## Overview

**GraphQuery: Knowledge Navigator** is a Streamlit-based application that leverages Google's Gemini 2.5 Pro model to create and query knowledge graphs from documents. This tool extracts entities and relationships from PDF documents, builds an interactive knowledge graph, and allows users to query the graph using natural language.

## Key Features

- **PDF Document Processing**: Extract text from PDF documents and process it for entity and relationship extraction
- **Knowledge Graph Creation**: Automatically identify entities and relationships to build a structured knowledge graph
- **3D Graph Visualization**: Interactive 3D visualization of the knowledge graph using Plotly
- **Natural Language Querying**: Ask questions about the knowledge graph in plain English
- **Relevant Subgraph Extraction**: Identify and visualize the most relevant parts of the graph for a specific query
- **Gemini 2.5 Pro Integration**: Powered by Google's advanced language model for sophisticated entity extraction and querying

## Installation

### Prerequisites

- Python 3.8 or higher
- Google API key for Gemini 2.5 Pro

### Steps

1. Clone the repository:

   ```bash
   gh repo clone Mathews-Tom/IntelliForge-Applied-GenAI-Playbook
   cd IntelliForge-Applied-GenAI-Playbook
   ```

2. Install the required Python libraries:

   ```bash
   pip install -r apps/graph_query/requirements.txt
   ```

3. Set up your Google API key:
   - Create a `.env` file in the project root with:

     ```bash
     GOOGLE_API_KEY=your_google_api_key_here
     ```

   - Or enter it directly in the Streamlit interface when prompted

## Usage

1. **Run the Application**:

   ```bash
   cd apps/graph_query
   streamlit run src/app.py
   ```

2. **Build Knowledge Graph**:
   - Upload a PDF document using the file uploader
   - Click "Extract Entities and Build Knowledge Graph" to process the document
   - View the extracted relationships and the 3D visualization of the knowledge graph

3. **Query Knowledge Graph**:
   - Enter a natural language query about the content of the document
   - View the answer generated based on the knowledge graph
   - Explore the relevant subgraph visualization showing the relationships most pertinent to your query

## Example Queries

- "What are the main concepts discussed in the document?"
- "What is the relationship between [Entity A] and [Entity B]?"
- "What factors influence [Entity C]?"
- "Summarize the key relationships involving [Entity D]"

## Technical Details

### Architecture

- **Streamlit**: Provides the web interface
- **PyPDF2**: Extracts text from PDF documents
- **NetworkX**: Represents and manipulates the knowledge graph
- **Plotly**: Creates interactive 3D visualizations of the graph
- **Gemini 2.5 Pro**: Extracts entities and relationships, and answers queries
- **Core Utilities**: Leverages shared Gemini integration from the core module

### Knowledge Graph Construction

1. **Text Extraction**: PDF documents are processed to extract text
2. **Text Chunking**: Long documents are split into manageable chunks
3. **Entity and Relationship Extraction**: Gemini 2.5 Pro identifies entities and their relationships
4. **Graph Building**: Extracted relationships are used to construct a directed graph
5. **Visualization**: The graph is rendered as an interactive 3D visualization

## Future Enhancements

- Add support for additional document formats (DOCX, HTML, etc.)
- Implement more sophisticated entity resolution and deduplication
- Add the ability to merge multiple knowledge graphs
- Integrate with external knowledge bases and ontologies
- Implement graph-based reasoning capabilities
- Add export functionality for the knowledge graph

## License

This project is licensed under the CC0 License - see the LICENSE file in the repository root for details.

## Contact

For questions or contributions, please open an issue on the GitHub repository.
