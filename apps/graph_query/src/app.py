"""
GraphQuery: Knowledge Navigator
A Streamlit application for knowledge graph-based retrieval using Gemini 2.5 Pro.
"""

import os
import re
import sys
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PyPDF2 import PdfReader

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from core.llm.gemini_utils import GeminiModelType, generate_content

# Initialize session state for graph data
if "graph" not in st.session_state:
    st.session_state.graph = nx.DiGraph()

if "documents" not in st.session_state:
    st.session_state.documents = []


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file: st.UploadedFile) -> str | None:
    """
    Extract text from a PDF file.

    Args:
        pdf_file: The uploaded PDF file.

    Returns:
        Extracted text as a string or None if extraction fails.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_path = temp_file.name

        reader = PdfReader(temp_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        # Clean up the temporary file
        os.unlink(temp_path)

        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None


# Function to chunk text
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to split.
        chunk_size: Size of each chunk.
        overlap: Overlap between chunks.

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# Function to extract entities and relationships
def extract_entities_and_relationships(text: str) -> list[tuple[str, str, str]]:
    """
    Extract entities and relationships from text using Gemini.

    Args:
        text: Text to analyze.

    Returns:
        List of (entity1, relationship, entity2) tuples.
    """
    prompt = f"""
    Extract key entities and their relationships from this text.
    Format each relationship exactly as: (entity1)-[relationship]->(entity2)
    Return one relationship per line.
    Only include clear, explicit relationships from the text.

    Text: {text}
    """

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)

        if response:
            # Parse the response to extract relationships
            relationships = []
            pattern = r"\(([^)]+)\)-\[([^\]]+)\]->\(([^)]+)\)"

            for line in response.split("\n"):
                line = line.strip()
                matches = re.findall(pattern, line)

                for match in matches:
                    if len(match) == 3:
                        entity1, relationship, entity2 = match
                        entity1 = entity1.strip()
                        relationship = relationship.strip()
                        entity2 = entity2.strip()

                        if entity1 and relationship and entity2:
                            relationships.append((entity1, relationship, entity2))

            return relationships
        else:
            st.error("Failed to extract entities and relationships.")
            return []
    except Exception as e:
        st.error(f"Error extracting entities and relationships: {e}")
        return []


# Function to build knowledge graph
def build_knowledge_graph(relationships: list[tuple[str, str, str]]) -> nx.DiGraph:
    """
    Build a NetworkX graph from relationships.

    Args:
        relationships: List of (entity1, relationship, entity2) tuples.

    Returns:
        NetworkX DiGraph.
    """
    G = nx.DiGraph()

    for entity1, relationship, entity2 in relationships:
        G.add_node(entity1)
        G.add_node(entity2)
        G.add_edge(entity1, entity2, label=relationship)

    return G


# Function to create 3D graph visualization
def create_3d_graph_visualization(G: nx.DiGraph) -> go.Figure:
    """
    Create a 3D graph visualization using Plotly.

    Args:
        G: NetworkX graph.

    Returns:
        Plotly figure.
    """
    # Create positions for nodes using a spring layout
    pos = nx.spring_layout(G, dim=3, seed=42)

    # Create edges
    edge_x = []
    edge_y = []
    edge_z = []
    edge_text = []

    for edge in G.edges(data=True):
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        edge_text.append(edge[2]["label"])

    # Create edge trace
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        line=dict(width=2, color="rgba(100, 100, 100, 0.8)"),
        hoverinfo="none",
        mode="lines",
    )

    # Create nodes
    node_x = []
    node_y = []
    node_z = []
    node_text = []

    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(node)

    # Create node trace
    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers",
        marker=dict(
            size=8,
            color="rgba(0, 200, 100, 0.8)",
            line=dict(width=1, color="rgba(50, 50, 50, 0.8)"),
        ),
        text=node_text,
        hoverinfo="text",
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])

    # Update layout
    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig


# Function to query the knowledge graph
def query_knowledge_graph(G: nx.DiGraph, query: str) -> tuple[str, nx.DiGraph]:
    """
    Query the knowledge graph using Gemini.

    Args:
        G: NetworkX graph.
        query: User query.

    Returns:
        Tuple of (answer, relevant_subgraph).
    """
    # Convert graph to a text representation
    graph_text = ""
    for edge in G.edges(data=True):
        source, target, data = edge
        relationship = data.get("label", "related_to")
        graph_text += f"({source})-[{relationship}]->({target})\n"

    # Create prompt for Gemini
    prompt = f"""
    I have a knowledge graph with the following relationships:

    {graph_text}

    Based on this knowledge graph, answer the following question:
    {query}

    Also, identify the specific entities and relationships from the knowledge graph that are most relevant to this question.
    Format your answer as follows:

    Answer: [Your detailed answer here]

    Relevant Entities and Relationships:
    1. [entity1] - [relationship] -> [entity2]
    2. [entity3] - [relationship] -> [entity4]
    ...
    """

    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)

        if response:
            # Split the response into answer and relevant relationships
            parts = response.split("Relevant Entities and Relationships:")

            answer = parts[0].replace("Answer:", "").strip()

            # Extract relevant relationships
            relevant_relationships = []
            if len(parts) > 1:
                rel_text = parts[1].strip()
                pattern = r"([^-\n]+)\s*-\s*\[([^\]]+)\]\s*->\s*([^\n]+)"
                matches = re.findall(pattern, rel_text)

                for match in matches:
                    if len(match) == 3:
                        entity1, relationship, entity2 = match
                        entity1 = entity1.strip()
                        relationship = relationship.strip()
                        entity2 = entity2.strip()

                        if entity1 and relationship and entity2:
                            relevant_relationships.append(
                                (entity1, relationship, entity2)
                            )

            # Create a subgraph with relevant relationships
            subgraph = nx.DiGraph()
            for entity1, relationship, entity2 in relevant_relationships:
                subgraph.add_node(entity1)
                subgraph.add_node(entity2)
                subgraph.add_edge(entity1, entity2, label=relationship)

            return answer, subgraph
        else:
            return "Failed to generate an answer.", nx.DiGraph()
    except Exception as e:
        return f"Error querying knowledge graph: {e}", nx.DiGraph()


# Streamlit app
st.title("🌐 GraphQuery: Knowledge Navigator")
st.markdown("""
This application demonstrates knowledge graph-based retrieval and querying using Gemini 2.5 Pro.
Upload a PDF document to extract entities and relationships, build a knowledge graph, and query it.
""")

# Sidebar for API keys and settings
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Google API key:", type="password")

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("API key saved!")
    else:
        st.warning("Please enter your Google API key to proceed.")

    st.markdown("---")
    st.header("Graph Settings")

    # Add graph visualization settings here if needed

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    GraphQuery uses knowledge graphs to represent and query complex relationships
    extracted from documents, providing more context-aware and structured information retrieval.
    """)

# Main content
if "GOOGLE_API_KEY" in os.environ:
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Build Knowledge Graph", "Query Knowledge Graph"])

    with tab1:
        st.header("Build Knowledge Graph")

        # File upload
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

        if uploaded_file:
            # Extract text from PDF
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)

                if text:
                    st.success(
                        f"Successfully extracted {len(text)} characters from the PDF."
                    )

                    # Show text preview
                    with st.expander("Text Preview"):
                        st.text(text[:1000] + "..." if len(text) > 1000 else text)

                    # Chunk the text
                    chunks = chunk_text(text)
                    st.info(f"Split text into {len(chunks)} chunks.")

                    # Store chunks in session state
                    st.session_state.documents = chunks

                    # Process button
                    if st.button("Extract Entities and Build Knowledge Graph"):
                        # Clear existing graph
                        st.session_state.graph = nx.DiGraph()

                        # Process each chunk
                        progress_bar = st.progress(0)
                        all_relationships = []

                        for i, chunk in enumerate(chunks):
                            with st.spinner(
                                f"Processing chunk {i + 1}/{len(chunks)}..."
                            ):
                                relationships = extract_entities_and_relationships(
                                    chunk
                                )
                                all_relationships.extend(relationships)

                            # Update progress
                            progress_bar.progress((i + 1) / len(chunks))

                        # Build knowledge graph
                        st.session_state.graph = build_knowledge_graph(
                            all_relationships
                        )

                        # Display statistics
                        st.success(
                            f"Knowledge graph built with {st.session_state.graph.number_of_nodes()} entities and {st.session_state.graph.number_of_edges()} relationships."
                        )

                        # Display relationships
                        with st.expander("View Extracted Relationships"):
                            relationships_df = pd.DataFrame(
                                all_relationships,
                                columns=["Entity 1", "Relationship", "Entity 2"],
                            )
                            st.dataframe(relationships_df)

                        # Visualize graph
                        if st.session_state.graph.number_of_nodes() > 0:
                            st.subheader("Knowledge Graph Visualization")
                            fig = create_3d_graph_visualization(st.session_state.graph)
                            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Query Knowledge Graph")

        # Check if graph exists
        if st.session_state.graph.number_of_nodes() > 0:
            # Display graph statistics
            st.info(
                f"Knowledge graph contains {st.session_state.graph.number_of_nodes()} entities and {st.session_state.graph.number_of_edges()} relationships."
            )

            # Query input
            query = st.text_area("Enter your query:", height=100)

            if st.button("Submit Query"):
                if query:
                    with st.spinner("Querying knowledge graph..."):
                        answer, relevant_subgraph = query_knowledge_graph(
                            st.session_state.graph, query
                        )

                        # Display answer
                        st.subheader("Answer")
                        st.markdown(answer)

                        # Display relevant subgraph
                        if relevant_subgraph.number_of_nodes() > 0:
                            st.subheader("Relevant Relationships")

                            # Display as table
                            relationships = []
                            for edge in relevant_subgraph.edges(data=True):
                                source, target, data = edge
                                relationship = data.get("label", "related_to")
                                relationships.append((source, relationship, target))

                            relationships_df = pd.DataFrame(
                                relationships,
                                columns=["Entity 1", "Relationship", "Entity 2"],
                            )
                            st.dataframe(relationships_df)

                            # Visualize subgraph
                            st.subheader("Relevant Subgraph Visualization")
                            fig = create_3d_graph_visualization(relevant_subgraph)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please enter a query.")
        else:
            st.warning(
                "No knowledge graph available. Please build a knowledge graph first."
            )

# Add footer
st.markdown("---")
st.markdown("GraphQuery: Knowledge Navigator | Powered by Gemini 2.5 Pro")
st.markdown("GraphQuery: Knowledge Navigator | Powered by Gemini 2.5 Pro")
