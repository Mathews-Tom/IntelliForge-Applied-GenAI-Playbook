"""
ToolAugmentedRAG: Retrieval + Live Data Integration
A Streamlit application for RAG with external tool integration using Gemini 2.5 Pro.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import streamlit as st

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from core.llm.gemini_utils import GeminiModelType, generate_content
from core.utils.data_helpers import load_documents_from_csv
from core.utils.file_io import save_uploaded_file
from core.utils.retrieval_utils import embedding_retrieval
from core.utils.ui_helpers import (
    create_footer,
    create_header,
    create_query_input,
    create_sidebar,
    display_documents,
    display_processing_steps,
    file_uploader,
    timed_operation,
)

# Import tools
from tools import StockTool, WeatherTool, WebSearchTool


# Function to load documents from a CSV file
@st.cache_data
def load_documents(file_path):
    """Load documents from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        if "text" not in df.columns:
            st.error("CSV file must contain a 'text' column.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return None


# Function to detect required tools
def detect_tools(query: str) -> Dict[str, bool]:
    """
    Detect which tools are needed for the query.
    
    Args:
        query: The user query
        
    Returns:
        Dictionary mapping tool names to boolean indicating if needed
    """
    prompt = f"""
    Analyze this query to determine if it requires any external tools to provide a complete answer.
    
    Query: "{query}"
    
    Available tools:
    1. Stock Tool: For queries about current stock prices, market data, or company financial information
    2. Weather Tool: For queries about current weather conditions or forecasts
    3. Web Search Tool: For queries requiring recent information, news, or data not likely to be in a static knowledge base
    
    For each tool, determine if it's needed to fully answer the query.
    Return your analysis as a JSON object with the following structure:
    {{
        "stock_tool": true/false,
        "weather_tool": true/false,
        "web_search_tool": true/false,
        "explanation": "Brief explanation of your reasoning"
    }}
    """
    
    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)
        
        if response:
            # Try to parse the response as JSON
            try:
                import json
                result = json.loads(response.strip())
                
                # Ensure all tools are included in the result
                tools = {
                    "stock_tool": result.get("stock_tool", False),
                    "weather_tool": result.get("weather_tool", False),
                    "web_search_tool": result.get("web_search_tool", False),
                    "explanation": result.get("explanation", "No explanation provided")
                }
                
                return tools
            except:
                # If parsing fails, return a default result
                return {
                    "stock_tool": False,
                    "weather_tool": False,
                    "web_search_tool": False,
                    "explanation": "Failed to parse tool detection result"
                }
        else:
            return {
                "stock_tool": False,
                "weather_tool": False,
                "web_search_tool": False,
                "explanation": "Failed to detect required tools"
            }
    except Exception as e:
        st.error(f"Error detecting tools: {e}")
        return {
            "stock_tool": False,
            "weather_tool": False,
            "web_search_tool": False,
            "explanation": f"Error in tool detection: {e}"
        }


# Function to execute tools
def execute_tools(query: str, required_tools: Dict[str, bool]) -> Dict[str, Any]:
    """
    Execute the required tools.
    
    Args:
        query: The user query
        required_tools: Dictionary mapping tool names to boolean indicating if needed
        
    Returns:
        Dictionary mapping tool names to tool results
    """
    tool_results = {}
    
    # Execute Stock Tool if needed
    if required_tools.get("stock_tool", False):
        stock_tool = StockTool()
        tool_results["stock_tool"] = stock_tool.run(query)
    
    # Execute Weather Tool if needed
    if required_tools.get("weather_tool", False):
        weather_tool = WeatherTool()
        tool_results["weather_tool"] = weather_tool.run(query)
    
    # Execute Web Search Tool if needed
    if required_tools.get("web_search_tool", False):
        web_search_tool = WebSearchTool()
        tool_results["web_search_tool"] = web_search_tool.run(query)
    
    return tool_results


# Function to generate an answer
def generate_answer(
    query: str,
    static_context: str,
    tool_results: Dict[str, Any]
) -> str:
    """
    Generate an answer based on static context and tool results.
    
    Args:
        query: The user query
        static_context: Retrieved context from static documents
        tool_results: Results from executed tools
        
    Returns:
        Generated answer
    """
    # Prepare tool results text
    tool_results_text = ""
    
    if "stock_tool" in tool_results:
        result = tool_results["stock_tool"]
        tool_results_text += "Stock Tool Results:\n"
        
        if result.get("success", False) and result.get("data"):
            for stock_data in result["data"]:
                if "error" in stock_data:
                    tool_results_text += f"- {stock_data['ticker']}: {stock_data['error']}\n"
                else:
                    tool_results_text += f"- {stock_data['name']} ({stock_data['ticker']}): {stock_data['price']} {stock_data['currency']} "
                    change = stock_data['change']
                    if change >= 0:
                        tool_results_text += f"(+{change})"
                    else:
                        tool_results_text += f"({change})"
                    tool_results_text += f", {stock_data['change_percent']}%\n"
        else:
            tool_results_text += f"- {result.get('message', 'No data available')}\n"
        
        tool_results_text += "\n"
    
    if "weather_tool" in tool_results:
        result = tool_results["weather_tool"]
        tool_results_text += "Weather Tool Results:\n"
        
        if result.get("success", False) and result.get("data"):
            for weather_data in result["data"]:
                if "error" in weather_data:
                    tool_results_text += f"- {weather_data['location']}: {weather_data['error']}\n"
                else:
                    temp = weather_data['temperature']['current']
                    unit = weather_data['temperature']['unit']
                    conditions = weather_data['conditions']
                    tool_results_text += f"- {weather_data['location']}: {temp}{unit}, {conditions}, "
                    tool_results_text += f"Humidity: {weather_data['humidity']}%, "
                    tool_results_text += f"Wind: {weather_data['wind']['speed']} {weather_data['wind']['unit']} {weather_data['wind']['direction']}\n"
        else:
            tool_results_text += f"- {result.get('message', 'No data available')}\n"
        
        tool_results_text += "\n"
    
    if "web_search_tool" in tool_results:
        result = tool_results["web_search_tool"]
        tool_results_text += "Web Search Tool Results:\n"
        
        if result.get("success", False) and result.get("data"):
            for i, search_result in enumerate(result["data"]):
                tool_results_text += f"- {search_result['title']}\n"
                tool_results_text += f"  Source: {search_result['source']}\n"
                tool_results_text += f"  Snippet: {search_result['snippet']}\n\n"
        else:
            tool_results_text += f"- {result.get('message', 'No search results available')}\n"
        
        tool_results_text += "\n"
    
    # Create prompt
    prompt = f"""
    Answer the following question based on the provided static context and tool results.
    
    Question: {query}
    
    Static Context (from knowledge base):
    {static_context}
    
    Tool Results (from external tools):
    {tool_results_text}
    
    Your task is to:
    1. Synthesize information from both the static context and tool results
    2. Clearly indicate when you're using information from tools vs. static knowledge
    3. Provide a comprehensive answer that addresses all aspects of the query
    4. If there are conflicts between static context and tool results, prioritize the tool results as they are more current
    5. If the tools failed to provide relevant information, rely on the static context
    6. Be transparent about the sources of information in your answer
    
    Answer:
    """
    
    try:
        response = generate_content(GeminiModelType.GEMINI_2_5_PRO, prompt)
        return response if response else "Failed to generate an answer."
    except Exception as e:
        return f"Error generating answer: {e}"


# Main Streamlit app
def main():
    # Create header
    create_header(
        "ToolAugmentedRAG: Retrieval + Live Data Integration",
        "An advanced RAG system that combines static document retrieval with dynamic data from external tools, powered by Gemini 2.5 Pro.",
        icon="🛠️"
    )
    
    # Create sidebar
    create_sidebar(
        "ToolAugmentedRAG Settings",
        "Configure how the tool-augmented retrieval system works."
    )
    
    # Sidebar settings
    with st.sidebar:
        st.header("Retrieval Settings")
        
        top_k = st.slider(
            "Number of documents to retrieve:", 
            min_value=1, 
            max_value=10, 
            value=5
        )
        
        # Tool override settings
        st.header("Tool Settings")
        override_tools = st.checkbox("Override automatic tool detection", value=False)
        
        if override_tools:
            use_stock_tool = st.checkbox("Use Stock Tool", value=False)
            use_weather_tool = st.checkbox("Use Weather Tool", value=False)
            use_web_search_tool = st.checkbox("Use Web Search Tool", value=False)
        
        show_process = st.checkbox("Show detailed process", value=True)
    
    # Main content
    if "GOOGLE_API_KEY" in os.environ:
        # File upload
        st.subheader("1. Upload Static Documents")
        
        uploaded_file = file_uploader(
            "Upload a CSV file with documents (must contain a 'text' column)",
            types=["csv"],
            key="document_upload"
        )
        
        use_sample_data = st.checkbox("Use sample data instead", value=True)
        
        # Load documents
        documents_df = None
        
        if uploaded_file:
            # Save the uploaded file
            file_path = save_uploaded_file(uploaded_file)
            documents_df = load_documents(file_path)
            use_sample_data = False
        elif use_sample_data:
            sample_path = Path(__file__).parent.parent / "data" / "static_docs.csv"
            documents_df = load_documents(sample_path)
        
        if documents_df is not None:
            # Extract documents
            documents = documents_df["text"].tolist()
            
            # Display the number of documents
            st.info(f"Loaded {len(documents)} static documents.")
            
            # Allow users to view the documents
            with st.expander("View Static Documents"):
                for i, doc in enumerate(documents):
                    title = documents_df["title"][i] if "title" in documents_df.columns else f"Document {i+1}"
                    category = documents_df["category"][i] if "category" in documents_df.columns else "Unknown"
                    st.markdown(f"**{title}** (Category: {category})")
                    st.markdown(doc)
                    st.markdown("---")
            
            # Query input
            st.subheader("2. Ask a Question")
            query, search_clicked = create_query_input(
                "Enter your question:",
                button_text="Search with Tools"
            )
            
            # Process query
            if search_clicked and query:
                # Initialize processing steps
                steps = []
                
                # Step 1: Static document retrieval
                with st.spinner("Retrieving static documents..."):
                    start_time = time.time()
                    
                    # Retrieve documents
                    retrieved_docs = embedding_retrieval(query, documents, top_k=top_k)
                    
                    retrieval_time = time.time() - start_time
                    
                    # Add to steps
                    steps.append({
                        "title": "Static Document Retrieval",
                        "content": f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f} seconds.",
                        "time": retrieval_time
                    })
                
                # Step 2: Tool detection
                with st.spinner("Detecting required tools..."):
                    start_time = time.time()
                    
                    # Detect tools
                    if override_tools:
                        required_tools = {
                            "stock_tool": use_stock_tool,
                            "weather_tool": use_weather_tool,
                            "web_search_tool": use_web_search_tool,
                            "explanation": "Tools manually selected by user"
                        }
                    else:
                        required_tools = detect_tools(query)
                    
                    detection_time = time.time() - start_time
                    
                    # Add to steps
                    steps.append({
                        "title": "Tool Detection",
                        "content": {
                            "Stock Tool Required": "Yes" if required_tools.get("stock_tool", False) else "No",
                            "Weather Tool Required": "Yes" if required_tools.get("weather_tool", False) else "No",
                            "Web Search Tool Required": "Yes" if required_tools.get("web_search_tool", False) else "No",
                            "Explanation": required_tools.get("explanation", "No explanation provided")
                        },
                        "time": detection_time
                    })
                
                # Step 3: Tool execution
                tool_results = {}
                if any(required_tools.get(tool, False) for tool in ["stock_tool", "weather_tool", "web_search_tool"]):
                    with st.spinner("Executing tools..."):
                        start_time = time.time()
                        
                        # Execute tools
                        tool_results = execute_tools(query, required_tools)
                        
                        execution_time = time.time() - start_time
                        
                        # Add to steps
                        tool_content = {}
                        for tool_name, result in tool_results.items():
                            if result.get("success", False):
                                tool_content[tool_name] = f"Success: {result.get('message', 'No message')}"
                            else:
                                tool_content[tool_name] = f"Failed: {result.get('message', 'No message')}"
                        
                        steps.append({
                            "title": "Tool Execution",
                            "content": tool_content if tool_content else "No tools were executed.",
                            "time": execution_time
                        })
                
                # Step 4: Generate answer
                with st.spinner("Generating answer..."):
                    start_time = time.time()
                    
                    # Prepare static context
                    static_context = "\n\n".join([doc for doc, _ in retrieved_docs])
                    
                    # Generate answer
                    answer = generate_answer(query, static_context, tool_results)
                    
                    generation_time = time.time() - start_time
                    
                    # Add to steps
                    steps.append({
                        "title": "Answer Generation",
                        "content": "Generated answer combining static context and tool results.",
                        "time": generation_time
                    })
                
                # Display results
                st.subheader("Answer")
                st.markdown(answer)
                
                # Display tool results if any tools were used
                if tool_results:
                    st.subheader("Tool Results")
                    
                    # Display stock tool results
                    if "stock_tool" in tool_results:
                        with st.expander("Stock Tool Results", expanded=True):
                            result = tool_results["stock_tool"]
                            
                            if result.get("success", False) and result.get("data"):
                                for stock_data in result["data"]:
                                    if "error" in stock_data:
                                        st.error(f"{stock_data['ticker']}: {stock_data['error']}")
                                    else:
                                        col1, col2, col3 = st.columns([2, 1, 1])
                                        with col1:
                                            st.markdown(f"**{stock_data['name']} ({stock_data['ticker']})**")
                                        with col2:
                                            st.markdown(f"**{stock_data['price']} {stock_data['currency']}**")
                                        with col3:
                                            change = stock_data['change_percent']
                                            if change >= 0:
                                                st.markdown(f"<span style='color:green'>+{change}%</span>", unsafe_allow_html=True)
                                            else:
                                                st.markdown(f"<span style='color:red'>{change}%</span>", unsafe_allow_html=True)
                                        
                                        st.markdown(f"Volume: {stock_data.get('volume', 'N/A')}")
                                        st.markdown(f"Market Cap: {stock_data.get('market_cap', 'N/A')}")
                                        st.markdown(f"Last Updated: {stock_data.get('timestamp', 'N/A')}")
                                        st.markdown("---")
                            else:
                                st.warning(result.get("message", "No stock data available"))
                    
                    # Display weather tool results
                    if "weather_tool" in tool_results:
                        with st.expander("Weather Tool Results", expanded=True):
                            result = tool_results["weather_tool"]
                            
                            if result.get("success", False) and result.get("data"):
                                for weather_data in result["data"]:
                                    if "error" in weather_data:
                                        st.error(f"{weather_data['location']}: {weather_data['error']}")
                                    else:
                                        st.markdown(f"**{weather_data['location']}**")
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            temp = weather_data['temperature']['current']
                                            unit = weather_data['temperature']['unit']
                                            st.markdown(f"**Temperature: {temp}{unit}**")
                                            st.markdown(f"Feels like: {weather_data['temperature']['feels_like']}{unit}")
                                            st.markdown(f"Min/Max: {weather_data['temperature']['min']}{unit} / {weather_data['temperature']['max']}{unit}")
                                        with col2:
                                            st.markdown(f"**Conditions: {weather_data['conditions']}**")
                                            st.markdown(f"Humidity: {weather_data['humidity']}%")
                                            st.markdown(f"Wind: {weather_data['wind']['speed']} {weather_data['wind']['unit']} {weather_data['wind']['direction']}")
                                        
                                        st.markdown(f"Pressure: {weather_data['pressure']} hPa")
                                        st.markdown(f"Visibility: {weather_data['visibility']} km")
                                        st.markdown(f"Last Updated: {weather_data['timestamp']}")
                                        st.markdown("---")
                            else:
                                st.warning(result.get("message", "No weather data available"))
                    
                    # Display web search tool results
                    if "web_search_tool" in tool_results:
                        with st.expander("Web Search Tool Results", expanded=True):
                            result = tool_results["web_search_tool"]
                            
                            if result.get("success", False) and result.get("data"):
                                for search_result in result["data"]:
                                    st.markdown(f"**{search_result['title']}**")
                                    st.markdown(f"Source: {search_result['source']}")
                                    st.markdown(f"Date: {search_result.get('date', 'Unknown')}")
                                    st.markdown(f"{search_result['snippet']}")
                                    if search_result.get('link'):
                                        st.markdown(f"[Read more]({search_result['link']})")
                                    st.markdown("---")
                            else:
                                st.warning(result.get("message", "No search results available"))
                
                # Display retrieved documents
                st.subheader("Retrieved Static Documents")
                
                # Convert to format expected by display_documents
                docs_with_scores = []
                for doc, score in retrieved_docs:
                    docs_with_scores.append({"text": doc, "score": score})
                
                scores = [score for _, score in retrieved_docs]
                display_documents(docs_with_scores, scores=scores)
                
                # Display processing steps if enabled
                if show_process:
                    display_processing_steps(steps)
            
            elif search_clicked:
                st.warning("Please enter a question.")
        else:
            st.warning("Please upload a CSV file with documents or use the sample data.")
    
    # Add footer
    create_footer()


if __name__ == "__main__":
    main()
