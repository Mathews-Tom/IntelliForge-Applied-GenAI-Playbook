"""
InsightAgent: Data Analysis
A Streamlit application for data analysis using Gemini 2.5 Pro.
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from core.llm.gemini_utils import GeminiModelType, generate_content


# Function to preprocess and save uploaded file
def preprocess_and_save(
    uploaded_file: st.UploadedFile,
) -> tuple[str | None, list[str] | None, pd.DataFrame | None]:
    """
    Preprocess and save the uploaded file.

    Args:
        uploaded_file: The uploaded file from Streamlit.

    Returns:
        tuple: (temp_path, columns, dataframe) or (None, None, None) if processing fails
    """
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp_path = temp_file.name

        # Check file type and process accordingly
        if uploaded_file.name.endswith(".csv"):
            # For CSV files
            df = pd.read_csv(uploaded_file)
            df.to_csv(temp_path, index=False)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            # For Excel files
            df = pd.read_excel(uploaded_file)
            df.to_csv(temp_path, index=False)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None

        # Get column names
        columns = df.columns.tolist()

        return temp_path, columns, df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None


# Streamlit app
st.title("📊 InsightAgent: Data Analysis")
st.markdown("""
This application allows you to upload a dataset and ask questions about it.
The AI will generate SQL queries and provide answers based on your data.
""")

# Sidebar for API keys
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Google API key:", type="password")

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("API key saved!")
    else:
        st.warning("Please enter your Google API key to proceed.")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    InsightAgent uses Gemini 2.5 Pro to analyze your data and generate insights.
    Upload a CSV or Excel file to get started.
    """)

# File upload widget
uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file", type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None and "GOOGLE_API_KEY" in os.environ:
    # Preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if temp_path and columns and df is not None:
        # Display the uploaded data as a table
        st.write("Uploaded Data:")
        st.dataframe(df)  # Use st.dataframe for an interactive table

        # Display the columns of the uploaded data
        st.write("Uploaded columns:", columns)

        # Main query input widget
        user_query = st.text_area("Ask a query about the data:")

        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                try:
                    # Show loading spinner while processing
                    with st.spinner("Processing your query..."):
                        # Prepare the prompt for Gemini
                        prompt = f"""
                        You are an expert data analyst. I have a dataset with the following columns:
                        {columns}

                        Here's a sample of the data:
                        {df.head(5).to_string()}

                        My question is: {user_query}

                        First, generate a SQL query that would answer this question.
                        Then, analyze the data and provide a comprehensive answer.
                        Format your response as follows:

                        SQL Query:
                        ```sql
                        [Your SQL query here]
                        ```

                        Analysis:
                        [Your detailed analysis here]
                        """

                        # Generate response using Gemini
                        response = generate_content(
                            GeminiModelType.GEMINI_2_5_PRO, prompt
                        )

                        if response:
                            st.markdown(response)
                        else:
                            st.error("Failed to generate a response. Please try again.")

                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.error(
                        "Please try rephrasing your query or check if the data format is correct."
                    )

# Add footer
st.markdown("---")
st.markdown("InsightAgent: Data Analysis | Powered by Gemini 2.5 Pro")
st.markdown("---")
st.markdown("InsightAgent: Data Analysis | Powered by Gemini 2.5 Pro")
