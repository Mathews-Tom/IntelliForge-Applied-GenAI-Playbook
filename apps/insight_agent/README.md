# InsightAgent: Data Analysis

## Overview

**InsightAgent: Data Analysis** is a Streamlit-based application that leverages the power of Google's Gemini 2.5 Pro model to analyze datasets and generate insights. This tool allows users to upload datasets in CSV or Excel format, ask questions in natural language, and receive SQL queries and comprehensive analyses.

## Key Features

- **File Upload**: Support for CSV and Excel file uploads
- **Natural Language Querying**: Ask questions about your data in plain English
- **SQL Query Generation**: Automatically generates SQL queries to address user questions
- **Gemini 2.5 Pro Integration**: Powered by Google's advanced language model for sophisticated analysis
- **Interactive Data Table**: Displays uploaded datasets in an interactive table
- **User-friendly Interface**: Simple, intuitive Streamlit UI with sidebar configuration

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
   pip install -r apps/insight_agent/requirements.txt
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
   cd apps/insight_agent
   streamlit run src/app.py
   ```

2. **Interface**:
   - Upload a CSV or Excel file using the file uploader
   - Enter your query about the uploaded data in plain English
   - Click **Submit Query** to get the SQL query and analysis

## Example Queries

- "What is the average sales amount for each region?"
- "Show me the top 5 customers by total purchase value"
- "Is there a correlation between customer age and purchase frequency?"
- "Identify outliers in the transaction amounts"

## Technical Details

### Architecture

- **Streamlit**: Provides the web interface
- **Pandas**: Handles data processing and manipulation
- **Gemini 2.5 Pro**: Generates SQL queries and analyses based on natural language input
- **Core Utilities**: Leverages shared Gemini integration from the core module

## Future Enhancements

- Add support for additional file formats (JSON, Parquet)
- Implement data visualization capabilities
- Add export functionality for analyses
- Integrate with database connections for direct querying

## License

This project is licensed under the CC0 License - see the LICENSE file in the repository root for details.

## Contact

For questions or contributions, please open an issue on the GitHub repository.
