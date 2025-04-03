# FiscalAgent: Financial Insights

## Overview

**FiscalAgent: Financial Insights** is a Streamlit-based application that leverages Google's Gemini 2.5 Pro model to provide comprehensive financial analysis and insights. The application integrates real-time financial data, web search capabilities, and advanced AI analysis to help users make informed financial decisions.

## Key Features

- **Stock Analysis**: Real-time stock price data, charts, and AI-generated analysis
- **Company Information**: Detailed company profiles including sector, industry, and business summaries
- **Analyst Recommendations**: Latest analyst ratings and price targets
- **Financial News**: Recent news articles related to specific stocks or financial topics
- **Web Search Integration**: Retrieves relevant information from the web to enhance responses
- **Conversation History**: Stores user interactions in a SQLite database for context awareness
- **Interactive UI**: User-friendly Streamlit interface with tabs for different functionalities

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
   pip install -r apps/fiscal_agent/requirements.txt
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
   cd apps/fiscal_agent
   streamlit run src/app.py
   ```

2. **Chat Interface**:
   - Enter financial questions in the text area
   - The agent will analyze your query, gather relevant data, and provide a comprehensive response
   - For stock-specific queries, the agent automatically retrieves and incorporates stock data

3. **Stock Analysis**:
   - Enter a stock ticker symbol (e.g., AAPL, MSFT)
   - Select a time period for analysis
   - View stock price charts, company information, recent news, and AI-generated analysis

## Example Queries

- "What is the current stock price of Tesla and how has it performed over the last month?"
- "Compare the financial performance of Apple and Microsoft"
- "What are the latest analyst recommendations for Amazon?"
- "Explain the impact of rising interest rates on the banking sector"
- "What are the best dividend stocks in the technology sector?"

## Technical Details

### Architecture

- **Streamlit**: Provides the web interface
- **yfinance**: Retrieves stock data, company information, and news
- **SQLite**: Stores conversation history
- **BeautifulSoup**: Parses web search results
- **Gemini 2.5 Pro**: Generates financial analysis and insights
- **Core Utilities**: Leverages shared Gemini integration from the core module

## Future Enhancements

- Add portfolio tracking and management capabilities
- Implement technical analysis indicators and visualizations
- Integrate with additional financial data sources
- Add cryptocurrency market data and analysis
- Implement sentiment analysis for financial news
- Develop personalized investment recommendations

## License

This project is licensed under the CC0 License - see the LICENSE file in the repository root for details.

## Contact

For questions or contributions, please open an issue on the GitHub repository.
