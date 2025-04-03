"""
FiscalAgent: Financial Insights
A Streamlit application for financial analysis using Gemini 2.5 Pro.
"""

import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

# Add the project root to the Python path to import core modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from core.llm.gemini_utils import GeminiModelType, generate_content


# Initialize SQLite database for conversation history
def init_db() -> None:
    """Initialize SQLite database for conversation history"""
    conn = sqlite3.connect("fiscal_agent.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations
    (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent TEXT,
        query TEXT,
        response TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()


# Save conversation to database
def save_conversation(agent: str, query: str, response: str) -> None:
    """Save conversation to database"""
    conn = sqlite3.connect("fiscal_agent.db")
    c = conn.cursor()
    c.execute(
        """
    INSERT INTO conversations (agent, query, response)
    VALUES (?, ?, ?)
    """,
        (agent, query, response),
    )
    conn.commit()
    conn.close()


# Get stock data
def get_stock_data(ticker: str, period: str = "1mo") -> pd.DataFrame | None:
    """Get stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None


# Get stock news
def get_stock_news(ticker: str, limit: int = 5) -> list[dict]:
    """Get news for a stock"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news[:limit] if news else []
    except Exception as e:
        st.error(f"Error fetching stock news: {e}")
        return []


# Get company info
def get_company_info(ticker: str) -> dict:
    """Get company information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        st.error(f"Error fetching company info: {e}")
        return {}


# Get analyst recommendations
def get_analyst_recommendations(ticker: str) -> pd.DataFrame | None:
    """Get analyst recommendations"""
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        return recommendations
    except Exception as e:
        st.error(f"Error fetching analyst recommendations: {e}")
        return None


# Web search function
def web_search(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """Simple web search function using DuckDuckGo"""
    try:
        # Format the query for a URL
        search_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        for result in soup.select(".result__body")[:num_results]:
            title_elem = result.select_one(".result__title")
            link_elem = result.select_one(".result__url")
            snippet_elem = result.select_one(".result__snippet")

            title = title_elem.get_text() if title_elem else "No title"
            link = link_elem.get("href") if link_elem else "#"
            snippet = snippet_elem.get_text() if snippet_elem else "No description"

            results.append({"title": title, "link": link, "snippet": snippet})

        return results
    except Exception as e:
        st.error(f"Error during web search: {e}")
        return []


# Initialize the database
init_db()

# Streamlit app
st.title("💼 FiscalAgent: Financial Insights")
st.markdown("""
This application provides financial insights and analysis using Gemini 2.5 Pro.
You can ask questions about stocks, financial markets, and economic trends.
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
    st.markdown("### Tools Available")
    st.markdown("""
    - Stock Price Data
    - Company Information
    - Analyst Recommendations
    - Financial News
    - Web Search
    """)

# Main content
if "GOOGLE_API_KEY" in os.environ:
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Chat", "Stock Analysis"])

    with tab1:
        st.header("Financial Assistant")

        # User input
        user_query = st.text_area("Ask a financial question:", height=100)

        if st.button("Submit", key="submit_query"):
            if user_query:
                with st.spinner("Analyzing your question..."):
                    # Determine if this is a stock-specific query
                    stock_prompt = f"""
                    Analyze this query: "{user_query}"

                    If this query is about a specific stock or company, extract the stock ticker symbol.
                    If no specific stock is mentioned, return "NONE".

                    Return only the ticker symbol or "NONE", nothing else.
                    """

                    ticker_response = generate_content(
                        GeminiModelType.GEMINI_2_5_PRO, stock_prompt
                    )
                    ticker = ticker_response.strip() if ticker_response else "NONE"

                    # Collect context based on the query
                    context = ""

                    # If a specific stock was mentioned, get data
                    if ticker != "NONE":
                        st.info(f"Analyzing data for stock: {ticker}")

                        # Get stock price data
                        stock_data = get_stock_data(ticker)
                        if stock_data is not None and not stock_data.empty:
                            context += f"\nRecent stock price data for {ticker}:\n"
                            context += stock_data.tail().to_string() + "\n\n"

                        # Get company info
                        company_info = get_company_info(ticker)
                        if company_info:
                            context += f"Company information for {ticker}:\n"
                            for key in [
                                "shortName",
                                "longName",
                                "industry",
                                "sector",
                                "website",
                                "longBusinessSummary",
                            ]:
                                if key in company_info:
                                    context += f"{key}: {company_info[key]}\n"
                            context += "\n"

                        # Get analyst recommendations
                        recommendations = get_analyst_recommendations(ticker)
                        if recommendations is not None and not recommendations.empty:
                            context += f"Recent analyst recommendations for {ticker}:\n"
                            context += recommendations.tail().to_string() + "\n\n"

                        # Get news
                        news = get_stock_news(ticker)
                        if news:
                            context += f"Recent news for {ticker}:\n"
                            for i, article in enumerate(news):
                                context += f"{i + 1}. {article['title']} - {article['publisher']}\n"
                            context += "\n"

                    # For all queries, do a web search for additional context
                    search_results = web_search(user_query)
                    if search_results:
                        context += "Web search results:\n"
                        for i, result in enumerate(search_results):
                            context += f"{i + 1}. {result['title']}\n{result['snippet']}\n{result['link']}\n\n"

                    # Generate the final prompt with all context
                    final_prompt = f"""
                    You are a financial expert assistant. Answer the following question using the provided context.
                    If the context doesn't contain enough information, use your knowledge but clearly indicate when you're doing so.

                    Question: {user_query}

                    Context:
                    {context}

                    Provide a comprehensive, well-structured answer. Include relevant data points from the context.
                    If appropriate, suggest follow-up questions the user might want to ask.
                    """

                    # Generate response using Gemini
                    response = generate_content(
                        GeminiModelType.GEMINI_2_5_PRO, final_prompt
                    )

                    if response:
                        st.markdown(response)
                        # Save conversation to database
                        save_conversation("FiscalAgent", user_query, response)
                    else:
                        st.error("Failed to generate a response. Please try again.")
            else:
                st.warning("Please enter a question.")

    with tab2:
        st.header("Stock Analysis")

        # Stock ticker input
        ticker_symbol = st.text_input("Enter stock ticker symbol (e.g., AAPL, MSFT):")

        if ticker_symbol:
            # Time period selection
            period = st.selectbox(
                "Select time period:",
                options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                index=0,
            )

            if st.button("Analyze", key="analyze_stock"):
                with st.spinner(f"Analyzing {ticker_symbol}..."):
                    # Get stock data
                    stock_data = get_stock_data(ticker_symbol, period)

                    if stock_data is not None and not stock_data.empty:
                        # Display stock price chart
                        st.subheader(f"{ticker_symbol} Stock Price")
                        st.line_chart(stock_data["Close"])

                        # Display company info
                        company_info = get_company_info(ticker_symbol)
                        if company_info:
                            st.subheader("Company Information")
                            info_cols = st.columns(2)

                            with info_cols[0]:
                                if "longName" in company_info:
                                    st.write(f"**Name:** {company_info['longName']}")
                                if "sector" in company_info:
                                    st.write(f"**Sector:** {company_info['sector']}")
                                if "industry" in company_info:
                                    st.write(
                                        f"**Industry:** {company_info['industry']}"
                                    )

                            with info_cols[1]:
                                if "marketCap" in company_info:
                                    st.write(
                                        f"**Market Cap:** ${company_info['marketCap']:,}"
                                    )
                                if "trailingPE" in company_info:
                                    st.write(
                                        f"**P/E Ratio:** {company_info['trailingPE']:.2f}"
                                    )
                                if (
                                    "dividendYield" in company_info
                                    and company_info["dividendYield"] is not None
                                ):
                                    st.write(
                                        f"**Dividend Yield:** {company_info['dividendYield'] * 100:.2f}%"
                                    )

                            if "longBusinessSummary" in company_info:
                                st.write("**Business Summary:**")
                                st.write(company_info["longBusinessSummary"])

                        # Display news
                        news = get_stock_news(ticker_symbol)
                        if news:
                            st.subheader("Recent News")
                            for article in news:
                                st.markdown(f"**{article['title']}**")
                                st.markdown(
                                    f"*{article['publisher']}* - {datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M')}"
                                )
                                st.markdown(f"[Read more]({article['link']})")
                                st.markdown("---")

                        # Generate AI analysis
                        analysis_prompt = f"""
                        Analyze the following stock data for {ticker_symbol} over the {period} period:

                        Price data:
                        {stock_data.tail().to_string()}

                        Company information:
                        {company_info.get("longName", ticker_symbol)} - {company_info.get("sector", "N/A")} - {company_info.get("industry", "N/A")}
                        Market Cap: ${company_info.get("marketCap", "N/A"):,}
                        P/E Ratio: {company_info.get("trailingPE", "N/A")}
                        Dividend Yield: {company_info.get("dividendYield", "N/A") * 100 if company_info.get("dividendYield") is not None else "N/A"}%

                        Provide a comprehensive analysis including:
                        1. Price trend analysis
                        2. Key performance indicators
                        3. Potential factors affecting the stock
                        4. Brief outlook

                        Format your response in markdown with clear sections.
                        """

                        st.subheader("AI Analysis")
                        with st.spinner("Generating analysis..."):
                            analysis = generate_content(
                                GeminiModelType.GEMINI_2_5_PRO, analysis_prompt
                            )
                            if analysis:
                                st.markdown(analysis)
                            else:
                                st.error(
                                    "Failed to generate analysis. Please try again."
                                )
                    else:
                        st.error(
                            f"Could not retrieve data for {ticker_symbol}. Please check the ticker symbol and try again."
                        )

# Add footer
st.markdown("---")
st.markdown("FiscalAgent: Financial Insights | Powered by Gemini 2.5 Pro")
st.markdown("---")
st.markdown("FiscalAgent: Financial Insights | Powered by Gemini 2.5 Pro")
