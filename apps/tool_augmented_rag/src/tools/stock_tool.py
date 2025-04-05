"""
Stock price tool for ToolAugmentedRAG.
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Any, List, Optional


class StockTool:
    """Tool for retrieving stock price information."""
    
    def __init__(self):
        """Initialize the stock tool."""
        self.name = "Stock Tool"
        self.description = "Retrieves current stock price information"
    
    def extract_ticker_symbols(self, query: str) -> List[str]:
        """
        Extract potential ticker symbols from the query.
        
        Args:
            query: The user query
            
        Returns:
            List of potential ticker symbols
        """
        # Common stock tickers often mentioned
        common_tickers = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "amazon": "AMZN",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "meta": "META",
            "facebook": "META",
            "tesla": "TSLA",
            "nvidia": "NVDA",
            "netflix": "NFLX",
            "disney": "DIS",
            "coca cola": "KO",
            "coca-cola": "KO",
            "walmart": "WMT",
            "ibm": "IBM",
            "intel": "INTC",
            "amd": "AMD",
            "jp morgan": "JPM",
            "jpmorgan": "JPM",
            "bank of america": "BAC",
            "goldman sachs": "GS",
            "dow jones": "^DJI",
            "s&p 500": "^GSPC",
            "s&p500": "^GSPC",
            "nasdaq": "^IXIC"
        }
        
        # Look for ticker symbols in the query (uppercase 1-5 letter words)
        words = query.split()
        potential_tickers = []
        
        # Check for explicit ticker symbols (uppercase 1-5 letter words)
        for word in words:
            # Remove punctuation
            clean_word = word.strip(",.!?():;'\"")
            if clean_word.isupper() and 1 <= len(clean_word) <= 5 and clean_word.isalpha():
                potential_tickers.append(clean_word)
        
        # Check for company names
        query_lower = query.lower()
        for company, ticker in common_tickers.items():
            if company in query_lower:
                potential_tickers.append(ticker)
        
        # Remove duplicates while preserving order
        unique_tickers = []
        for ticker in potential_tickers:
            if ticker not in unique_tickers:
                unique_tickers.append(ticker)
        
        return unique_tickers
    
    def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get current stock data for a ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock data
        """
        try:
            # Get stock info
            stock = yf.Ticker(ticker)
            
            # Get current price data
            data = stock.history(period="1d")
            
            if data.empty:
                return {
                    "ticker": ticker,
                    "error": "No data available for this ticker"
                }
            
            # Get company info
            info = stock.info
            
            # Extract relevant data
            result = {
                "ticker": ticker,
                "name": info.get("shortName", "Unknown"),
                "price": round(data["Close"].iloc[-1], 2),
                "currency": info.get("currency", "USD"),
                "change": round(data["Close"].iloc[-1] - data["Open"].iloc[0], 2),
                "change_percent": round(((data["Close"].iloc[-1] / data["Open"].iloc[0]) - 1) * 100, 2),
                "volume": data["Volume"].iloc[-1],
                "market_cap": info.get("marketCap", "Unknown"),
                "timestamp": data.index[-1].strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
        except Exception as e:
            return {
                "ticker": ticker,
                "error": f"Error retrieving data: {str(e)}"
            }
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the stock tool on a query.
        
        Args:
            query: The user query
            
        Returns:
            Dictionary with stock data results
        """
        # Extract potential ticker symbols
        tickers = self.extract_ticker_symbols(query)
        
        if not tickers:
            return {
                "tool_name": self.name,
                "success": False,
                "message": "No stock ticker symbols identified in the query",
                "data": None
            }
        
        # Get data for each ticker
        results = []
        for ticker in tickers:
            stock_data = self.get_stock_data(ticker)
            results.append(stock_data)
        
        return {
            "tool_name": self.name,
            "success": True,
            "message": f"Retrieved data for {len(results)} stocks",
            "data": results
        }
