"""
Web search tool for ToolAugmentedRAG.
"""

import requests
import json
from typing import Dict, Any, List, Optional
import re
from datetime import datetime


class WebSearchTool:
    """Tool for performing web searches."""
    
    def __init__(self):
        """Initialize the web search tool."""
        self.name = "Web Search Tool"
        self.description = "Performs web searches to find recent information"
        # In a real application, you would use a search API like Google Custom Search or Bing
        self.api_key = None  # Replace with your search API key if available
        self.use_mock_data = True  # Use mock data if no API key is available
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a web search for the query.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List of search result dictionaries
        """
        if self.use_mock_data or not self.api_key:
            return self._get_mock_search_results(query, num_results)
        
        try:
            # This is a placeholder for a real search API implementation
            # In a real application, you would use Google Custom Search API, Bing API, etc.
            # Example with Google Custom Search:
            # url = f"https://www.googleapis.com/customsearch/v1?key={self.api_key}&cx=YOUR_SEARCH_ENGINE_ID&q={query}&num={num_results}"
            # response = requests.get(url)
            # data = response.json()
            # 
            # results = []
            # for item in data.get("items", []):
            #     results.append({
            #         "title": item.get("title", ""),
            #         "link": item.get("link", ""),
            #         "snippet": item.get("snippet", ""),
            #         "source": "web"
            #     })
            # 
            # return results
            
            # For now, return mock data
            return self._get_mock_search_results(query, num_results)
        except Exception as e:
            return [{
                "title": "Error performing search",
                "link": "",
                "snippet": f"Error: {str(e)}",
                "source": "error"
            }]
    
    def _get_mock_search_results(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Generate mock search results for demonstration purposes.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List of mock search result dictionaries
        """
        # Define some mock search results based on common topics
        mock_results = {
            "stock market": [
                {
                    "title": "Stock Market Today: Latest News and Updates",
                    "link": "https://example.com/finance/stock-market-today",
                    "snippet": "Get the latest updates on the stock market, including market trends, stock movements, and expert analysis. Markets showed mixed results today with technology stocks leading gains.",
                    "source": "Example Financial News"
                },
                {
                    "title": "Global Markets: Asian Shares Rise, U.S. Futures Point Higher",
                    "link": "https://example.com/markets/global-update",
                    "snippet": "Asian shares rose on optimism about economic recovery, while U.S. futures indicated a positive opening. Investors are watching upcoming earnings reports and economic data.",
                    "source": "Example Market Watch"
                },
                {
                    "title": "Market Analysis: What's Driving Today's Stock Movements",
                    "link": "https://example.com/analysis/market-drivers",
                    "snippet": "Analysis of the factors influencing today's market movements, including economic indicators, corporate earnings, and geopolitical developments.",
                    "source": "Example Investing"
                }
            ],
            "weather forecast": [
                {
                    "title": "7-Day Weather Forecast: What to Expect This Week",
                    "link": "https://example.com/weather/forecast",
                    "snippet": "The latest 7-day weather forecast shows a mix of conditions with temperatures ranging from 15°C to 25°C. Expect rain midweek followed by clearing conditions.",
                    "source": "Example Weather Service"
                },
                {
                    "title": "Severe Weather Alert: Storm System Approaching",
                    "link": "https://example.com/weather/alerts",
                    "snippet": "Meteorologists are tracking a storm system that could bring heavy rain and strong winds to the region. Residents are advised to stay informed about developing conditions.",
                    "source": "Example Weather Center"
                },
                {
                    "title": "Climate Patterns: How Current Weather Fits Long-Term Trends",
                    "link": "https://example.com/climate/analysis",
                    "snippet": "Analysis of how current weather patterns compare to historical data and long-term climate trends. Researchers note increasing frequency of extreme weather events.",
                    "source": "Example Climate Institute"
                }
            ],
            "technology news": [
                {
                    "title": "Latest Tech Innovations: What's New in the Tech World",
                    "link": "https://example.com/tech/innovations",
                    "snippet": "Roundup of the latest technology innovations, including advancements in AI, quantum computing, and consumer electronics. Several major companies announced new products this week.",
                    "source": "Example Tech Review"
                },
                {
                    "title": "AI Breakthrough: New Model Shows Human-Like Reasoning",
                    "link": "https://example.com/ai/breakthrough",
                    "snippet": "Researchers have developed a new AI model that demonstrates unprecedented capabilities in reasoning and problem-solving, approaching human-like understanding in specific domains.",
                    "source": "Example Science Daily"
                },
                {
                    "title": "Tech Industry Update: Major Companies Report Quarterly Earnings",
                    "link": "https://example.com/business/tech-earnings",
                    "snippet": "Summary of quarterly earnings reports from major technology companies, showing strong growth in cloud services and AI-related products despite economic headwinds.",
                    "source": "Example Business News"
                }
            ]
        }
        
        # Default results for queries not in our mock database
        default_results = [
            {
                "title": f"Search Results for: {query}",
                "link": f"https://example.com/search?q={query}",
                "snippet": f"This is a mock search result for '{query}'. In a real application, this would be actual content from the web relevant to your query.",
                "source": "Example Search Engine"
            },
            {
                "title": f"Latest Information About: {query}",
                "link": f"https://example.com/info/{query}",
                "snippet": f"Find the most recent and relevant information about '{query}'. This mock result would contain excerpts from web pages that match your search terms.",
                "source": "Example Information Portal"
            },
            {
                "title": f"{query} - Comprehensive Guide",
                "link": f"https://example.com/guide/{query}",
                "snippet": f"A comprehensive guide to understanding {query}, including background information, recent developments, and expert analysis.",
                "source": "Example Knowledge Base"
            }
        ]
        
        # Find the best matching mock results
        results = None
        for key in mock_results:
            if key.lower() in query.lower():
                results = mock_results[key]
                break
        
        # Use default results if no match found
        if results is None:
            results = default_results
        
        # Add current date to make results seem recent
        current_date = datetime.now().strftime("%B %d, %Y")
        for result in results:
            result["date"] = current_date
        
        # Return the requested number of results
        return results[:num_results]
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the web search tool on a query.
        
        Args:
            query: The user query
            
        Returns:
            Dictionary with search results
        """
        # Perform the search
        results = self.search(query, num_results=5)
        
        return {
            "tool_name": self.name,
            "success": True,
            "message": f"Found {len(results)} search results",
            "data": results
        }
