"""
Weather tool for ToolAugmentedRAG.
"""

import requests
import json
from typing import Dict, Any, List, Optional
import re


class WeatherTool:
    """Tool for retrieving weather information."""
    
    def __init__(self):
        """Initialize the weather tool."""
        self.name = "Weather Tool"
        self.description = "Retrieves current weather information"
        # Using OpenWeatherMap API for demonstration
        # In a real application, you would use your own API key
        self.api_key = None  # Replace with your OpenWeatherMap API key if available
        self.use_mock_data = True  # Use mock data if no API key is available
    
    def extract_locations(self, query: str) -> List[str]:
        """
        Extract potential location names from the query.
        
        Args:
            query: The user query
            
        Returns:
            List of potential locations
        """
        # Common major cities
        common_cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
            "Austin", "Jacksonville", "Fort Worth", "Columbus", "San Francisco",
            "Charlotte", "Indianapolis", "Seattle", "Denver", "Washington",
            "Boston", "El Paso", "Nashville", "Detroit", "Portland",
            "London", "Paris", "Tokyo", "Sydney", "Berlin",
            "Rome", "Madrid", "Toronto", "Dubai", "Singapore"
        ]
        
        # Look for city names in the query
        potential_locations = []
        
        # Check for common cities
        for city in common_cities:
            if city.lower() in query.lower():
                potential_locations.append(city)
        
        # Look for location patterns like "in [Location]" or "weather in [Location]"
        location_patterns = [
            r"(?:in|at|for|of)\s+([A-Z][a-zA-Z\s]+)(?:,|\s+[A-Z]|\?|\.|\s*$)",
            r"(?:weather|temperature|forecast|conditions)\s+(?:in|at|for)\s+([A-Z][a-zA-Z\s]+)(?:,|\s+[A-Z]|\?|\.|\s*$)"
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, query)
            potential_locations.extend(matches)
        
        # Remove duplicates while preserving order
        unique_locations = []
        for location in potential_locations:
            if location.strip() and location.strip() not in unique_locations:
                unique_locations.append(location.strip())
        
        return unique_locations
    
    def get_weather_data(self, location: str) -> Dict[str, Any]:
        """
        Get current weather data for a location.
        
        Args:
            location: Location name
            
        Returns:
            Dictionary with weather data
        """
        if self.use_mock_data or not self.api_key:
            return self._get_mock_weather_data(location)
        
        try:
            # Make API request to OpenWeatherMap
            url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.api_key}&units=metric"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant data
                result = {
                    "location": location,
                    "temperature": {
                        "current": round(data["main"]["temp"], 1),
                        "feels_like": round(data["main"]["feels_like"], 1),
                        "min": round(data["main"]["temp_min"], 1),
                        "max": round(data["main"]["temp_max"], 1),
                        "unit": "°C"
                    },
                    "conditions": data["weather"][0]["description"],
                    "humidity": data["main"]["humidity"],
                    "wind": {
                        "speed": data["wind"]["speed"],
                        "unit": "m/s",
                        "direction": self._get_wind_direction(data["wind"]["deg"])
                    },
                    "pressure": data["main"]["pressure"],
                    "visibility": data.get("visibility", "Unknown"),
                    "timestamp": data["dt"]
                }
                
                return result
            else:
                return {
                    "location": location,
                    "error": f"Error retrieving data: {response.status_code}"
                }
        except Exception as e:
            return {
                "location": location,
                "error": f"Error retrieving data: {str(e)}"
            }
    
    def _get_mock_weather_data(self, location: str) -> Dict[str, Any]:
        """
        Generate mock weather data for demonstration purposes.
        
        Args:
            location: Location name
            
        Returns:
            Dictionary with mock weather data
        """
        import random
        from datetime import datetime
        
        # Map locations to somewhat realistic temperatures
        temp_map = {
            "New York": (5, 25),  # (min, max) temperature range in Celsius
            "Los Angeles": (15, 30),
            "Chicago": (0, 25),
            "Houston": (15, 35),
            "Phoenix": (20, 40),
            "London": (5, 20),
            "Paris": (5, 25),
            "Tokyo": (10, 30),
            "Sydney": (15, 30),
            "Berlin": (0, 25)
        }
        
        # Default temperature range
        temp_range = temp_map.get(location, (10, 30))
        
        # Generate random temperature within the range
        temp = round(random.uniform(temp_range[0], temp_range[1]), 1)
        
        # Possible weather conditions
        conditions = ["Clear", "Partly cloudy", "Cloudy", "Light rain", "Rain", "Thunderstorm", "Foggy", "Snowy", "Windy"]
        
        # Generate mock data
        result = {
            "location": location,
            "temperature": {
                "current": temp,
                "feels_like": round(temp + random.uniform(-2, 2), 1),
                "min": round(temp - random.uniform(0, 5), 1),
                "max": round(temp + random.uniform(0, 5), 1),
                "unit": "°C"
            },
            "conditions": random.choice(conditions),
            "humidity": random.randint(30, 90),
            "wind": {
                "speed": round(random.uniform(0, 10), 1),
                "unit": "m/s",
                "direction": random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
            },
            "pressure": random.randint(990, 1030),
            "visibility": random.randint(5, 20),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
    
    def _get_wind_direction(self, degrees: float) -> str:
        """
        Convert wind direction in degrees to cardinal direction.
        
        Args:
            degrees: Wind direction in degrees
            
        Returns:
            Cardinal direction (N, NE, E, etc.)
        """
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = round(degrees / 45) % 8
        return directions[index]
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the weather tool on a query.
        
        Args:
            query: The user query
            
        Returns:
            Dictionary with weather data results
        """
        # Extract potential locations
        locations = self.extract_locations(query)
        
        if not locations:
            return {
                "tool_name": self.name,
                "success": False,
                "message": "No locations identified in the query",
                "data": None
            }
        
        # Get data for each location
        results = []
        for location in locations:
            weather_data = self.get_weather_data(location)
            results.append(weather_data)
        
        return {
            "tool_name": self.name,
            "success": True,
            "message": f"Retrieved weather data for {len(results)} locations",
            "data": results
        }
