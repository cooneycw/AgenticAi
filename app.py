from shiny import App, ui, render, reactive
import asyncio
import json
import time
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import openai
from openai import OpenAI
import requests
import pandas as pd
from dotenv import load_dotenv
import os
import re
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta

# Load environment variables
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
FLIGHTRADAR24_KEY = os.getenv("FR24_KEY")


class AgentRole(Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    SYNTHESIZER = "synthesizer"


@dataclass
class AgentMessage:
    agent_role: AgentRole
    content: str
    timestamp: datetime
    tools_used: List[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class TaskResult:
    success: bool
    data: Any
    error: str = None


@dataclass
class TravelQuery:
    """Structured representation of a travel query"""
    regions: List[str]
    forecast_date: str
    forecast_date_parsed: datetime
    temperature_range: Tuple[int, int]
    origin_city: str
    additional_criteria: List[str]
    raw_query: str


class AgenticAISystem:
    def __init__(self, openai_key: str, openweather_key: str = None, flightradar24_key: str = None):
        self.client = OpenAI(api_key=openai_key)
        self.openweather_key = openweather_key
        self.flightradar24_key = flightradar24_key
        self.conversation_history: List[AgentMessage] = []

    async def parse_travel_query(self, query: str) -> TravelQuery:
        """Use OpenAI to intelligently parse the travel query into structured data"""
        try:
            system_prompt = """You are a travel query parser. Extract structured information from travel queries.

Parse the query and return a JSON object with these fields:
- regions: List of regions/countries to search (e.g., ["Europe", "South America", "Asia"])
- forecast_date: A natural language date (e.g., "tomorrow", "next Wednesday", "in 2 weeks")
- temperature_range: [min_temp, max_temp] in Celsius
- origin_city: Where the person is traveling from (if not specified, use "Toronto")
- additional_criteria: List of other requirements (e.g., ["morning flights", "direct flights", "under $500"])

Important:
- If no origin city is mentioned, default to "Toronto"
- Convert "short flights" to specific criteria like "under 6 hours"
- Be generous with temperature ranges if exact numbers aren't given
- Use common region names: "Europe", "Asia", "South America", "North America", "Africa", "Australia"

Examples:
Query: "Find European destinations with 20-27°C tomorrow"
Response: {"regions": ["Europe"], "forecast_date": "tomorrow", "temperature_range": [20, 27], "origin_city": "Toronto", "additional_criteria": []}

Query: "South American cities with warm weather next Wednesday, morning flights from Toronto"
Response: {"regions": ["South America"], "forecast_date": "next Wednesday", "temperature_range": [20, 30], "origin_city": "Toronto", "additional_criteria": ["morning flights"]}

Return only valid JSON, no explanations."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse this travel query: {query}"}
                ],
                max_tokens=300,
                temperature=0.1
            )

            parsed_data = json.loads(response.choices[0].message.content)

            # Parse the natural language date
            forecast_date_parsed = self._parse_natural_date(parsed_data.get("forecast_date", "tomorrow"))

            return TravelQuery(
                regions=parsed_data.get("regions", ["Europe"]),
                forecast_date=parsed_data.get("forecast_date", "tomorrow"),
                forecast_date_parsed=forecast_date_parsed,
                temperature_range=tuple(parsed_data.get("temperature_range", [20, 27])),
                origin_city=parsed_data.get("origin_city", "Toronto"),
                additional_criteria=parsed_data.get("additional_criteria", []),
                raw_query=query
            )

        except Exception as e:
            print(f"Query parsing failed: {e}")
            # Fallback to simple parsing
            return TravelQuery(
                regions=["Europe"],
                forecast_date="tomorrow",
                forecast_date_parsed=datetime.now() + timedelta(days=1),
                temperature_range=(20, 27),
                origin_city="Toronto",
                additional_criteria=[],
                raw_query=query
            )

    def _parse_natural_date(self, date_str: str) -> datetime:
        """Parse natural language dates into datetime objects"""
        date_str = date_str.lower().strip()
        today = datetime.now()

        if date_str in ["tomorrow"]:
            return today + timedelta(days=1)
        elif "next week" in date_str:
            return today + timedelta(weeks=1)
        elif "next month" in date_str:
            return today + relativedelta(months=1)
        elif "wednesday" in date_str:
            days_ahead = 2 - today.weekday()  # Wednesday is 2
            if days_ahead <= 0:
                days_ahead += 7
            return today + timedelta(days=days_ahead)
        elif "friday" in date_str:
            days_ahead = 4 - today.weekday()  # Friday is 4
            if days_ahead <= 0:
                days_ahead += 7
            return today + timedelta(days=days_ahead)
        elif "weekend" in date_str:
            days_ahead = 5 - today.weekday()  # Saturday is 5
            if days_ahead <= 0:
                days_ahead += 7
            return today + timedelta(days=days_ahead)
        elif re.search(r'\d+\s*(day|week|month)', date_str):
            # Handle "in 3 days", "in 2 weeks", "2 weeks", etc.
            match = re.search(r'(\d+)\s*(day|week|month)', date_str)
            if match:
                num = int(match.group(1))
                unit = match.group(2)
                if unit.startswith("day"):
                    return today + timedelta(days=num)
                elif unit.startswith("week"):
                    return today + timedelta(weeks=num)
                elif unit.startswith("month"):
                    return today + relativedelta(months=num)
        elif "2 weeks" in date_str or "two weeks" in date_str:
            return today + timedelta(weeks=2)
        elif "3 weeks" in date_str or "three weeks" in date_str:
            return today + timedelta(weeks=3)
        elif "january" in date_str:
            # Handle "in January" or "next January"
            next_jan = today.replace(month=1, day=15)
            if next_jan <= today:
                next_jan = next_jan.replace(year=today.year + 1)
            return next_jan

        # Fallback to tomorrow
        return today + timedelta(days=1)

    async def get_cities_for_regions(self, regions: List[str], max_cities_per_region: int = 8) -> List[Dict[str, str]]:
        """Use OpenAI to generate appropriate cities for any region"""
        try:
            all_cities = []

            for region in regions:
                if region.lower() in ["any", "global"]:
                    region = "globally popular destinations"
                elif region.lower() == "europe":
                    region = "Europe"
                elif region.lower() == "asia":
                    region = "Asia"
                elif region.lower() in ["south america", "south american"]:
                    region = "South America"
                elif region.lower() in ["north america", "north american"]:
                    region = "North America"

                system_prompt = f"""You are a travel expert. Generate a list of {max_cities_per_region} cities in {region} that:
1. Have major international airports
2. Are popular tourist or business destinations
3. Have populations over 300,000
4. Are geographically diverse across the region
5. Are likely to have good weather infrastructure

Return ONLY a JSON list in this format:
[{{"city": "Barcelona", "country": "Spain"}}, {{"city": "Rome", "country": "Italy"}}]

Focus on cities that travelers actually visit, not just largest cities."""

                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Generate {max_cities_per_region} diverse cities for {region}"}
                    ],
                    max_tokens=400,
                    temperature=0.7
                )

                try:
                    cities_data = json.loads(response.choices[0].message.content)
                    for city_data in cities_data:
                        city_data["region"] = region
                        all_cities.append(city_data)
                except json.JSONDecodeError:
                    print(f"Failed to parse cities for {region}")
                    continue

            return all_cities

        except Exception as e:
            print(f"City generation failed: {e}")
            # Fallback to a few well-known cities
            return [
                {"city": "Barcelona", "country": "Spain", "region": "Europe"},
                {"city": "Vancouver", "country": "Canada", "region": "North America"},
                {"city": "Sydney", "country": "Australia", "region": "Australia"}
            ]

    async def get_city_coordinates(self, city: str, country: str) -> Optional[Dict[str, float]]:
        """Get real coordinates for a city using free geocoding API"""
        try:
            import aiohttp
            query = f"{city}, {country}"
            url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={'User-Agent': 'AgenticAI-Travel/1.0'}) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data:
                            return {
                                'lat': float(data[0]['lat']),
                                'lon': float(data[0]['lon'])
                            }
            return None
        except Exception as e:
            print(f"Geocoding failed for {city}: {e}")
            return None

    async def get_weather_forecast(self, lat: float, lon: float, city_name: str, target_date: datetime) -> TaskResult:
        """Get weather forecast for a specific date using Open-Meteo API with fallback for distant dates"""
        try:
            import aiohttp

            # Check if target date is too far in the future (most APIs only go 14-16 days out)
            days_ahead = (target_date - datetime.now()).days

            if days_ahead > 16:
                # Use historical/seasonal data for dates too far in the future
                return await self.get_seasonal_weather_estimate(lat, lon, city_name, target_date)

            target_date_str = target_date.strftime('%Y-%m-%d')

            # Open-Meteo API for date-specific forecast
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m,weather_code&start_date={target_date_str}&end_date={target_date_str}&timezone=auto"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        daily = data.get('daily', {})

                        if all(key in daily and daily[key] for key in
                               ['temperature_2m_max', 'temperature_2m_min', 'relative_humidity_2m']):
                            max_temp = daily['temperature_2m_max'][0]
                            min_temp = daily['temperature_2m_min'][0]
                            avg_temp = (max_temp + min_temp) / 2
                            humidity = daily['relative_humidity_2m'][0]
                            weather_code = daily.get('weather_code', [0])[0]

                            # Weather condition from code
                            weather_desc = self._weather_code_to_description(weather_code)

                            return TaskResult(success=True, data={
                                'city': city_name,
                                'date': target_date_str,
                                'temp_max': max_temp,
                                'temp_min': min_temp,
                                'temp_avg': avg_temp,
                                'humidity': humidity,
                                'weather_desc': weather_desc,
                                'data_type': 'forecast',
                                'meets_criteria': False  # Will be set by analyzer
                            })
                    elif response.status == 400:
                        # API doesn't support this date range, fall back to seasonal estimates
                        return await self.get_seasonal_weather_estimate(lat, lon, city_name, target_date)

                    return TaskResult(success=False, data=None, error=f"Weather API error: {response.status}")
        except Exception as e:
            return TaskResult(success=False, data=None, error=f"Weather lookup failed: {str(e)}")

    async def get_seasonal_weather_estimate(self, lat: float, lon: float, city_name: str,
                                            target_date: datetime) -> TaskResult:
        """Provide seasonal weather estimates for dates too far in the future"""
        try:
            # Use AI to generate realistic seasonal weather estimates
            month = target_date.strftime('%B')

            system_prompt = f"""You are a meteorologist providing seasonal weather estimates for {city_name} in {month}.

Based on historical climate data and seasonal patterns, provide realistic weather estimates in JSON format:
{{
    "temp_max": 23.5,
    "temp_min": 16.2,
    "humidity": 62,
    "weather_desc": "Partly cloudy"
}}

Consider:
- Geographic location and climate zone
- Seasonal weather patterns for {month}
- Typical temperature ranges for this region
- Realistic humidity levels

Return only valid JSON, no explanations."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Provide seasonal weather estimate for {city_name} in {month}"}
                ],
                max_tokens=150,
                temperature=0.3
            )

            try:
                weather_data = json.loads(response.choices[0].message.content)

                max_temp = weather_data.get('temp_max', 20)
                min_temp = weather_data.get('temp_min', 15)
                avg_temp = (max_temp + min_temp) / 2
                humidity = weather_data.get('humidity', 60)
                weather_desc = weather_data.get('weather_desc', 'Variable conditions')

                return TaskResult(success=True, data={
                    'city': city_name,
                    'date': target_date.strftime('%Y-%m-%d'),
                    'temp_max': max_temp,
                    'temp_min': min_temp,
                    'temp_avg': avg_temp,
                    'humidity': humidity,
                    'weather_desc': weather_desc,
                    'data_type': 'seasonal_estimate',
                    'meets_criteria': False  # Will be set by analyzer
                })

            except json.JSONDecodeError:
                # Fallback to hardcoded seasonal estimates
                return self._get_hardcoded_seasonal_estimate(city_name, target_date)

        except Exception as e:
            # Final fallback to hardcoded estimates
            return self._get_hardcoded_seasonal_estimate(city_name, target_date)

    def _get_hardcoded_seasonal_estimate(self, city_name: str, target_date: datetime) -> TaskResult:
        """Hardcoded seasonal estimates as final fallback"""
        month = target_date.month

        # Rough seasonal estimates for European cities
        if "spain" in city_name.lower() or "barcelona" in city_name.lower():
            temps = {6: (26, 18), 7: (29, 21), 8: (29, 21), 12: (15, 7), 1: (14, 6), 2: (16, 8)}
        elif "italy" in city_name.lower() or "rome" in city_name.lower():
            temps = {6: (25, 17), 7: (28, 20), 8: (28, 20), 12: (14, 6), 1: (13, 4), 2: (15, 6)}
        elif "germany" in city_name.lower() or "berlin" in city_name.lower():
            temps = {6: (22, 13), 7: (24, 15), 8: (24, 15), 12: (5, -1), 1: (3, -3), 2: (5, -2)}
        elif "greece" in city_name.lower() or "athens" in city_name.lower():
            temps = {6: (28, 20), 7: (32, 23), 8: (32, 23), 12: (16, 9), 1: (14, 7), 2: (15, 8)}
        else:
            # Default European estimate
            temps = {6: (23, 15), 7: (26, 17), 8: (25, 16), 12: (10, 3), 1: (8, 1), 2: (10, 2)}

        max_temp, min_temp = temps.get(month, (20, 12))
        avg_temp = (max_temp + min_temp) / 2

        return TaskResult(success=True, data={
            'city': city_name,
            'date': target_date.strftime('%Y-%m-%d'),
            'temp_max': max_temp,
            'temp_min': min_temp,
            'temp_avg': avg_temp,
            'humidity': 65,
            'weather_desc': 'Seasonal average conditions',
            'data_type': 'hardcoded_estimate',
            'meets_criteria': False
        })

    def _weather_code_to_description(self, code: int) -> str:
        """Convert weather code to description"""
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
            67: "Heavy freezing rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
            82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        return weather_codes.get(code, "Unknown")

    async def calculate_flight_info(self, origin_city: str, dest_city: str, dest_country: str) -> Dict[str, Any]:
        """Calculate flight information between cities"""
        try:
            # Handle default origin city
            if origin_city.lower() in ["current location", "toronto"]:
                origin_city = "Toronto"
                origin_country = "Canada"
            else:
                origin_country = ""

            # Get coordinates for both cities
            origin_coords = await self.get_city_coordinates(origin_city, origin_country)
            dest_coords = await self.get_city_coordinates(dest_city, dest_country)

            if not origin_coords:
                return {"error": f"Could not get coordinates for {origin_city}"}
            if not dest_coords:
                return {"error": f"Could not get coordinates for {dest_city}"}

            # Calculate distance using Haversine formula
            distance_km = self._calculate_distance(
                origin_coords['lat'], origin_coords['lon'],
                dest_coords['lat'], dest_coords['lon']
            )

            # Estimate flight time and routing
            if distance_km < 1500:
                flight_time = distance_km / 700 + 0.5  # Short flights
                routing = "Direct flights likely"
            elif distance_km < 4000:
                flight_time = distance_km / 800 + 0.75  # Medium flights
                routing = "Direct or 1 stop"
            else:
                flight_time = distance_km / 850 + 1.5  # Long flights
                routing = "1-2 stops typical"

            hours = int(flight_time)
            minutes = int((flight_time - hours) * 60)

            return {
                'distance_km': distance_km,
                'flight_time_hours': flight_time,
                'flight_time_display': f"{hours}h {minutes}m",
                'routing': routing,
                'origin': origin_city,
                'destination': f"{dest_city}, {dest_country}"
            }

        except Exception as e:
            return {"error": f"Flight calculation failed: {str(e)}"}

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using Haversine formula"""
        import math

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth's radius in km
        return c * r

    async def call_agent(self, role: AgentRole, parsed_query: TravelQuery, context: str = "") -> AgentMessage:
        """Truly autonomous agents that make real decisions using OpenAI"""
        tools_used = []

        try:
            if role == AgentRole.PLANNER:
                # Planner creates a strategy based on the parsed query
                system_prompt = """You are a strategic planning agent for travel research. 
Given a parsed travel query, create a specific action plan for gathering data.

Consider:
- What data sources are needed
- What order to gather information
- How to optimize the research process
- Potential challenges and fallbacks

Be specific about the steps and reasoning."""

                query_summary = f"""
Query: {parsed_query.raw_query}
Regions: {parsed_query.regions}
Date: {parsed_query.forecast_date} ({parsed_query.forecast_date_parsed.strftime('%Y-%m-%d')})
Temperature: {parsed_query.temperature_range[0]}-{parsed_query.temperature_range[1]}°C
Origin: {parsed_query.origin_city}
Criteria: {parsed_query.additional_criteria}
"""

                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Create a research plan for: {query_summary}"}
                    ],
                    max_tokens=400,
                    temperature=0.7
                )

                agent_data = response.choices[0].message.content
                tools_used = ["strategic_analysis", "query_parsing"]

            elif role == AgentRole.RESEARCHER:
                # Researcher gathers actual data
                agent_data = "**RESEARCH IN PROGRESS:**\n\n"
                agent_data += f"📋 **Query Analysis:**\n"
                agent_data += f"   - Regions: {', '.join(parsed_query.regions)}\n"
                agent_data += f"   - Date: {parsed_query.forecast_date} ({parsed_query.forecast_date_parsed.strftime('%Y-%m-%d')})\n"
                agent_data += f"   - Temperature: {parsed_query.temperature_range[0]}-{parsed_query.temperature_range[1]}°C\n"
                agent_data += f"   - Origin: {parsed_query.origin_city}\n\n"
                metadata = {}

                try:
                    # Step 1: Get cities for the specified regions
                    cities = await self.get_cities_for_regions(parsed_query.regions, 6)
                    agent_data += f"✅ Generated {len(cities)} cities across {len(parsed_query.regions)} regions\n"
                    tools_used.append("ai_city_generation")

                    # Step 2: Get weather data for each city
                    weather_results = []
                    successful_forecasts = 0

                    agent_data += f"🌡️ Fetching weather forecasts for {parsed_query.forecast_date}...\n"

                    # Check if date is too far in future
                    days_ahead = (parsed_query.forecast_date_parsed - datetime.now()).days
                    if days_ahead > 16:
                        agent_data += f"⚠️ Date is {days_ahead} days ahead - using seasonal estimates (weather APIs only forecast ~16 days)\n"
                    else:
                        agent_data += f"✅ Date is {days_ahead} days ahead - using real weather forecast API\n"

                    agent_data += f"\n"

                    for i, city_data in enumerate(cities):
                        try:
                            agent_data += f"   🔍 Processing {city_data['city']}, {city_data['country']}...\n"
                            coords = await self.get_city_coordinates(city_data['city'], city_data['country'])
                            if coords:
                                agent_data += f"   📍 Coordinates: {coords['lat']:.2f}, {coords['lon']:.2f}\n"
                                weather_result = await self.get_weather_forecast(
                                    coords['lat'], coords['lon'],
                                    f"{city_data['city']}, {city_data['country']}",
                                    parsed_query.forecast_date_parsed
                                )
                                if weather_result.success:
                                    weather_results.append(weather_result.data)
                                    successful_forecasts += 1
                                    data_type = weather_result.data.get('data_type', 'forecast')
                                    type_icon = "🔮" if data_type == 'seasonal_estimate' else "📊" if data_type == 'forecast' else "📈"
                                    agent_data += f"   ✅ {city_data['city']}: {weather_result.data['temp_avg']:.1f}°C {type_icon}\n"
                                else:
                                    agent_data += f"   ❌ {city_data['city']}: Weather API failed - {weather_result.error[:60]}\n"
                            else:
                                agent_data += f"   ❌ {city_data['city']}: Could not get coordinates\n"
                        except Exception as e:
                            agent_data += f"   ❌ {city_data['city']}: Exception - {str(e)[:60]}\n"
                            continue

                    agent_data += f"\n✅ Retrieved weather forecasts for {successful_forecasts}/{len(cities)} cities\n"
                    tools_used.append("weather_api")

                    # Step 3: Get flight information for all cities (not just good weather ones)
                    flight_results = []
                    agent_data += f"\n✈️ Calculating flight information from {parsed_query.origin_city}...\n"

                    for weather_data in weather_results:
                        try:
                            city_name = weather_data['city'].split(',')[0].strip()
                            country_name = weather_data['city'].split(',')[1].strip() if ',' in weather_data[
                                'city'] else ""

                            flight_info = await self.calculate_flight_info(parsed_query.origin_city, city_name,
                                                                           country_name)
                            if 'error' not in flight_info:
                                combined_data = {**weather_data, **flight_info}
                                flight_results.append(combined_data)
                                agent_data += f"   ✈️ {city_name}: {flight_info['flight_time_display']}, {flight_info['distance_km']:.0f}km\n"
                            else:
                                agent_data += f"   ❌ {city_name}: Flight calculation failed\n"
                        except Exception as e:
                            agent_data += f"   ❌ {city_name}: Error - {str(e)[:50]}\n"
                            continue

                    agent_data += f"\n✅ Calculated flight data for {len(flight_results)} destinations\n"
                    tools_used.append("flight_calculations")

                    # Store results in metadata for other agents
                    agent_data += f"\n**RESEARCH SUMMARY:**\n"
                    agent_data += f"- Target date: {parsed_query.forecast_date_parsed.strftime('%A, %B %d, %Y')}\n"
                    agent_data += f"- Temperature criteria: {parsed_query.temperature_range[0]}-{parsed_query.temperature_range[1]}°C\n"
                    agent_data += f"- Origin: {parsed_query.origin_city}\n"
                    agent_data += f"- Complete data for: {len(flight_results)} destinations\n"

                    if flight_results:
                        avg_temp = sum(d['temp_avg'] for d in flight_results) / len(flight_results)
                        avg_flight_time = sum(d['flight_time_hours'] for d in flight_results) / len(flight_results)
                        agent_data += f"- Average temperature: {avg_temp:.1f}°C\n"
                        agent_data += f"- Average flight time: {avg_flight_time:.1f} hours\n"

                    # Store raw data for analysis
                    metadata = {
                        "weather_results": weather_results,
                        "flight_results": flight_results,
                        "parsed_query": asdict(parsed_query),
                        "research_success": True
                    }

                except Exception as e:
                    agent_data += f"\n❌ Research failed: {str(e)}\n"
                    metadata = {
                        "weather_results": [],
                        "flight_results": [],
                        "parsed_query": asdict(parsed_query),
                        "research_success": False,
                        "error": str(e)
                    }

            elif role == AgentRole.ANALYZER:
                # Analyzer processes the gathered data
                metadata = {}

                # Get research data from previous agent
                research_messages = [msg for msg in self.conversation_history if msg.agent_role == AgentRole.RESEARCHER]

                if research_messages and research_messages[-1].metadata.get('research_success'):
                    research_data = research_messages[-1].metadata
                    flight_results = research_data.get('flight_results', [])

                    if flight_results:
                        temp_min, temp_max = parsed_query.temperature_range

                        agent_data = "**ANALYSIS IN PROGRESS:**\n\n"

                        # Analyze each destination
                        analyzed_destinations = []
                        meeting_criteria = 0

                        for dest in flight_results:
                            try:
                                # Check if temperature meets criteria
                                meets_temp = temp_min <= dest['temp_avg'] <= temp_max
                                if meets_temp:
                                    meeting_criteria += 1

                                # Calculate efficiency scores (0-10 scale)
                                temp_target = (temp_min + temp_max) / 2
                                temp_score = 10 if meets_temp else max(0, 10 - abs(dest['temp_avg'] - temp_target) * 2)
                                flight_score = max(0, 10 - (dest['flight_time_hours'] / 15))  # Prefer shorter flights
                                humidity_score = 10 if 40 <= dest['humidity'] <= 70 else max(0, 10 - abs(
                                    dest['humidity'] - 55) / 5)

                                # Weight the scores
                                overall_score = (temp_score * 0.4) + (flight_score * 0.3) + (humidity_score * 0.3)

                                analyzed_destinations.append({
                                    **dest,
                                    'meets_criteria': meets_temp,
                                    'temp_score': temp_score,
                                    'flight_score': flight_score,
                                    'humidity_score': humidity_score,
                                    'overall_score': overall_score
                                })

                            except Exception as e:
                                # Skip destinations with incomplete data
                                continue

                        # Sort by overall score
                        analyzed_destinations.sort(key=lambda x: x['overall_score'], reverse=True)

                        agent_data += f"📊 **ANALYSIS RESULTS:**\n"
                        agent_data += f"- Temperature criteria: {temp_min}-{temp_max}°C\n"
                        agent_data += f"- Destinations meeting criteria: {meeting_criteria}/{len(analyzed_destinations)}\n"
                        agent_data += f"- Analysis date: {parsed_query.forecast_date} ({parsed_query.forecast_date_parsed.strftime('%Y-%m-%d')})\n\n"

                        if analyzed_destinations:
                            agent_data += "🏆 **TOP DESTINATIONS:**\n"
                            for i, dest in enumerate(analyzed_destinations[:5], 1):
                                city_name = dest['city'].split(',')[0]
                                status = "✅" if dest['meets_criteria'] else "⚠️"
                                agent_data += f"{i}. {status} **{city_name}** (Score: {dest['overall_score']:.1f}/10)\n"
                                agent_data += f"   🌡️ {dest['temp_avg']:.1f}°C | ✈️ {dest['flight_time_display']} | 💧 {dest['humidity']:.0f}%\n"
                                agent_data += f"   📍 {dest['weather_desc']} | 🛫 {dest['routing']}\n\n"

                            # Use AI for deeper analysis
                            top_cities = [d['city'].split(',')[0] for d in analyzed_destinations[:3]]
                            avg_flight_time = sum(d['flight_time_hours'] for d in analyzed_destinations[:3]) / 3

                            system_prompt = """You are a data analysis agent. Provide insights on travel destination analysis.
Focus on patterns, trade-offs, and strategic recommendations based on the data."""

                            analysis_context = f"""
Analysis Results:
- Query: {parsed_query.raw_query}
- Criteria: {temp_min}-{temp_max}°C on {parsed_query.forecast_date}
- Destinations meeting temperature: {meeting_criteria}/{len(analyzed_destinations)}
- Top 3 destinations: {', '.join(top_cities)}
- Average flight time to top 3: {avg_flight_time:.1f} hours
- Temperature range in results: {min(d['temp_avg'] for d in analyzed_destinations):.1f}-{max(d['temp_avg'] for d in analyzed_destinations):.1f}°C
"""

                            try:
                                response = self.client.chat.completions.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": f"Provide analysis insights: {analysis_context}"}
                                    ],
                                    max_tokens=300,
                                    temperature=0.7
                                )

                                agent_data += "🔍 **AI INSIGHTS:**\n"
                                agent_data += response.choices[0].message.content

                            except Exception as e:
                                agent_data += f"⚠️ AI analysis unavailable: {str(e)[:50]}"

                            metadata = {"analyzed_destinations": analyzed_destinations}
                            tools_used = ["data_analysis", "scoring_algorithm", "ranking_system", "ai_insights"]
                        else:
                            agent_data += "❌ No valid destinations to analyze"
                            metadata = {}
                            tools_used = ["data_analysis"]
                    else:
                        agent_data = "❌ No flight/weather data available from research phase"
                        metadata = {}
                        tools_used = []
                else:
                    agent_data = "❌ Research phase failed or no data available - cannot perform analysis"
                    metadata = {}
                    tools_used = []

            elif role == AgentRole.SYNTHESIZER:
                # Synthesizer creates final recommendations
                metadata = {}

                # Get analysis data
                analysis_messages = [msg for msg in self.conversation_history if msg.agent_role == AgentRole.ANALYZER]

                if analysis_messages and analysis_messages[-1].metadata.get('analyzed_destinations'):
                    destinations = analysis_messages[-1].metadata['analyzed_destinations']

                    agent_data = "**FINAL RECOMMENDATIONS:**\n\n"

                    if destinations:
                        # Prepare data for AI synthesis
                        top_destinations = destinations[:5]

                        summary_data = []
                        for i, dest in enumerate(top_destinations, 1):
                            city_name = dest['city'].split(',')[0]
                            status = "✅ MEETS CRITERIA" if dest['meets_criteria'] else "⚠️ CLOSE MATCH"

                            summary_data.append(f"""
{i}. {city_name} - {status}
   Temperature: {dest['temp_avg']:.1f}°C (Target: {parsed_query.temperature_range[0]}-{parsed_query.temperature_range[1]}°C)
   Flight: {dest['flight_time_display']} from {parsed_query.origin_city}
   Weather: {dest['weather_desc']}, Humidity: {dest['humidity']:.0f}%
   Routing: {dest['routing']}, Distance: {dest['distance_km']:.0f}km
   Overall Score: {dest['overall_score']:.1f}/10
""")

                        destinations_summary = "\n".join(summary_data)

                        # Use AI to create personalized recommendations
                        system_prompt = """You are a travel recommendation synthesis agent. Create specific, actionable travel recommendations.

Provide:
1. Top 3 destination recommendations with compelling reasons
2. Practical booking advice and timing
3. Weather insights and what to expect
4. Flight booking strategy
5. Alternative options if plans change

Be specific, enthusiastic, and actionable. Focus on value and experience."""

                        try:
                            response = self.client.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user",
                                     "content": f"Create travel recommendations based on:\n\nQuery: {parsed_query.raw_query}\nDate: {parsed_query.forecast_date}\nOrigin: {parsed_query.origin_city}\n\nAnalyzed Destinations:\n{destinations_summary}"}
                                ],
                                max_tokens=600,
                                temperature=0.8
                            )

                            agent_data += response.choices[0].message.content

                        except Exception as e:
                            # Fallback to structured recommendations if AI fails
                            agent_data += "🏆 **TOP RECOMMENDATIONS:**\n\n"

                            for i, dest in enumerate(top_destinations[:3], 1):
                                city_name = dest['city'].split(',')[0]
                                country_name = dest['city'].split(',')[1].strip() if ',' in dest['city'] else ""
                                status_emoji = "🎯" if dest['meets_criteria'] else "⭐"

                                agent_data += f"**{i}. {status_emoji} {city_name}, {country_name}**\n"
                                agent_data += f"   🌡️ Weather: {dest['temp_avg']:.1f}°C, {dest['weather_desc']}\n"
                                agent_data += f"   ✈️ Flight: {dest['flight_time_display']} ({dest['routing']})\n"
                                agent_data += f"   💡 Why: "

                                if dest['meets_criteria']:
                                    agent_data += "Perfect temperature match"
                                else:
                                    temp_diff = abs(dest['temp_avg'] - sum(parsed_query.temperature_range) / 2)
                                    agent_data += f"Close temperature match (only {temp_diff:.1f}°C off target)"

                                if dest['flight_time_hours'] < 8:
                                    agent_data += ", short flight time"

                                agent_data += f"\n   📊 Overall Score: {dest['overall_score']:.1f}/10\n\n"

                            agent_data += "💼 **BOOKING STRATEGY:**\n"
                            agent_data += f"- Book flights 2-8 weeks in advance for best prices\n"
                            agent_data += f"- Target date: {parsed_query.forecast_date_parsed.strftime('%A, %B %d, %Y')}\n"
                            agent_data += f"- Check weather forecasts 1 week before travel\n"
                            agent_data += f"- Consider flexible dates for better flight options\n"

                        agent_data += f"\n\n📅 **TRAVEL DATE:** {parsed_query.forecast_date_parsed.strftime('%A, %B %d, %Y')}\n"
                        agent_data += f"🌡️ **TARGET WEATHER:** {parsed_query.temperature_range[0]}-{parsed_query.temperature_range[1]}°C\n"
                        agent_data += f"🛫 **DEPARTURE:** {parsed_query.origin_city}\n"

                        # Additional criteria handling
                        if parsed_query.additional_criteria:
                            agent_data += f"📋 **SPECIAL REQUIREMENTS:** {', '.join(parsed_query.additional_criteria)}\n"

                        tools_used = ["recommendation_synthesis", "ai_personalization", "booking_optimization"]
                        metadata = {
                            "final_recommendations": top_destinations[:3],
                            "booking_date": parsed_query.forecast_date_parsed.strftime('%Y-%m-%d'),
                            "total_options": len(destinations)
                        }
                    else:
                        agent_data = "❌ No suitable destinations found based on your criteria. Consider:\n"
                        agent_data += "- Adjusting temperature range\n"
                        agent_data += "- Flexible dates\n"
                        agent_data += "- Different regions\n"
                        agent_data += "- Connecting flights for more options"
                        tools_used = ["fallback_recommendations"]
                        metadata = {}
                else:
                    agent_data = "❌ No analysis data available - previous agents may have failed.\n\n"
                    agent_data += "**SUGGESTED NEXT STEPS:**\n"
                    agent_data += "1. Check your query for clarity\n"
                    agent_data += "2. Ensure API connections are working\n"
                    agent_data += "3. Try a simpler query first\n"
                    agent_data += "4. Verify date format is reasonable"
                    tools_used = ["error_handling"]
                    metadata = {}

            message = AgentMessage(
                agent_role=role,
                content=agent_data,
                timestamp=datetime.now(),
                tools_used=tools_used,
                metadata=metadata if 'metadata' in locals() else {}
            )

            self.conversation_history.append(message)
            return message

        except Exception as e:
            error_message = AgentMessage(
                agent_role=role,
                content=f"❌ Agent {role.value} encountered error: {str(e)}",
                timestamp=datetime.now(),
                tools_used=[],
                metadata={"error": True}
            )
            self.conversation_history.append(error_message)
            return error_message

    async def run_agentic_workflow(self, user_query: str) -> List[AgentMessage]:
        """Run the complete agentic workflow with intelligent query parsing"""

        # Clear previous conversation
        self.conversation_history = []

        # Step 1: Parse the query intelligently
        parsed_query = await self.parse_travel_query(user_query)

        # Step 2: Run agent workflow
        workflow_messages = []

        # Planning
        plan_msg = await self.call_agent(AgentRole.PLANNER, parsed_query)
        workflow_messages.append(plan_msg)

        # Research
        research_msg = await self.call_agent(AgentRole.RESEARCHER, parsed_query, plan_msg.content)
        workflow_messages.append(research_msg)

        # Analysis
        analysis_msg = await self.call_agent(AgentRole.ANALYZER, parsed_query, research_msg.content)
        workflow_messages.append(analysis_msg)

        # Synthesis
        synthesis_msg = await self.call_agent(AgentRole.SYNTHESIZER, parsed_query, analysis_msg.content)
        workflow_messages.append(synthesis_msg)

        return workflow_messages


# Initialize system
ai_system = AgenticAISystem(OPENAI_KEY, OPENWEATHER_KEY, FLIGHTRADAR24_KEY) if OPENAI_KEY else None

# Shiny UI
app_ui = ui.page_fillable(
    ui.h1("🤖 Dynamic Agentic AI System", class_="text-center mb-4"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Configuration"),
            ui.input_password("api_key", "OpenAI API Key:", placeholder="sk-..."),
            ui.input_action_button("connect", "Connect", class_="btn-primary mb-3"),
            ui.hr(),

            ui.h4("Example Queries"),
            ui.input_action_button("example1", "🌍 European cities, 1 week", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example2", "🏝️ Asian destinations, Friday", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example3", "🌮 Mexico warm weather, 5 days", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example4", "❄️ European winter, 10 days", class_="btn-outline-info btn-sm mb-1"),

            width=300
        ),

        ui.div(
            ui.div(
                ui.input_text_area(
                    "user_query",
                    "Enter your travel query (any region, any date, any criteria):",
                    placeholder="e.g., 'Find warm Asian destinations for next Friday with short flights from Vancouver' or 'European cities with mild weather in 3 weeks, prefer morning departures'",
                    rows=3,
                    width="100%"
                ),
                ui.input_action_button("run_workflow", "🚀 Run AI Agents", class_="btn-success mb-4"),
                class_="mb-4"
            ),

            ui.div(
                ui.output_ui("status_display"),
                class_="mb-3"
            ),

            ui.div(
                ui.h3("AI Agent Workflow"),
                ui.output_ui("conversation_display"),
                class_="conversation-container"
            )
        )
    ),

    # Enhanced CSS
    ui.tags.style("""
        .conversation-container {
            max-height: 700px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 20px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        }
        .agent-message {
            margin-bottom: 25px;
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        .agent-message:hover {
            transform: translateY(-2px);
        }
        .agent-planner { 
            border-left-color: #007bff; 
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        }
        .agent-researcher { 
            border-left-color: #28a745; 
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        }
        .agent-analyzer { 
            border-left-color: #ffc107; 
            background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        }
        .agent-synthesizer { 
            border-left-color: #dc3545; 
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        }
        .agent-header {
            font-weight: bold;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 1.1em;
        }
        .tools-used {
            font-size: 0.85em;
            color: #666;
            font-style: italic;
            background: rgba(255,255,255,0.7);
            padding: 4px 8px;
            border-radius: 6px;
        }
        .timestamp {
            font-size: 0.8em;
            color: #888;
        }
        .btn-outline-info {
            margin-bottom: 5px;
            width: 100%;
            text-align: left;
            font-size: 0.85em;
        }
    """)
)


def server(input, output, session):
    workflow_status = reactive.Value(
        f"✅ Ready: OpenAI ✅, Weather API {'✅' if OPENWEATHER_KEY else '❌'}" if ai_system
        else "⚠️ Please connect OpenAI API key"
    )
    conversation_messages = reactive.Value([])

    @reactive.Effect
    @reactive.event(input.connect)
    def connect_ai_system():
        global ai_system
        openai_key = input.api_key() or OPENAI_KEY

        if openai_key:
            try:
                ai_system = AgenticAISystem(openai_key, OPENWEATHER_KEY, FLIGHTRADAR24_KEY)
                workflow_status.set(f"✅ Connected | Weather API {'✅' if OPENWEATHER_KEY else '❌'}")
            except Exception as e:
                workflow_status.set(f"❌ Connection failed: {str(e)}")
        else:
            workflow_status.set("⚠️ Please enter OpenAI API key")

    @reactive.Effect
    @reactive.event(input.example1)
    def load_example1():
        ui.update_text_area("user_query",
                            value="Find European destinations with temperatures between 18-25°C in 1 week, prefer short flights from London")

    @reactive.Effect
    @reactive.event(input.example2)
    def load_example2():
        ui.update_text_area("user_query",
                            value="Asian cities with warm weather next Friday, morning flights from Los Angeles")

    @reactive.Effect
    @reactive.event(input.example3)
    def load_example3():
        ui.update_text_area("user_query",
                            value="Mexican destinations with 25-30°C weather in 5 days, beachside preferred, flights from Houston")

    @reactive.Effect
    @reactive.event(input.example4)
    def load_example4():
        ui.update_text_area("user_query",
                            value="Find European winter destinations with 5-15°C in 10 days, flights from Toronto under 8 hours")

    @reactive.Effect
    @reactive.event(input.run_workflow)
    async def run_agentic_workflow():
        global ai_system

        if not ai_system:
            workflow_status.set("❌ Please connect to OpenAI API first")
            return

        if not input.user_query():
            workflow_status.set("⚠️ Please enter a query")
            return

        workflow_status.set("🔄 AI agents working...")
        conversation_messages.set([])

        try:
            messages = await ai_system.run_agentic_workflow(input.user_query())
            conversation_messages.set(messages)
            workflow_status.set("✅ AI workflow completed")
        except Exception as e:
            workflow_status.set(f"❌ Workflow failed: {str(e)}")

    @output
    @render.ui
    def status_display():
        status = workflow_status.get()
        alert_class = "alert-info" if "✅" in status else "alert-warning" if "⚠️" in status else "alert-danger"
        return ui.div(ui.p(status, class_="mb-0"), class_=f"alert {alert_class}")

    @output
    @render.ui
    def conversation_display():
        messages = conversation_messages.get()
        if not messages:
            return ui.div(
                ui.p("Ready for your travel query! Try any region, date, or criteria.",
                     class_="text-muted text-center"),
                style="padding: 40px;"
            )

        agent_info = {
            AgentRole.PLANNER: {"icon": "🎯", "name": "Strategic Planner", "class": "agent-planner"},
            AgentRole.RESEARCHER: {"icon": "🔍", "name": "Data Researcher", "class": "agent-researcher"},
            AgentRole.ANALYZER: {"icon": "📊", "name": "Analysis Engine", "class": "agent-analyzer"},
            AgentRole.SYNTHESIZER: {"icon": "🎯", "name": "Recommendation Synthesizer", "class": "agent-synthesizer"}
        }

        elements = []
        for msg in messages:
            info = agent_info[msg.agent_role]

            tools_display = ""
            if msg.tools_used:
                tools_display = f"Tools: {', '.join(msg.tools_used)}"

            elements.append(
                ui.div(
                    ui.div(
                        ui.span(f"{info['icon']} {info['name']}", class_="agent-name"),
                        ui.span(tools_display, class_="tools-used") if tools_display else "",
                        class_="agent-header"
                    ),
                    ui.div(msg.content, style="white-space: pre-wrap; line-height: 1.6;"),
                    ui.div(f"⏱️ {msg.timestamp.strftime('%H:%M:%S')}", class_="timestamp mt-2"),
                    class_=f"agent-message {info['class']}"
                )
            )

        return ui.div(*elements)


app = App(app_ui, server)

if __name__ == "__main__":
    app.run(debug=True)