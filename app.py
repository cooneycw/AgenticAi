from shiny import App, ui, render, reactive
import asyncio
import json
import time
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
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
import statistics
import random

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
class AgentUpdate:
    """Real-time updates from agents"""
    agent_role: AgentRole
    update_type: str  # "progress", "data", "complete", "error"
    content: str
    data: Dict[str, Any] = None
    progress_pct: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


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


@dataclass
class ClimateBaseline:
    """Historical climate baseline for a city/month"""
    city: str
    month: int
    avg_temp_high: float
    avg_temp_low: float
    avg_temp_mean: float
    avg_humidity: float
    avg_precipitation_days: int
    typical_weather_patterns: List[str]
    climate_zone: str
    data_source: str = "ai_historical_analysis"


@dataclass
class WeatherDeviation:
    """Analysis of how current forecast deviates from historical norms"""
    temperature_deviation_c: float
    humidity_deviation_pct: float
    deviation_severity: str  # "normal", "slight", "significant", "extreme"
    deviation_context: str
    traveler_impact: str
    confidence_level: str


class EnhancedClimateAnalyzer:
    """Enhanced climate analysis using OpenAI's historical knowledge"""

    def __init__(self, openai_client):
        self.client = openai_client
        self.climate_cache = {}  # Cache historical baselines

    async def get_historical_climate_baseline(self, city: str, country: str,
                                              target_month: int) -> ClimateBaseline:
        """Get historical climate baseline using OpenAI's knowledge"""

        cache_key = f"{city}_{country}_{target_month}"
        if cache_key in self.climate_cache:
            return self.climate_cache[cache_key]

        try:
            month_name = datetime(2024, target_month, 1).strftime('%B')

            system_prompt = """You are a climate data expert with access to comprehensive historical weather data. 
            Provide detailed historical climate information for specific cities and months based on long-term averages (30+ years of data).

            Return a JSON object with these precise fields:
            {
                "avg_temp_high": 23.5,
                "avg_temp_low": 16.2, 
                "avg_temp_mean": 19.8,
                "avg_humidity": 62,
                "avg_precipitation_days": 8,
                "typical_weather_patterns": ["partly cloudy mornings", "afternoon sunshine", "occasional evening showers"],
                "climate_zone": "Mediterranean",
                "seasonal_notes": "Peak tourist season with stable weather",
                "record_high": 35,
                "record_low": 8,
                "sunshine_hours_per_day": 7.5,
                "wind_speed_kmh": 12,
                "data_reliability": "high"
            }

            Base this on actual historical climate data patterns. Be precise and realistic."""

            user_prompt = f"""Provide historical climate baseline for {city}, {country} in {month_name}.

            Focus on:
            - Long-term averages (30+ year normals)
            - Typical weather patterns for this specific month
            - What tourists/travelers typically experience
            - Climate zone classification
            - Record extremes for context

            City: {city}, {country}
            Month: {month_name}"""

            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1  # Low temperature for factual data
            )

            try:
                climate_data = json.loads(response.choices[0].message.content)

                baseline = ClimateBaseline(
                    city=f"{city}, {country}",
                    month=target_month,
                    avg_temp_high=climate_data.get('avg_temp_high', 20),
                    avg_temp_low=climate_data.get('avg_temp_low', 10),
                    avg_temp_mean=climate_data.get('avg_temp_mean', 15),
                    avg_humidity=climate_data.get('avg_humidity', 60),
                    avg_precipitation_days=climate_data.get('avg_precipitation_days', 5),
                    typical_weather_patterns=climate_data.get('typical_weather_patterns', ["variable conditions"]),
                    climate_zone=climate_data.get('climate_zone', "temperate")
                )

                # Store additional metadata
                baseline.seasonal_notes = climate_data.get('seasonal_notes', '')
                baseline.record_high = climate_data.get('record_high', 30)
                baseline.record_low = climate_data.get('record_low', 0)
                baseline.sunshine_hours = climate_data.get('sunshine_hours_per_day', 5)
                baseline.data_reliability = climate_data.get('data_reliability', 'moderate')

                # Cache the result
                self.climate_cache[cache_key] = baseline
                return baseline

            except json.JSONDecodeError as e:
                print(f"Failed to parse climate data for {city}: {e}")
                return self._create_fallback_baseline(city, country, target_month)

        except Exception as e:
            print(f"Climate baseline lookup failed for {city}: {e}")
            return self._create_fallback_baseline(city, country, target_month)

    def _create_fallback_baseline(self, city: str, country: str, month: int) -> ClimateBaseline:
        """Create a basic fallback baseline when AI lookup fails"""
        # Simple seasonal estimates based on hemisphere and latitude
        if month in [12, 1, 2]:  # Winter
            temp_high, temp_low = 10, 2
        elif month in [3, 4, 5]:  # Spring
            temp_high, temp_low = 18, 8
        elif month in [6, 7, 8]:  # Summer
            temp_high, temp_low = 26, 16
        else:  # Fall
            temp_high, temp_low = 16, 6

        return ClimateBaseline(
            city=f"{city}, {country}",
            month=month,
            avg_temp_high=temp_high,
            avg_temp_low=temp_low,
            avg_temp_mean=(temp_high + temp_low) / 2,
            avg_humidity=65,
            avg_precipitation_days=6,
            typical_weather_patterns=["seasonal average conditions"],
            climate_zone="unknown",
            data_source="fallback_estimate"
        )

    async def analyze_weather_deviations(self, current_forecast: Dict[str, Any],
                                         historical_baseline: ClimateBaseline) -> WeatherDeviation:
        """Analyze how current forecast deviates from historical norms"""

        try:
            # Calculate temperature deviation
            forecast_temp = current_forecast.get('temp_avg', current_forecast.get('temp_max', 20))
            temp_deviation = forecast_temp - historical_baseline.avg_temp_mean

            # Calculate humidity deviation
            forecast_humidity = current_forecast.get('humidity', 60)
            humidity_deviation = forecast_humidity - historical_baseline.avg_humidity

            # Determine deviation severity
            severity = self._calculate_deviation_severity(temp_deviation, humidity_deviation)

            # Generate contextual analysis using AI
            context_analysis = await self._generate_deviation_context(
                current_forecast, historical_baseline, temp_deviation, humidity_deviation, severity
            )

            return WeatherDeviation(
                temperature_deviation_c=temp_deviation,
                humidity_deviation_pct=humidity_deviation,
                deviation_severity=severity,
                deviation_context=context_analysis['context'],
                traveler_impact=context_analysis['impact'],
                confidence_level=context_analysis['confidence']
            )

        except Exception as e:
            return WeatherDeviation(
                temperature_deviation_c=0,
                humidity_deviation_pct=0,
                deviation_severity="unknown",
                deviation_context=f"Analysis failed: {str(e)}",
                traveler_impact="Unknown impact",
                confidence_level="low"
            )

    def _calculate_deviation_severity(self, temp_deviation: float, humidity_deviation: float) -> str:
        """Calculate overall deviation severity based on temperature and humidity"""

        abs_temp_dev = abs(temp_deviation)
        abs_humidity_dev = abs(humidity_deviation)

        # Temperature deviation thresholds
        if abs_temp_dev >= 8:  # 8°C+ deviation is extreme
            temp_severity = "extreme"
        elif abs_temp_dev >= 5:  # 5-8°C deviation is significant
            temp_severity = "significant"
        elif abs_temp_dev >= 3:  # 3-5°C deviation is slight
            temp_severity = "slight"
        else:
            temp_severity = "normal"

        # Humidity deviation thresholds
        if abs_humidity_dev >= 25:  # 25%+ deviation is extreme
            humidity_severity = "extreme"
        elif abs_humidity_dev >= 15:  # 15-25% deviation is significant
            humidity_severity = "significant"
        elif abs_humidity_dev >= 10:  # 10-15% deviation is slight
            humidity_severity = "slight"
        else:
            humidity_severity = "normal"

        # Return the most severe of the two
        severity_levels = ["normal", "slight", "significant", "extreme"]
        temp_idx = severity_levels.index(temp_severity)
        humidity_idx = severity_levels.index(humidity_severity)

        return severity_levels[max(temp_idx, humidity_idx)]

    async def _generate_deviation_context(self, forecast: Dict[str, Any],
                                          baseline: ClimateBaseline,
                                          temp_deviation: float, humidity_deviation: float,
                                          severity: str) -> Dict[str, str]:
        """Generate contextual analysis of weather deviations using AI"""

        try:
            city_name = baseline.city
            month_name = datetime(2024, baseline.month, 1).strftime('%B')

            system_prompt = """You are a travel weather analyst. Explain weather deviations from historical norms in practical terms for travelers.

            Provide a JSON response with:
            {
                "context": "Brief explanation of the deviation and likely causes",
                "impact": "What this means for travelers (clothing, activities, comfort)",
                "confidence": "high/medium/low - how reliable is this analysis"
            }

            Be specific, practical, and traveler-focused. Mention if this is unusually good or bad for tourism."""

            deviation_summary = f"""
            Location: {city_name}
            Month: {month_name}
            Historical Average: {baseline.avg_temp_mean:.1f}°C, {baseline.avg_humidity:.0f}% humidity
            Current Forecast: {forecast.get('temp_avg', 0):.1f}°C, {forecast.get('humidity', 0):.0f}% humidity
            Temperature Deviation: {temp_deviation:+.1f}°C from normal
            Humidity Deviation: {humidity_deviation:+.0f}% from normal
            Severity: {severity}
            Climate Zone: {baseline.climate_zone}
            Typical Patterns: {', '.join(baseline.typical_weather_patterns[:2])}
            """

            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this weather deviation for travelers:\n\n{deviation_summary}"}
                ],
                max_tokens=300,
                temperature=0.3
            )

            try:
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                content = response.choices[0].message.content
                return {
                    "context": content[:100] + "..." if len(content) > 100 else content,
                    "impact": "Deviation from normal conditions detected",
                    "confidence": "medium"
                }

        except Exception as e:
            return {
                "context": f"Weather deviating {temp_deviation:+.1f}°C from historical average",
                "impact": "Check current conditions before travel",
                "confidence": "low"
            }


class StreamingAgenticAISystem:
    def __init__(self, openai_key: str, openweather_key: str = None, flightradar24_key: str = None):
        self.client = OpenAI(api_key=openai_key)
        self.openweather_key = openweather_key
        self.flightradar24_key = flightradar24_key
        self.conversation_history: List[AgentMessage] = []

        # Initialize enhanced climate analyzer
        self.climate_analyzer = EnhancedClimateAnalyzer(self.client)

    def safe_get_temp_range(self, parsed_query: TravelQuery) -> tuple:
        """Safely extract temperature range, with fallbacks"""
        try:
            temp_range = getattr(parsed_query, 'temperature_range', (20, 27))

            if not isinstance(temp_range, (tuple, list)):
                print(f"Warning: temperature_range is not tuple/list: {type(temp_range)}")
                return (20, 27)

            if len(temp_range) < 2:
                print(f"Warning: temperature_range has < 2 elements: {temp_range}")
                return (20, 27)

            # Return first two elements as tuple
            return (float(temp_range[0]), float(temp_range[1]))

        except Exception as e:
            print(f"Error accessing temperature_range: {e}")
            return (20, 27)

    async def parse_travel_query(self, query: str) -> TravelQuery:
        """Use OpenAI to intelligently parse the travel query into structured data"""
        print(f"DEBUG: Parsing query: {query}")

        try:
            system_prompt = """You are a travel query parser. Extract structured information from travel queries.

Parse the query and return a JSON object with these fields:
- regions: List of regions/countries to search (e.g., ["Europe", "South America", "Asia"])
- forecast_date: A natural language date (e.g., "tomorrow", "next Wednesday", "in 2 weeks")
- temperature_range: [min_temp, max_temp] in Celsius (ALWAYS provide exactly 2 numbers)
- origin_city: Where the person is traveling from (if not specified, use "Toronto")
- additional_criteria: List of other requirements (e.g., ["morning flights", "direct flights", "under $500"])

CRITICAL: temperature_range MUST always be exactly 2 numbers in an array!

Return only valid JSON, no explanations."""

            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse this travel query: {query}"}
                ],
                max_tokens=300,
                temperature=0.1
            )

            parsed_data = {}
            try:
                content = response.choices[0].message.content.strip()
                print(f"DEBUG: OpenAI response: {content}")
                parsed_data = json.loads(content)
                print(f"DEBUG: Parsed data: {parsed_data}")
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}, using fallback")
                parsed_data = {}

            # ROBUST temperature range handling - this is the key fix
            temp_range = parsed_data.get("temperature_range", [20, 27])
            print(f"DEBUG: Initial temp_range: {temp_range} (type: {type(temp_range)})")

            # Handle various input types
            if temp_range is None:
                temp_range = [20, 27]
            elif isinstance(temp_range, (int, float)):
                # Single number - create range around it
                temp_range = [temp_range - 3, temp_range + 3]
            elif isinstance(temp_range, str):
                # Try to parse string
                try:
                    temp_range = [float(x.strip()) for x in temp_range.split('-') if x.strip()]
                    if len(temp_range) != 2:
                        temp_range = [20, 27]
                except:
                    temp_range = [20, 27]
            elif isinstance(temp_range, list):
                # Handle list cases
                if len(temp_range) == 0:
                    temp_range = [20, 27]
                elif len(temp_range) == 1:
                    temp_range = [temp_range[0] - 3, temp_range[0] + 3]
                elif len(temp_range) > 2:
                    temp_range = temp_range[:2]
                # If exactly 2, keep as is
            else:
                # Unknown type
                temp_range = [20, 27]

            # Ensure we have exactly 2 valid numbers
            try:
                temp_range = [float(temp_range[0]), float(temp_range[1])]
                # Ensure logical order (min <= max)
                if temp_range[0] > temp_range[1]:
                    temp_range = [temp_range[1], temp_range[0]]
            except (ValueError, TypeError, IndexError):
                print(f"DEBUG: Failed to convert temp_range to numbers, using fallback")
                temp_range = [20, 27]

            print(f"DEBUG: Final temp_range: {temp_range}")

            # Parse date safely
            forecast_date_str = parsed_data.get("forecast_date", "tomorrow")
            try:
                forecast_date_parsed = self._parse_natural_date(forecast_date_str)
            except Exception as e:
                print(f"Date parsing failed: {e}, using tomorrow")
                forecast_date_parsed = datetime.now() + timedelta(days=1)
                forecast_date_str = "tomorrow"

            # Create the TravelQuery object
            result = TravelQuery(
                regions=parsed_data.get("regions", ["Europe"]),
                forecast_date=forecast_date_str,
                forecast_date_parsed=forecast_date_parsed,
                temperature_range=tuple(temp_range),  # Convert to tuple - guaranteed 2 elements
                origin_city=parsed_data.get("origin_city", "Toronto"),
                additional_criteria=parsed_data.get("additional_criteria", []),
                raw_query=query
            )

            print(
                f"DEBUG: Created TravelQuery with temp_range: {result.temperature_range} (type: {type(result.temperature_range)}, len: {len(result.temperature_range)})")
            return result

        except Exception as e:
            print(f"Query parsing completely failed: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")

            # Ultra-safe fallback
            result = TravelQuery(
                regions=["Europe"],
                forecast_date="tomorrow",
                forecast_date_parsed=datetime.now() + timedelta(days=1),
                temperature_range=(20, 27),  # Hard-coded safe tuple
                origin_city="Toronto",
                additional_criteria=[],
                raw_query=query
            )
            print(f"DEBUG: Using fallback TravelQuery with temp_range: {result.temperature_range}")
            return result

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

    async def get_cities_for_regions(self, regions: List[str], max_cities_per_region: int = 10) -> List[Dict[str, str]]:
        """Use OpenAI to generate diverse cities for any region with better randomization"""
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

                # Enhanced prompt for better diversity
                system_prompt = f"""You are a travel expert generating diverse cities in {region}. 

IMPORTANT: Include a mix of:
- 40% major tourist destinations (capitals, famous cities)
- 30% secondary cities (regional centers, cultural hubs)
- 20% emerging destinations (up-and-coming, lesser-known gems)
- 10% unique/unusual destinations (interesting but still accessible)

Requirements for each city:
1. Has international airport access (direct or via connections)
2. Population over 200,000 OR significant tourist infrastructure
3. Geographically diverse across the region
4. Mix of different climates/altitudes within region
5. Include some "hidden gems" that travelers might not immediately think of

AVOID: Repeating the same obvious cities (Paris, London, Rome) unless they're genuinely the best options.

Return EXACTLY {max_cities_per_region} cities in this JSON format:
[{{"city": "Barcelona", "country": "Spain"}}, {{"city": "Porto", "country": "Portugal"}}]

Focus on geographical and cultural diversity, not just population size."""

                # Add randomization to the prompt
                temperature = random.uniform(0.7, 0.9)  # Variable creativity

                response = self.client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",
                         "content": f"Generate {max_cities_per_region} diverse cities for {region}. Mix famous and lesser-known destinations. Be geographically diverse."}
                    ],
                    max_tokens=500,
                    temperature=temperature  # Variable randomness
                )

                try:
                    cities_data = json.loads(response.choices[0].message.content)
                    for city_data in cities_data:
                        city_data["region"] = region
                        all_cities.append(city_data)
                except json.JSONDecodeError:
                    print(f"Failed to parse cities for {region}")
                    continue

            # Shuffle the final list for additional randomization
            random.shuffle(all_cities)
            return all_cities

        except Exception as e:
            print(f"City generation failed: {e}")
            # Enhanced fallback with more diverse cities
            return [
                {"city": "Barcelona", "country": "Spain", "region": "Europe"},
                {"city": "Porto", "country": "Portugal", "region": "Europe"},
                {"city": "Krakow", "country": "Poland", "region": "Europe"},
                {"city": "Bruges", "country": "Belgium", "region": "Europe"},
                {"city": "Tallinn", "country": "Estonia", "region": "Europe"},
                {"city": "Valencia", "country": "Spain", "region": "Europe"}
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

    async def get_weather_forecast_with_climate_analysis(self, lat: float, lon: float,
                                                         city_name: str, target_date: datetime) -> TaskResult:
        """Enhanced weather forecast with historical climate analysis"""

        # Get the basic weather forecast (existing method)
        forecast_result = await self.get_weather_forecast(lat, lon, city_name, target_date)

        if not forecast_result.success:
            return forecast_result

        # Extract city and country from city_name
        if ',' in city_name:
            city, country = city_name.split(',', 1)
            city = city.strip()
            country = country.strip()
        else:
            city = city_name
            country = "Unknown"

        try:
            # Get historical baseline for this city/month
            historical_baseline = await self.climate_analyzer.get_historical_climate_baseline(
                city, country, target_date.month
            )

            # Analyze deviations from historical norms
            deviation_analysis = await self.climate_analyzer.analyze_weather_deviations(
                forecast_result.data, historical_baseline
            )

            # Enhanced forecast data with climate context
            enhanced_data = {
                **forecast_result.data,

                # Historical context
                'historical_baseline': {
                    'avg_temp_high': historical_baseline.avg_temp_high,
                    'avg_temp_low': historical_baseline.avg_temp_low,
                    'avg_temp_mean': historical_baseline.avg_temp_mean,
                    'avg_humidity': historical_baseline.avg_humidity,
                    'climate_zone': historical_baseline.climate_zone,
                    'typical_patterns': historical_baseline.typical_weather_patterns
                },

                # Deviation analysis
                'climate_deviation': {
                    'temp_deviation_c': deviation_analysis.temperature_deviation_c,
                    'humidity_deviation_pct': deviation_analysis.humidity_deviation_pct,
                    'severity': deviation_analysis.deviation_severity,
                    'context': deviation_analysis.deviation_context,
                    'traveler_impact': deviation_analysis.traveler_impact,
                    'confidence': deviation_analysis.confidence_level
                },

                # Enhanced flags
                'is_unusually_warm': deviation_analysis.temperature_deviation_c > 3,
                'is_unusually_cold': deviation_analysis.temperature_deviation_c < -3,
                'is_unusually_humid': deviation_analysis.humidity_deviation_pct > 15,
                'is_unusually_dry': deviation_analysis.humidity_deviation_pct < -15,
                'climate_alert': deviation_analysis.deviation_severity in ['significant', 'extreme']
            }

            return TaskResult(success=True, data=enhanced_data)

        except Exception as e:
            # If climate analysis fails, return basic forecast with error note
            enhanced_data = {
                **forecast_result.data,
                'climate_analysis_error': f"Historical analysis failed: {str(e)}",
                'climate_alert': False
            }
            return TaskResult(success=True, data=enhanced_data)

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
                model="gpt-4.1-nano",
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

    async def get_airport_code_for_city(self, city: str, country: str) -> Optional[str]:
        """Get IATA airport code for a city - enhanced for major cities"""
        # Manual mapping for major cities to ensure accuracy
        city_airport_map = {
            ("toronto", "canada"): "YYZ",
            ("barcelona", "spain"): "BCN",
            ("madrid", "spain"): "MAD",
            ("paris", "france"): "CDG",
            ("london", "uk"): "LHR",
            ("london", "united kingdom"): "LHR",
            ("rome", "italy"): "FCO",
            ("milan", "italy"): "MXP",
            ("amsterdam", "netherlands"): "AMS",
            ("berlin", "germany"): "BER",
            ("munich", "germany"): "MUC",
            ("zurich", "switzerland"): "ZUR",
            ("vienna", "austria"): "VIE",
            ("istanbul", "turkey"): "IST",
            ("athens", "greece"): "ATH",
            ("lisbon", "portugal"): "LIS",
            ("dublin", "ireland"): "DUB",
            ("copenhagen", "denmark"): "CPH",
            ("stockholm", "sweden"): "ARN",
            ("oslo", "norway"): "OSL",
            ("helsinki", "finland"): "HEL",
            ("prague", "czech republic"): "PRG",
            ("prague", "czechia"): "PRG",
            ("budapest", "hungary"): "BUD",
            ("warsaw", "poland"): "WAW",
            ("brussels", "belgium"): "BRU",
            ("tokyo", "japan"): "NRT",
            ("seoul", "south korea"): "ICN",
            ("singapore", "singapore"): "SIN",
            ("bangkok", "thailand"): "BKK",
            ("sydney", "australia"): "SYD",
            ("melbourne", "australia"): "MEL",
        }

        key = (city.lower(), country.lower())
        return city_airport_map.get(key)

    async def calculate_flight_info(self, origin_city: str, dest_city: str, dest_country: str) -> Dict[str, Any]:
        """Enhanced flight information with real FR24 API data"""
        try:
            # Handle default origin city - always default to Toronto
            if origin_city.lower() in ["current location", "toronto", ""]:
                origin_city = "Toronto"
                origin_country = "Canada"
            else:
                origin_country = ""

            # Get airport codes for both cities
            origin_airport = await self.get_airport_code_for_city(origin_city, origin_country)
            dest_airport = await self.get_airport_code_for_city(dest_city, dest_country)

            # Get coordinates for geographic calculations
            origin_coords = await self.get_city_coordinates(origin_city, origin_country)
            dest_coords = await self.get_city_coordinates(dest_city, dest_country)

            if not origin_coords:
                return {"error": f"Could not get coordinates for {origin_city}"}
            if not dest_coords:
                return {"error": f"Could not get coordinates for {dest_city}"}

            # Calculate basic geographic data
            distance_km = self._calculate_distance(
                origin_coords['lat'], origin_coords['lon'],
                dest_coords['lat'], dest_coords['lon']
            )

            # Always set fallback geographic estimates (ensure consistent data structure)
            if distance_km < 1500:
                flight_time = distance_km / 700 + 0.5
                routing = "Direct flights likely"
            elif distance_km < 4000:
                flight_time = distance_km / 800 + 0.75
                routing = "Direct or 1 stop"
            else:
                flight_time = distance_km / 850 + 1.5
                routing = "1-2 stops typical"

            hours = int(flight_time)
            minutes = int((flight_time - hours) * 60)

            # Initialize result with basic data
            result = {
                'distance_km': distance_km,
                'origin': origin_city,
                'destination': f"{dest_city}, {dest_country}",
                'origin_airport': origin_airport or "Unknown",
                'dest_airport': dest_airport or "Unknown",
                'fr24_data_available': False,
                'flight_time_hours': flight_time,
                'flight_time_display': f"{hours}h {minutes}m",
                'routing': routing,
                'data_source': 'geographic_estimate'
            }

            return result

        except Exception as e:
            return {"error": f"Enhanced flight calculation failed: {str(e)}"}

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

    async def stream_agent_workflow(self, user_query: str) -> AsyncGenerator[AgentUpdate, None]:
        """Stream real-time updates from the agentic workflow"""

        try:
            print(f"DEBUG: stream_agent_workflow started for query: {user_query}")

            # Step 1: Parse the query
            yield AgentUpdate(AgentRole.PLANNER, "progress", "🎯 Parsing travel query...", progress_pct=5.0)
            print("DEBUG: Yielded planner progress update")

            parsed_query = await self.parse_travel_query(user_query)
            temp_min, temp_max = self.safe_get_temp_range(parsed_query)

            yield AgentUpdate(AgentRole.PLANNER, "complete",
                              f"✅ Query parsed: {', '.join(parsed_query.regions)}, {temp_min}-{temp_max}°C, {parsed_query.forecast_date}",
                              data={"parsed_query": asdict(parsed_query)}, progress_pct=100.0)
            print("DEBUG: Yielded planner complete update")

            # Step 2: Research Phase with streaming updates
            yield AgentUpdate(AgentRole.RESEARCHER, "progress", "🔍 Starting enhanced research...", progress_pct=5.0)
            print("DEBUG: Yielded researcher start update")

            # Generate cities
            yield AgentUpdate(AgentRole.RESEARCHER, "progress",
                              f"🌍 Generating diverse cities for {', '.join(parsed_query.regions)}...",
                              progress_pct=15.0)

            try:
                cities = await self.get_cities_for_regions(parsed_query.regions,
                                                           8)  # Reduced from 12 to 8 for faster processing
                print(f"DEBUG: Generated {len(cities)} cities")
                yield AgentUpdate(AgentRole.RESEARCHER, "data",
                                  f"✅ Found {len(cities)} diverse cities: {', '.join([c['city'] for c in cities[:3]])}...",
                                  data={"cities": cities}, progress_pct=25.0)
            except Exception as e:
                print(f"DEBUG: City generation failed: {e}")
                yield AgentUpdate(AgentRole.RESEARCHER, "error", f"❌ City generation failed: {str(e)}",
                                  progress_pct=25.0)
                return

            # Weather analysis with streaming - simplified for debugging
            weather_results = []
            climate_alerts = []
            total_cities = min(len(cities), 6)  # Limit to 6 cities for faster processing
            cities = cities[:total_cities]

            yield AgentUpdate(AgentRole.RESEARCHER, "progress",
                              f"🌡️ Starting weather analysis for {total_cities} cities on {parsed_query.forecast_date}...",
                              progress_pct=30.0)

            for i, city_data in enumerate(cities):
                progress = 30.0 + (i / total_cities) * 40.0  # 30% to 70%

                yield AgentUpdate(AgentRole.RESEARCHER, "progress",
                                  f"  🌡️ Analyzing {city_data['city']}... ({i + 1}/{total_cities})",
                                  progress_pct=progress)

                try:
                    # Simplified weather analysis for debugging
                    coords = await self.get_city_coordinates(city_data['city'], city_data['country'])
                    if coords:
                        # Use simpler weather method first to isolate issues
                        weather_result = await self.get_weather_forecast(
                            coords['lat'], coords['lon'],
                            f"{city_data['city']}, {city_data['country']}",
                            parsed_query.forecast_date_parsed
                        )

                        if weather_result.success:
                            weather_results.append(weather_result.data)
                            temp = weather_result.data['temp_avg']

                            yield AgentUpdate(AgentRole.RESEARCHER, "data",
                                              f"    ✅ {city_data['city']}: {temp:.1f}°C",
                                              data={"latest_weather": weather_result.data},
                                              progress_pct=progress)
                        else:
                            yield AgentUpdate(AgentRole.RESEARCHER, "progress",
                                              f"    ❌ {city_data['city']}: Weather failed",
                                              progress_pct=progress)
                    else:
                        yield AgentUpdate(AgentRole.RESEARCHER, "progress",
                                          f"    ❌ {city_data['city']}: No coordinates",
                                          progress_pct=progress)
                except Exception as e:
                    print(f"DEBUG: Weather analysis failed for {city_data['city']}: {e}")
                    yield AgentUpdate(AgentRole.RESEARCHER, "progress",
                                      f"    ❌ {city_data['city']}: {str(e)[:30]}",
                                      progress_pct=progress)

            print(f"DEBUG: Weather analysis complete, {len(weather_results)} successful")

            # Simplified flight analysis
            yield AgentUpdate(AgentRole.RESEARCHER, "progress",
                              f"✈️ Starting flight analysis from {parsed_query.origin_city}...", progress_pct=70.0)

            flight_results = []
            for i, weather_data in enumerate(weather_results):
                city_name = weather_data['city'].split(',')[0].strip()
                country_name = weather_data['city'].split(',')[1].strip() if ',' in weather_data['city'] else ""

                progress = 70.0 + (i / len(weather_results)) * 20.0  # 70% to 90%
                yield AgentUpdate(AgentRole.RESEARCHER, "progress",
                                  f"  🛫 Flight analysis: {city_name}... ({i + 1}/{len(weather_results)})",
                                  progress_pct=progress)

                try:
                    flight_info = await self.calculate_flight_info(parsed_query.origin_city, city_name, country_name)
                    if 'error' not in flight_info:
                        combined_data = {**weather_data, **flight_info}
                        flight_results.append(combined_data)

                        flight_time = flight_info.get('flight_time_display', 'Unknown')
                        yield AgentUpdate(AgentRole.RESEARCHER, "data",
                                          f"    ✅ {city_name}: {flight_time}",
                                          data={"latest_flight": combined_data},
                                          progress_pct=progress)
                    else:
                        yield AgentUpdate(AgentRole.RESEARCHER, "progress",
                                          f"    ❌ {city_name}: Flight calculation failed",
                                          progress_pct=progress)
                except Exception as e:
                    print(f"DEBUG: Flight analysis failed for {city_name}: {e}")
                    yield AgentUpdate(AgentRole.RESEARCHER, "progress",
                                      f"    ❌ {city_name}: Flight error",
                                      progress_pct=progress)

            # Research summary
            yield AgentUpdate(AgentRole.RESEARCHER, "complete",
                              f"🎉 Research complete! {len(flight_results)} destinations ready",
                              data={"flight_results": flight_results, "climate_alerts": climate_alerts},
                              progress_pct=100.0)

            print(f"DEBUG: Research phase complete with {len(flight_results)} results")

            # Step 3: Analysis Phase - Simplified
            yield AgentUpdate(AgentRole.ANALYZER, "progress", "📊 Starting enhanced analysis...", progress_pct=10.0)

            if not flight_results:
                yield AgentUpdate(AgentRole.ANALYZER, "error", "❌ No flight results to analyze", progress_pct=100.0)
                return

            analyzed_destinations = []
            meeting_criteria = 0

            for i, dest in enumerate(flight_results):
                progress = 10.0 + (i / len(flight_results)) * 80.0
                city_name = dest['city'].split(',')[0]

                yield AgentUpdate(AgentRole.ANALYZER, "progress",
                                  f"  📊 Analyzing {city_name}... ({i + 1}/{len(flight_results)})",
                                  progress_pct=progress)

                try:
                    # Simplified scoring
                    meets_temp = temp_min <= dest['temp_avg'] <= temp_max
                    if meets_temp:
                        meeting_criteria += 1

                    # Simple scoring algorithm
                    temp_score = 10 if meets_temp else max(0, 10 - abs(dest['temp_avg'] - (temp_min + temp_max) / 2))
                    flight_score = max(0, 10 - (dest.get('flight_time_hours', 8.0) / 15))
                    overall_score = (temp_score * 0.6) + (flight_score * 0.4)

                    analyzed_destinations.append({
                        **dest,
                        'meets_criteria': meets_temp,
                        'temp_score': temp_score,
                        'flight_score': flight_score,
                        'overall_score': overall_score
                    })

                    status = "✅" if meets_temp else "⚠️"
                    yield AgentUpdate(AgentRole.ANALYZER, "data",
                                      f"    {status} {city_name}: Score {overall_score:.1f}/10",
                                      data={"analyzed_city": analyzed_destinations[-1]},
                                      progress_pct=progress)
                except Exception as e:
                    print(f"DEBUG: Analysis failed for {city_name}: {e}")
                    yield AgentUpdate(AgentRole.ANALYZER, "progress",
                                      f"    ❌ {city_name}: Analysis error",
                                      progress_pct=progress)

            # Sort by score
            analyzed_destinations.sort(key=lambda x: x['overall_score'], reverse=True)

            if analyzed_destinations:
                top_city = analyzed_destinations[0]['city'].split(',')[0]
                yield AgentUpdate(AgentRole.ANALYZER, "complete",
                                  f"🏆 Analysis complete! Top: {top_city} ({analyzed_destinations[0]['overall_score']:.1f}/10), {meeting_criteria}/{len(flight_results)} meet criteria",
                                  data={"analyzed_destinations": analyzed_destinations},
                                  progress_pct=100.0)
            else:
                yield AgentUpdate(AgentRole.ANALYZER, "complete",
                                  "❌ No valid destinations analyzed",
                                  progress_pct=100.0)

            print(f"DEBUG: Analysis phase complete")

            # Step 4: Synthesis Phase - Simplified
            yield AgentUpdate(AgentRole.SYNTHESIZER, "progress", "🎯 Generating recommendations...", progress_pct=50.0)

            if analyzed_destinations:
                top_destinations = analyzed_destinations[:3]

                yield AgentUpdate(AgentRole.SYNTHESIZER, "data",
                                  f"🏆 Top 3 destinations: {', '.join([d['city'].split(',')[0] for d in top_destinations])}",
                                  progress_pct=80.0)

                yield AgentUpdate(AgentRole.SYNTHESIZER, "complete",
                                  f"🎉 Complete! {len(top_destinations)} recommendations ready",
                                  data={
                                      "recommendations": top_destinations,
                                      "climate_summary": {
                                          "total_alerts": len(climate_alerts),
                                          "meeting_criteria": meeting_criteria
                                      }
                                  },
                                  progress_pct=100.0)
            else:
                yield AgentUpdate(AgentRole.SYNTHESIZER, "complete",
                                  "❌ No suitable destinations found for recommendations",
                                  progress_pct=100.0)

            print(f"DEBUG: Synthesis phase complete - workflow finished")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"DEBUG: Stream workflow error: {error_details}")
            yield AgentUpdate(AgentRole.PLANNER, "error",
                              f"❌ Workflow failed: {str(e)}\n\nDebug: {error_details[-200:]}",
                              progress_pct=0.0)

    async def generate_weather_appropriate_itinerary(self, city_data: Dict[str, Any], days: int = 4) -> str:
        """Generate weather-appropriate 4-day itinerary for a destination"""
        try:
            city_name = city_data['city'].split(',')[0].strip()
            country_name = city_data['city'].split(',')[1].strip() if ',' in city_data['city'] else ""
            temp_avg = city_data['temp_avg']
            weather_desc = city_data['weather_desc']
            humidity = city_data.get('humidity', 60)

            # Climate context
            climate_context = ""
            if 'climate_deviation' in city_data:
                deviation = city_data['climate_deviation']
                if deviation['severity'] != 'normal':
                    climate_context = f" Note: Weather is {deviation['severity']} - {deviation['context']}"

            system_prompt = f"""You are an expert travel itinerary planner specializing in weather-appropriate activities.

Create a detailed {days}-day itinerary for {city_name}, {country_name} considering the specific weather conditions.

Weather Context:
- Temperature: {temp_avg:.1f}°C
- Conditions: {weather_desc}
- Humidity: {humidity}%{climate_context}

Your itinerary should:
1. Match activities to weather (indoor/outdoor balance)
2. Include specific recommendations for weather gear/clothing
3. Suggest best times of day for outdoor activities
4. Include backup indoor options for each day
5. Consider comfort levels for walking/sightseeing
6. Include local weather-appropriate experiences (e.g., hot weather = parks/fountains, cool weather = museums/cafes)

Format as:
**DAY 1: [Theme]**
- Morning (9-12): [Activity] - [Weather consideration]
- Afternoon (12-17): [Activity] - [Weather consideration]  
- Evening (17-21): [Activity] - [Weather consideration]
- Weather Gear: [Specific recommendations]
- Backup Plan: [Indoor alternative]

Focus on realistic, enjoyable activities that work well in these specific weather conditions."""

            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": f"Create a {days}-day weather-appropriate itinerary for {city_name}, {country_name}"}
                ],
                max_tokens=800,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"**ITINERARY GENERATION FAILED**\n\nError: {str(e)}\n\nFallback: Consider weather conditions ({city_data.get('temp_avg', 20):.1f}°C, {city_data.get('weather_desc', 'variable')}) when planning activities. Pack appropriately and have indoor backup plans."


# Initialize system
ai_system = StreamingAgenticAISystem(OPENAI_KEY, OPENWEATHER_KEY, FLIGHTRADAR24_KEY) if OPENAI_KEY else None

# Enhanced Streaming UI
app_ui = ui.page_fillable(
    ui.h1("🚀 Streaming Agentic AI Travel System", class_="text-center mb-4"),
    ui.p(
        "🌡️ Climate Intelligence • ✈️ Flight Analysis • 🔄 Real-Time Streaming • 🗓️ Weather-Appropriate Itineraries",
        class_="text-center text-muted mb-4"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Configuration"),
            ui.input_password("api_key", "OpenAI API Key:", placeholder="sk-..."),
            ui.input_action_button("connect", "Connect", class_="btn-primary mb-3"),
            ui.hr(),

            ui.h4("Enhanced Examples"),
            ui.input_action_button("example1", "🌍 Europe: Climate Analysis", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example2", "🏝️ Asia: Warm + Streaming", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example3", "🌮 Mexico: Beach Weather", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example4", "❄️ Dublin Test: Cooler Weather", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example5", "🌡️ Climate Anomaly Test", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example6", "🗺️ Diverse Cities Streaming", class_="btn-outline-info btn-sm mb-1"),

            ui.hr(),
            ui.h5("🚀 Streaming Features"),
            ui.p("✅ Real-Time Agent Updates", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Progressive Results Display", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Live Progress Tracking", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Visual Workflow Status", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Instant Feedback Loop", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Enhanced Error Handling", style="font-size: 0.85em; margin: 2px 0;"),

            width=300
        ),

        ui.div(
            ui.div(
                ui.input_text_area(
                    "user_query",
                    "Enter your travel query for streaming analysis:",
                    placeholder="e.g., 'Find cooler European destinations with 5-15°C in 10 days from Toronto' - watch the real-time agent workflow!",
                    rows=3,
                    width="100%"
                ),
                ui.input_action_button("run_workflow", "🚀 Start Streaming AI Agents", class_="btn-success mb-4"),
                class_="mb-4"
            ),

            # Agent Workflow Visual
            ui.div(
                ui.h4("🔄 Agent Workflow Status"),
                ui.output_ui("workflow_visual"),
                class_="mb-4"
            ),

            # Live Updates Display
            ui.div(
                ui.h4("📡 Live Agent Updates"),
                ui.output_ui("live_updates_display"),
                class_="mb-4"
            ),

            # Final Results Display
            ui.div(
                ui.h3("🎯 Final Recommendations & Itineraries"),
                ui.output_ui("final_results_display"),
                class_="conversation-container"
            )
        )
    ),

    # Enhanced CSS for streaming interface
    ui.tags.style("""
        .conversation-container {
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 20px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        }

        .live-updates-container {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
            font-family: 'Courier New', monospace;
        }

        .workflow-visual {
            padding: 15px;
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-radius: 12px;
            border: 1px solid #2196f3;
        }

        .agent-card {
            background: white;
            border-radius: 8px;
            padding: 12px;
            margin: 5px;
            text-align: center;
            min-height: 80px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .agent-pending {
            background: #f8f9fa;
            color: #6c757d;
            border: 2px solid #dee2e6;
        }

        .agent-active {
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            border: 2px solid #0056b3;
            animation: pulse 2s infinite;
        }

        .agent-complete {
            background: linear-gradient(135deg, #28a745, #1e7e34);
            color: white;
            border: 2px solid #1e7e34;
        }

        .agent-error {
            background: linear-gradient(135deg, #dc3545, #c82333);
            color: white;
            border: 2px solid #c82333;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        .update-progress { color: #007bff; }
        .update-data { color: #28a745; font-weight: bold; }
        .update-complete { color: #17a2b8; font-weight: bold; }
        .update-error { color: #dc3545; font-weight: bold; }

        .progress-bar {
            transition: width 0.5s ease;
        }

        .btn-outline-info {
            margin-bottom: 5px;
            width: 100%;
            text-align: left;
            font-size: 0.85em;
        }

        .timestamp {
            font-size: 0.75em;
            color: #6c757d;
        }
    """)
)


def server(input, output, session):
    workflow_status = reactive.Value(
        f"✅ Streaming System Ready: OpenAI ✅, Weather API {'✅' if OPENWEATHER_KEY else '❌'}, Climate Intelligence ✅" if ai_system
        else "⚠️ Please connect OpenAI API key for streaming features"
    )

    # Streaming reactive values
    live_updates = reactive.Value([])
    agent_states = reactive.Value({
        'planner': 'pending',
        'researcher': 'pending',
        'analyzer': 'pending',
        'synthesizer': 'pending'
    })
    final_results = reactive.Value({})
    workflow_active = reactive.Value(False)

    @reactive.Effect
    @reactive.event(input.connect)
    def connect_ai_system():
        global ai_system
        openai_key = input.api_key() or OPENAI_KEY

        if openai_key:
            try:
                ai_system = StreamingAgenticAISystem(openai_key, OPENWEATHER_KEY, FLIGHTRADAR24_KEY)
                workflow_status.set(
                    f"✅ Streaming System Connected | Weather API {'✅' if OPENWEATHER_KEY else '❌'} | Climate Intelligence ✅")
            except Exception as e:
                workflow_status.set(f"❌ Connection failed: {str(e)}")
        else:
            workflow_status.set("⚠️ Please enter OpenAI API key")

    # Example loading functions
    @reactive.Effect
    @reactive.event(input.example1)
    def load_example1():
        ui.update_text_area("user_query",
                            value="Find European destinations with temperatures between 18-25°C in 1 week, include climate analysis and streaming updates")

    @reactive.Effect
    @reactive.event(input.example2)
    def load_example2():
        ui.update_text_area("user_query",
                            value="Asian cities with warm weather next Friday, watch real-time agent workflow and climate analysis")

    @reactive.Effect
    @reactive.event(input.example3)
    def load_example3():
        ui.update_text_area("user_query",
                            value="Mexican coastal destinations with 25-30°C weather in 5 days, stream climate patterns analysis")

    @reactive.Effect
    @reactive.event(input.example4)
    def load_example4():
        ui.update_text_area("user_query",
                            value="Find cooler European destinations with 5-15°C in 10 days, show live weather anomaly detection")

    @reactive.Effect
    @reactive.event(input.example5)
    def load_example5():
        ui.update_text_area("user_query",
                            value="South American cities next month, stream climate deviation analysis in real-time")

    @reactive.Effect
    @reactive.event(input.example6)
    def load_example6():
        ui.update_text_area("user_query",
                            value="Diverse European cities in 2 weeks, stream diverse city generation and detailed itineraries")

    @reactive.Effect
    @reactive.event(input.run_workflow)
    def run_streaming_workflow():
        global ai_system

        if not ai_system:
            workflow_status.set("❌ Please connect to OpenAI API first")
            return

        if not input.user_query():
            workflow_status.set("⚠️ Please enter a query")
            return

        # Reset state
        workflow_active.set(True)
        live_updates.set([])
        final_results.set({})
        agent_states.set({
            'planner': 'pending',
            'researcher': 'pending',
            'analyzer': 'pending',
            'synthesizer': 'pending'
        })

        workflow_status.set("🔄 Streaming AI agents starting...")

        # Run the streaming workflow in the background
        asyncio.create_task(streaming_workflow_task(input.user_query()))

    async def streaming_workflow_task(query: str):
        """Background task for streaming workflow"""
        try:
            print(f"DEBUG: Starting streaming workflow for query: {query}")
            current_agent = None
            updates_list = []

            # Add initial update
            updates_list.append({
                'agent': 'system',
                'type': 'progress',
                'content': '🚀 Initializing streaming workflow...',
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'progress': 0
            })
            live_updates.set(updates_list.copy())

            update_count = 0
            async for update in ai_system.stream_agent_workflow(query):
                update_count += 1
                print(f"DEBUG: Received update #{update_count}: {update.agent_role.value} - {update.content[:50]}...")

                # Update agent states
                current_states = agent_states.get()

                # Reset previous agent to complete if we're on a new agent
                if current_agent and current_agent != update.agent_role.value:
                    current_states[current_agent] = 'complete'
                    print(f"DEBUG: Setting {current_agent} to complete")

                # Set current agent state
                if update.update_type == "error":
                    current_states[update.agent_role.value] = 'error'
                elif update.update_type == "complete":
                    current_states[update.agent_role.value] = 'complete'
                else:
                    current_states[update.agent_role.value] = 'active'

                current_agent = update.agent_role.value
                agent_states.set(current_states.copy())
                print(f"DEBUG: Agent states updated: {current_states}")

                # Add to live updates
                updates_list.append({
                    'agent': update.agent_role.value,
                    'type': update.update_type,
                    'content': update.content,
                    'timestamp': update.timestamp.strftime("%H:%M:%S"),
                    'progress': update.progress_pct
                })

                # Keep only last 50 updates for performance
                if len(updates_list) > 50:
                    updates_list = updates_list[-50:]

                live_updates.set(updates_list.copy())
                print(f"DEBUG: Live updates count: {len(updates_list)}")

                # Store final results
                if update.update_type == "complete" and update.data:
                    current_results = final_results.get()
                    current_results[update.agent_role.value] = update.data
                    final_results.set(current_results.copy())
                    print(f"DEBUG: Stored results for {update.agent_role.value}")

                # Small delay for UI responsiveness
                await asyncio.sleep(0.1)

            print(f"DEBUG: Streaming completed with {update_count} total updates")
            workflow_status.set("✅ Streaming workflow completed successfully!")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR: Streaming workflow failed: {error_details}")
            workflow_status.set(f"❌ Streaming workflow failed: {str(e)}")

            # Add error to updates
            try:
                updates_list = live_updates.get()
                updates_list.append({
                    'agent': 'system',
                    'type': 'error',
                    'content': f"❌ Workflow failed: {str(e)}",
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'progress': 0
                })
                live_updates.set(updates_list)
            except Exception as update_error:
                print(f"ERROR: Could not update live_updates: {update_error}")

        finally:
            workflow_active.set(False)
            print("DEBUG: Workflow marked as inactive")

    @output
    @render.ui
    def workflow_visual():
        """Visual representation of agent workflow progress"""
        states = agent_states.get()

        agents = [
            {'name': 'Planner', 'icon': '🎯', 'key': 'planner'},
            {'name': 'Researcher', 'icon': '🔍', 'key': 'researcher'},
            {'name': 'Analyzer', 'icon': '📊', 'key': 'analyzer'},
            {'name': 'Synthesizer', 'icon': '🎯', 'key': 'synthesizer'}
        ]

        elements = []
        for i, agent in enumerate(agents):
            state = states.get(agent['key'], 'pending')

            status_icons = {
                'pending': '⏳',
                'active': '🔄',
                'complete': '✅',
                'error': '❌'
            }

            elements.append(
                ui.div(
                    ui.div(
                        ui.h6(f"{agent['icon']} {agent['name']}", class_="mb-1"),
                        ui.div(status_icons[state], style="font-size: 1.5em;"),
                        class_=f"agent-card agent-{state}"
                    ),
                    class_="col-md-3"
                )
            )

            # Add arrow between agents (except last)
            if i < len(agents) - 1:
                elements.append(
                    ui.div(
                        ui.div("➜", style="font-size: 2em; text-align: center; padding-top: 20px;"),
                        class_="col-md-auto"
                    )
                )

        return ui.div(
            ui.div(*elements, class_="row align-items-center justify-content-center"),
            class_="workflow-visual"
        )

    @output
    @render.ui
    def live_updates_display():
        """Display live streaming updates"""
        updates = live_updates.get()

        if not updates:
            if workflow_active.get():
                return ui.div("🔄 Waiting for agent updates...", class_="text-center p-3 text-muted")
            else:
                return ui.div("Ready for streaming workflow...", class_="text-center p-3 text-muted")

        elements = []

        # Show recent updates (last 20)
        recent_updates = updates[-20:] if len(updates) > 20 else updates

        for update in recent_updates:
            # Icon and color based on type
            type_info = {
                'progress': {'icon': '🔄', 'class': 'update-progress'},
                'data': {'icon': '✅', 'class': 'update-data'},
                'complete': {'icon': '🎉', 'class': 'update-complete'},
                'error': {'icon': '❌', 'class': 'update-error'}
            }

            info = type_info.get(update['type'], {'icon': 'ℹ️', 'class': 'text-muted'})

            elements.append(
                ui.div(
                    f"[{update['timestamp']}] {info['icon']} {update['content']}",
                    class_=f"{info['class']} mb-1",
                    style="font-size: 0.9em; line-height: 1.4;"
                )
            )

        return ui.div(
            *elements,
            class_="live-updates-container"
        )

    @output
    @render.ui
    def final_results_display():
        """Display final results and recommendations"""
        results = final_results.get()

        if not results:
            return ui.div("Final results will appear here after workflow completion...",
                          class_="text-center p-4 text-muted")

        elements = []

        # Check if we have synthesizer results (final recommendations)
        if 'synthesizer' in results:
            synth_data = results['synthesizer']
            recommendations = synth_data.get('recommendations', [])
            itineraries = synth_data.get('itineraries', {})
            climate_summary = synth_data.get('climate_summary', {})

            if recommendations:
                elements.append(ui.h4("🏆 Top Recommendations"))

                for i, dest in enumerate(recommendations[:3], 1):
                    city_name = dest['city'].split(',')[0]
                    status = "🎯" if dest.get('meets_criteria') else "⭐"
                    score = dest.get('overall_score', 0)

                    # Climate indicators
                    indicators = []
                    if dest.get('climate_bonus', 0) > 0:
                        indicators.append("🌟")
                    if dest.get('climate_penalty', 0) > 0:
                        indicators.append("⚡")

                    indicator_str = "".join(indicators)

                    elements.append(
                        ui.div(
                            ui.h5(f"{i}. {status} {city_name} {indicator_str} (Score: {score:.1f}/10)"),
                            ui.p(
                                f"🌡️ {dest['temp_avg']:.1f}°C | ✈️ {dest.get('flight_time_display', 'N/A')} | 💧 {dest['humidity']:.0f}%"),
                            ui.p(f"📍 {dest['weather_desc']} | 🛫 {dest.get('routing', 'N/A')}"),

                            # Add itinerary if available
                            ui.div(
                                ui.h6(f"🗓️ 4-Day Weather-Appropriate Itinerary:"),
                                ui.div(
                                    itineraries.get(city_name, "Itinerary not available"),
                                    style="white-space: pre-wrap; background: #f8f9fa; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 0.85em;"
                                ),
                                class_="mt-2"
                            ) if city_name in itineraries else ui.div(),

                            class_="alert alert-success mb-3 p-3"
                        )
                    )

                # Climate summary
                if climate_summary:
                    elements.append(
                        ui.div(
                            ui.h5("🌡️ Climate Analysis Summary"),
                            ui.p(
                                f"• Climate alerts: {climate_summary.get('total_alerts', 0)} destinations with unusual weather"),
                            ui.p(f"• Meeting criteria: {climate_summary.get('meeting_criteria', 0)} destinations"),
                            ui.p(f"• Itineraries generated: {len(itineraries)} detailed weather-appropriate plans"),
                            class_="alert alert-info p-3"
                        )
                    )
            else:
                elements.append(
                    ui.div("No suitable recommendations found.", class_="alert alert-warning p-3")
                )

        # Show partial results if workflow is incomplete
        elif 'analyzer' in results:
            analyzer_data = results['analyzer']
            analyzed_destinations = analyzer_data.get('analyzed_destinations', [])

            if analyzed_destinations:
                elements.append(ui.h4("📊 Analysis Results (Partial)"))
                for dest in analyzed_destinations[:5]:
                    city_name = dest['city'].split(',')[0]
                    score = dest.get('overall_score', 0)
                    meets_criteria = dest.get('meets_criteria', False)

                    status = "✅" if meets_criteria else "⚠️"
                    elements.append(
                        ui.div(f"{status} {city_name}: Score {score:.1f}/10",
                               class_="alert alert-secondary mb-1 p-2")
                    )

        elif 'researcher' in results:
            research_data = results['researcher']
            flight_results = research_data.get('flight_results', [])

            if flight_results:
                elements.append(ui.h4("🔍 Research Results (Partial)"))
                elements.append(ui.p(f"Found {len(flight_results)} destinations with complete weather and flight data"))

        if not elements:
            elements.append(ui.div("Processing results...", class_="text-center p-3 text-muted"))

        return ui.div(*elements)

    @output
    @render.ui
    def status_display():
        status = workflow_status.get()
        alert_class = "alert-info" if "✅" in status else "alert-warning" if "⚠️" in status else "alert-danger"
        return ui.div(ui.p(status, class_="mb-0"), class_=f"alert {alert_class}")


app = App(app_ui, server)

if __name__ == "__main__":
    app.run(debug=True)