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
                model="gpt-4",
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
                model="gpt-4",
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


class AgenticAISystem:
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
                model="gpt-4",
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
                    model="gpt-4",
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

    async def call_flightradar24_api(self, origin_airport: str, dest_airport: str) -> Dict[str, Any]:
        """Call FlightRadar24 API for real flight data"""
        if not self.flightradar24_key:
            return {"error": "FlightRadar24 API key not available", "fallback": True}

        try:
            import aiohttp

            # FR24 API endpoint for route information
            url = f"https://api.flightradar24.com/common/v1/search.json"
            headers = {
                'Authorization': f'Bearer {self.flightradar24_key}',
                'User-Agent': 'AgenticAI-Travel/1.0'
            }

            # Search for flights between airports
            params = {
                'query': f"{origin_airport}-{dest_airport}",
                'limit': 10
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        fr24_data = await response.json()
                        return {
                            "success": True,
                            "raw_data": fr24_data,
                            "api_response_code": response.status
                        }
                    else:
                        return {
                            "error": f"FR24 API error: HTTP {response.status}",
                            "fallback": True,
                            "api_response_code": response.status
                        }

        except Exception as e:
            return {
                "error": f"FR24 API call failed: {str(e)}",
                "fallback": True
            }

    async def get_flight_schedules_fr24(self, origin_airport: str, dest_airport: str) -> Dict[str, Any]:
        """Get flight schedules from FlightRadar24 API"""
        if not self.flightradar24_key:
            return {"error": "FlightRadar24 API key not available", "fallback": True}

        try:
            import aiohttp

            # Alternative FR24 endpoint for schedules
            url = f"https://api.flightradar24.com/common/v1/airport.json"
            headers = {
                'Authorization': f'Bearer {self.flightradar24_key}',
                'User-Agent': 'AgenticAI-Travel/1.0'
            }

            params = {
                'code': origin_airport,
                'plugin[]': ['schedule', 'runways', 'arrivals', 'departures'],
                'page': 1,
                'limit': 25,
                'token': self.flightradar24_key
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        schedule_data = await response.json()
                        return {
                            "success": True,
                            "raw_schedule_data": schedule_data,
                            "api_response_code": response.status
                        }
                    else:
                        return {
                            "error": f"FR24 Schedule API error: HTTP {response.status}",
                            "fallback": True
                        }

        except Exception as e:
            return {
                "error": f"FR24 Schedule API failed: {str(e)}",
                "fallback": True
            }

    def parse_fr24_flight_data(self, fr24_response: Dict[str, Any], origin_airport: str, dest_airport: str) -> Dict[
        str, Any]:
        """Parse FlightRadar24 API response to extract useful flight information"""
        try:
            if not fr24_response.get("success"):
                return {"parsed": False, "error": "FR24 API call failed"}

            raw_data = fr24_response.get("raw_data", {})

            # Extract flight information from FR24 response
            flights = []
            airlines = set()
            total_flights = 0

            # FR24 API structure varies, adapt based on actual response
            if "results" in raw_data:
                for result in raw_data.get("results", [])[:10]:  # Limit to top 10
                    if result.get("type") == "flight":
                        flight_info = {
                            "flight_number": result.get("id", "Unknown"),
                            "airline": result.get("airline", {}).get("name", "Unknown"),
                            "aircraft": result.get("aircraft", {}).get("model", "Unknown"),
                            "duration": result.get("time", {}).get("scheduled", {}).get("duration", 0),
                            "departure_time": result.get("time", {}).get("scheduled", {}).get("departure"),
                            "arrival_time": result.get("time", {}).get("scheduled", {}).get("arrival"),
                        }
                        flights.append(flight_info)
                        airlines.add(flight_info["airline"])
                        total_flights += 1

            return {
                "parsed": True,
                "total_flights_found": total_flights,
                "airlines": list(airlines),
                "flight_details": flights,
                "route": f"{origin_airport} → {dest_airport}",
                "data_source": "FlightRadar24 API"
            }

        except Exception as e:
            return {
                "parsed": False,
                "error": f"Failed to parse FR24 data: {str(e)}"
            }

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

            # Initialize result with basic data
            result = {
                'distance_km': distance_km,
                'origin': origin_city,
                'destination': f"{dest_city}, {dest_country}",
                'origin_airport': origin_airport or "Unknown",
                'dest_airport': dest_airport or "Unknown",
                'fr24_data_available': False,
                'fr24_raw_response': None,
                'fr24_parsed_data': None,
                'recommended_flights': [],
                'api_status': "not_attempted"
            }

            # Try to get real flight data from FR24 API if we have airport codes
            if origin_airport and dest_airport and self.flightradar24_key:
                result['api_status'] = "attempting_fr24"

                # Call FR24 API for flight data
                fr24_response = await self.call_flightradar24_api(origin_airport, dest_airport)
                result['fr24_raw_response'] = fr24_response

                if fr24_response.get("success"):
                    result['fr24_data_available'] = True
                    result['api_status'] = "fr24_success"

                    # Parse the FR24 data
                    parsed_data = self.parse_fr24_flight_data(fr24_response, origin_airport, dest_airport)
                    result['fr24_parsed_data'] = parsed_data

                    if parsed_data.get("parsed"):
                        # Generate flight recommendations based on FR24 data
                        result['recommended_flights'] = self.generate_flight_recommendations(
                            parsed_data, distance_km
                        )
                else:
                    result['api_status'] = f"fr24_failed: {fr24_response.get('error', 'Unknown error')}"

                # Also try to get schedule data
                schedule_response = await self.get_flight_schedules_fr24(origin_airport, dest_airport)
                result['fr24_schedule_data'] = schedule_response

            else:
                if not self.flightradar24_key:
                    result['api_status'] = "no_fr24_key"
                elif not origin_airport or not dest_airport:
                    result['api_status'] = f"missing_airports: {origin_airport or 'origin'}, {dest_airport or 'dest'}"

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

            # Always set these fields regardless of FR24 status
            result.update({
                'flight_time_hours': flight_time,
                'flight_time_display': f"{hours}h {minutes}m",
                'routing': routing,
                'data_source': 'fr24_enhanced' if result['fr24_data_available'] else 'geographic_estimate'
            })

            # If FR24 data is available, potentially override with more accurate timing
            if result['fr24_data_available'] and result.get('recommended_flights'):
                # Use the best flight recommendation for timing if available
                best_flight = result['recommended_flights'][0] if result['recommended_flights'] else {}
                if best_flight.get('duration_hours') and best_flight['duration_hours'] > 0:
                    result.update({
                        'flight_time_hours': best_flight['duration_hours'],
                        'flight_time_display': best_flight['duration_display'],
                        'data_source': 'fr24_real_data'
                    })

            return result

        except Exception as e:
            return {"error": f"Enhanced flight calculation failed: {str(e)}"}

    def generate_flight_recommendations(self, parsed_data: Dict[str, Any], distance_km: float) -> List[Dict[str, Any]]:
        """Generate flight recommendations based on FR24 data"""
        try:
            recommendations = []

            if not parsed_data.get("parsed") or not parsed_data.get("flight_details"):
                return recommendations

            flights = parsed_data["flight_details"]

            # Sort flights by duration (shortest first) and add reliability scoring
            for flight in flights:
                duration_minutes = flight.get("duration", 0)
                if duration_minutes > 0:
                    duration_hours = duration_minutes / 60

                    # Calculate recommendation score (0-10)
                    # Factors: flight duration, airline reputation, frequency
                    duration_score = max(0, 10 - (duration_hours / 15))  # Shorter = better

                    # Simple airline reliability scoring (could be enhanced with real data)
                    airline = flight.get("airline", "").lower()
                    if any(carrier in airline for carrier in ["lufthansa", "swiss", "klm", "air canada"]):
                        reliability_score = 9
                    elif any(carrier in airline for carrier in ["british airways", "air france", "delta"]):
                        reliability_score = 8
                    elif any(carrier in airline for carrier in ["united", "american", "iberia"]):
                        reliability_score = 7
                    else:
                        reliability_score = 6

                    overall_score = (duration_score * 0.6) + (reliability_score * 0.4)

                    recommendations.append({
                        "flight_number": flight["flight_number"],
                        "airline": flight["airline"],
                        "aircraft": flight["aircraft"],
                        "duration_hours": duration_hours,
                        "duration_display": f"{int(duration_hours)}h {int((duration_hours % 1) * 60)}m",
                        "departure_time": flight.get("departure_time"),
                        "arrival_time": flight.get("arrival_time"),
                        "reliability_score": reliability_score,
                        "duration_score": duration_score,
                        "overall_score": overall_score,
                        "recommendation_rank": 0  # Will be set after sorting
                    })

            # Sort by overall score (highest first)
            recommendations.sort(key=lambda x: x["overall_score"], reverse=True)

            # Add ranking
            for i, rec in enumerate(recommendations):
                rec["recommendation_rank"] = i + 1

            return recommendations[:5]  # Return top 5 recommendations

        except Exception as e:
            return [{"error": f"Failed to generate recommendations: {str(e)}"}]

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
                model="gpt-4",
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

    async def call_agent(self, role: AgentRole, parsed_query: TravelQuery, context: str = "") -> AgentMessage:
        """Enhanced autonomous agents with climate intelligence"""
        tools_used = []
        metadata = {}  # Initialize metadata for all agents
        agent_data = ""  # Initialize agent_data

        try:
            if role == AgentRole.PLANNER:
                # Use safe temperature range access
                temp_min, temp_max = self.safe_get_temp_range(parsed_query)

                # Initialize metadata
                metadata = {}

                system_prompt = """You are a strategic planning agent for travel research with climate intelligence. 
Given a parsed travel query, create a specific action plan for gathering data.

Consider:
- What data sources are needed
- What order to gather information
- How to optimize the research process
- Climate analysis requirements
- Potential challenges and fallbacks

Be specific about the steps and reasoning."""

                query_summary = f"""
Query: {getattr(parsed_query, 'raw_query', 'Unknown query')}
Regions: {', '.join(getattr(parsed_query, 'regions', ['Europe']))}
Date: {getattr(parsed_query, 'forecast_date', 'tomorrow')} ({getattr(parsed_query, 'forecast_date_parsed', datetime.now()).strftime('%Y-%m-%d')})
Temperature: {temp_min}-{temp_max}°C
Origin: {getattr(parsed_query, 'origin_city', 'Toronto')}
Criteria: {', '.join(getattr(parsed_query, 'additional_criteria', []))}
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
                tools_used = ["strategic_analysis", "query_parsing", "climate_planning"]

            elif role == AgentRole.RESEARCHER:
                # Enhanced Researcher with climate analysis
                agent_data = "**ENHANCED RESEARCH IN PROGRESS:**\n\n"
                agent_data += f"📋 **Query Analysis:**\n"
                agent_data += f"   - Regions: {', '.join(parsed_query.regions)}\n"
                agent_data += f"   - Date: {parsed_query.forecast_date} ({parsed_query.forecast_date_parsed.strftime('%Y-%m-%d')})\n"

                temp_min, temp_max = self.safe_get_temp_range(parsed_query)
                agent_data += f"   - Temperature: {temp_min}-{temp_max}°C\n"
                agent_data += f"   - Origin: {parsed_query.origin_city}\n\n"
                metadata = {}

                try:
                    # Step 1: Get diverse cities for the specified regions (increased from 6 to 12)
                    cities = await self.get_cities_for_regions(parsed_query.regions, 12)
                    agent_data += f"✅ Generated {len(cities)} diverse cities across {len(parsed_query.regions)} regions\n"
                    tools_used.append("ai_city_generation")

                    # Step 2: Enhanced weather data with climate analysis
                    weather_results = []
                    climate_alerts = []
                    successful_forecasts = 0

                    agent_data += f"🌡️ **ENHANCED WEATHER ANALYSIS** for {parsed_query.forecast_date}...\n"
                    agent_data += f"   📊 Including historical climate baselines and deviation analysis\n"

                    # Check if date is too far in future
                    days_ahead = (parsed_query.forecast_date_parsed - datetime.now()).days
                    if days_ahead > 16:
                        agent_data += f"⚠️ Date is {days_ahead} days ahead - using seasonal estimates with historical context\n"
                    else:
                        agent_data += f"✅ Date is {days_ahead} days ahead - using real forecasts with climate analysis\n"

                    agent_data += f"\n"

                    for i, city_data in enumerate(cities):
                        try:
                            agent_data += f"   🔍 **{city_data['city']}, {city_data['country']}**\n"
                            coords = await self.get_city_coordinates(city_data['city'], city_data['country'])

                            if coords:
                                agent_data += f"   📍 Coordinates: {coords['lat']:.2f}, {coords['lon']:.2f}\n"

                                # Use enhanced weather forecast with climate analysis
                                weather_result = await self.get_weather_forecast_with_climate_analysis(
                                    coords['lat'], coords['lon'],
                                    f"{city_data['city']}, {city_data['country']}",
                                    parsed_query.forecast_date_parsed
                                )

                                if weather_result.success:
                                    weather_data = weather_result.data
                                    weather_results.append(weather_data)
                                    successful_forecasts += 1

                                    # Display enhanced weather information
                                    temp_avg = weather_data['temp_avg']
                                    weather_desc = weather_data['weather_desc']
                                    data_type = weather_data.get('data_type', 'forecast')
                                    type_icon = "🔮" if data_type == 'seasonal_estimate' else "📊" if data_type == 'forecast' else "📈"

                                    agent_data += f"   ✅ Current: {temp_avg:.1f}°C, {weather_desc} {type_icon}\n"

                                    # Add climate context if available
                                    if 'climate_deviation' in weather_data:
                                        deviation = weather_data['climate_deviation']
                                        baseline = weather_data.get('historical_baseline', {})

                                        historical_avg = baseline.get('avg_temp_mean', temp_avg)
                                        temp_deviation = deviation['temp_deviation_c']
                                        severity = deviation['severity']

                                        agent_data += f"   📊 Historical avg: {historical_avg:.1f}°C ({baseline.get('climate_zone', 'unknown')} climate)\n"

                                        if severity in ['significant', 'extreme']:
                                            if temp_deviation > 0:
                                                agent_data += f"   🔥 **WARMER than normal**: {temp_deviation:+.1f}°C ({severity})\n"
                                            else:
                                                agent_data += f"   🧊 **COOLER than normal**: {temp_deviation:+.1f}°C ({severity})\n"

                                            # Track climate alerts
                                            climate_alerts.append({
                                                'city': city_data['city'],
                                                'deviation': temp_deviation,
                                                'severity': severity,
                                                'context': deviation.get('context', '')
                                            })

                                        elif severity == 'slight':
                                            if temp_deviation > 0:
                                                agent_data += f"   ⬆️ Slightly warmer: {temp_deviation:+.1f}°C\n"
                                            else:
                                                agent_data += f"   ⬇️ Slightly cooler: {temp_deviation:+.1f}°C\n"
                                        else:
                                            agent_data += f"   ✅ Typical seasonal weather\n"

                                    agent_data += f"\n"
                                else:
                                    agent_data += f"   ❌ Weather forecast failed: {weather_result.error[:60]}\n"
                            else:
                                agent_data += f"   ❌ Could not get coordinates\n"

                        except Exception as e:
                            agent_data += f"   ❌ Error processing {city_data['city']}: {str(e)[:60]}\n"
                            continue

                    agent_data += f"✅ Enhanced weather analysis complete: {successful_forecasts}/{len(cities)} cities\n"

                    # Climate alerts summary
                    if climate_alerts:
                        agent_data += f"\n🚨 **CLIMATE ALERTS** ({len(climate_alerts)} destinations with unusual weather):\n"
                        for alert in climate_alerts:
                            deviation_type = "hotter" if alert['deviation'] > 0 else "colder"
                            agent_data += f"   ⚠️ **{alert['city']}**: {abs(alert['deviation']):.1f}°C {deviation_type} than normal ({alert['severity']})\n"
                    else:
                        agent_data += f"\n✅ All destinations showing typical seasonal weather patterns\n"

                    tools_used.extend(["enhanced_weather_api", "historical_climate_analysis", "deviation_detection"])

                    # Step 3: Enhanced Flight information with FR24 API
                    flight_results = []
                    agent_data += f"\n✈️ **ENHANCED FLIGHT ANALYSIS** from {parsed_query.origin_city}...\n"
                    if self.flightradar24_key:
                        agent_data += f"   📡 FlightRadar24 API: Connected ✅\n"
                    else:
                        agent_data += f"   📡 FlightRadar24 API: Not configured ⚠️ (using geographic estimates)\n"
                    agent_data += f"\n"

                    fr24_successful_calls = 0
                    fr24_failed_calls = 0

                    for weather_data in weather_results:
                        try:
                            city_name = weather_data['city'].split(',')[0].strip()
                            country_name = weather_data['city'].split(',')[1].strip() if ',' in weather_data[
                                'city'] else ""

                            agent_data += f"   🛫 **{city_name} Flight Analysis:**\n"

                            flight_info = await self.calculate_flight_info(parsed_query.origin_city, city_name,
                                                                           country_name)

                            if 'error' not in flight_info:
                                # Ensure flight_info has required fields before merging
                                if 'flight_time_hours' not in flight_info:
                                    # Add fallback flight time if missing
                                    distance_km = flight_info.get('distance_km', 5000)
                                    flight_time = distance_km / 800 + 1.0
                                    hours = int(flight_time)
                                    minutes = int((flight_time - hours) * 60)
                                    flight_info.update({
                                        'flight_time_hours': flight_time,
                                        'flight_time_display': f"{hours}h {minutes}m",
                                        'routing': 'Estimated',
                                        'data_source': 'fallback_estimate'
                                    })

                                combined_data = {**weather_data, **flight_info}
                                flight_results.append(combined_data)

                                # Display basic flight info
                                if flight_info.get('flight_time_display'):
                                    agent_data += f"   📍 Route: {flight_info['origin']} → {city_name}\n"
                                    agent_data += f"   🕐 Duration: {flight_info['flight_time_display']}, Distance: {flight_info['distance_km']:.0f}km\n"
                                    agent_data += f"   🛬 Airports: {flight_info.get('origin_airport', 'N/A')} → {flight_info.get('dest_airport', 'N/A')}\n"

                                # Display FR24 API status
                                api_status = flight_info.get('api_status', 'unknown')
                                agent_data += f"   📡 FR24 Status: {api_status}\n"

                                if flight_info.get('fr24_data_available'):
                                    fr24_successful_calls += 1
                                    agent_data += f"   ✅ **REAL FLIGHT DATA RETRIEVED**\n"
                                else:
                                    fr24_failed_calls += 1
                                    if api_status == "no_fr24_key":
                                        agent_data += f"   ⚠️ Using geographic estimates (no FR24 API key)\n"
                                    else:
                                        agent_data += f"   ❌ Using geographic estimates\n"

                                agent_data += f"\n"
                            else:
                                agent_data += f"   ❌ Flight calculation failed: {flight_info['error'][:60]}\n"
                                fr24_failed_calls += 1
                        except Exception as e:
                            agent_data += f"   ❌ Error processing flights for {city_name}: {str(e)[:50]}\n"
                            fr24_failed_calls += 1
                            continue

                    # Enhanced research summary with climate insights
                    agent_data += f"✅ Flight analysis complete for {len(flight_results)} destinations\n"
                    if self.flightradar24_key:
                        agent_data += f"📡 **FR24 API SUMMARY:**\n"
                        agent_data += f"   - Successful API calls: {fr24_successful_calls}\n"
                        agent_data += f"   - Failed/fallback calls: {fr24_failed_calls}\n"
                    else:
                        agent_data += f"📡 All flight data estimated (FR24 API not configured)\n"

                    agent_data += f"\n**ENHANCED RESEARCH SUMMARY:**\n"
                    agent_data += f"- Target date: {parsed_query.forecast_date_parsed.strftime('%A, %B %d, %Y')}\n"
                    agent_data += f"- Temperature criteria: {temp_min}-{temp_max}°C\n"
                    agent_data += f"- Origin: {parsed_query.origin_city}\n"
                    agent_data += f"- Complete data: {len(flight_results)} destinations\n"

                    if flight_results:
                        avg_temp = sum(d['temp_avg'] for d in flight_results) / len(flight_results)
                        avg_flight_time = sum(d.get('flight_time_hours', 8.0) for d in flight_results) / len(
                            flight_results)

                        # Calculate how many destinations meet temperature criteria
                        meeting_temp_criteria = sum(1 for d in flight_results
                                                    if temp_min <= d['temp_avg'] <= temp_max)

                        agent_data += f"- Meeting temperature criteria: {meeting_temp_criteria}/{len(flight_results)}\n"
                        agent_data += f"- Average temperature: {avg_temp:.1f}°C\n"
                        agent_data += f"- Average flight time: {avg_flight_time:.1f} hours\n"

                        # Climate context summary
                        if climate_alerts:
                            extreme_count = sum(1 for alert in climate_alerts if alert['severity'] == 'extreme')
                            significant_count = sum(1 for alert in climate_alerts if alert['severity'] == 'significant')

                            agent_data += f"- Climate anomalies: {len(climate_alerts)} destinations with unusual weather\n"
                        else:
                            agent_data += f"- Climate status: All destinations showing normal seasonal patterns\n"

                    # Store enhanced metadata
                    metadata = {
                        "weather_results": weather_results,
                        "flight_results": flight_results,
                        "climate_alerts": climate_alerts,
                        "parsed_query": asdict(parsed_query),
                        "research_success": True,
                        "climate_analysis_enabled": True,
                        "total_cities_processed": len(cities),
                        "successful_weather_calls": successful_forecasts,
                        "fr24_successful_calls": fr24_successful_calls,
                        "fr24_failed_calls": fr24_failed_calls
                    }

                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    agent_data += f"\n❌ Enhanced research failed: {str(e)}\n"
                    agent_data += f"📋 Error details: {error_details[-500:]}\n"  # Last 500 chars of traceback
                    metadata = {
                        "weather_results": [],
                        "flight_results": [],
                        "climate_alerts": [],
                        "parsed_query": asdict(parsed_query),
                        "research_success": False,
                        "error": str(e),
                        "error_details": error_details
                    }

            elif role == AgentRole.ANALYZER:
                # Enhanced Analyzer with climate deviation analysis and improved scoring
                metadata = {}

                # Get research data from previous agent
                research_messages = [msg for msg in self.conversation_history if msg.agent_role == AgentRole.RESEARCHER]

                if research_messages and research_messages[-1].metadata.get('research_success'):
                    research_data = research_messages[-1].metadata
                    flight_results = research_data.get('flight_results', [])
                    climate_alerts = research_data.get('climate_alerts', [])

                    if flight_results:
                        temp_min, temp_max = self.safe_get_temp_range(parsed_query)

                        agent_data = "**ENHANCED ANALYSIS IN PROGRESS:**\n\n"

                        # Analyze each destination with improved scoring
                        analyzed_destinations = []
                        meeting_criteria = 0
                        climate_advantages = []
                        climate_concerns = []

                        for dest in flight_results:
                            try:
                                # Check if temperature meets criteria
                                meets_temp = temp_min <= dest['temp_avg'] <= temp_max
                                if meets_temp:
                                    meeting_criteria += 1

                                # IMPROVED SCORING SYSTEM - addressing Dublin issue
                                temp_target = (temp_min + temp_max) / 2
                                temp_distance = abs(dest['temp_avg'] - temp_target)

                                # More nuanced temperature scoring
                                if meets_temp:
                                    # Perfect match gets 10, gradually decrease as we move away from center
                                    temp_range_size = temp_max - temp_min
                                    temp_score = 10 - (temp_distance / (temp_range_size / 2)) * 2
                                    temp_score = max(8, temp_score)  # Minimum of 8 if within range
                                else:
                                    # Outside range - score based on how close
                                    temp_score = max(0, 8 - temp_distance * 1.5)

                                # Enhanced flight scoring
                                flight_time_hours = dest.get('flight_time_hours', 8.0)
                                base_flight_score = max(0, 10 - (flight_time_hours / 12))  # More generous

                                # FR24 enhancement bonus
                                fr24_bonus = 1 if dest.get('fr24_data_available') else 0
                                flight_score = min(10, base_flight_score + fr24_bonus)

                                # More realistic humidity scoring
                                humidity_score = 10 if 30 <= dest['humidity'] <= 80 else max(6, 10 - abs(
                                    dest['humidity'] - 55) / 10)

                                # Climate stability score (less punitive)
                                climate_score = 10
                                climate_bonus = 0
                                climate_penalty = 0

                                if 'climate_deviation' in dest:
                                    deviation = dest['climate_deviation']
                                    severity = deviation['severity']
                                    temp_dev = deviation['temp_deviation_c']

                                    if severity == 'extreme':
                                        climate_penalty = 2  # Reduced from 3
                                        climate_score = 6  # Improved from 4
                                    elif severity == 'significant':
                                        climate_penalty = 1  # Reduced from 1.5
                                        climate_score = 7  # Improved from 6
                                    elif severity == 'slight':
                                        climate_score = 8.5  # Improved from 8
                                    else:  # normal
                                        climate_bonus = 0.3  # Slight reward for predictable weather

                                    # Special handling for favorable unusual weather
                                    if meets_temp and temp_dev > 0 and severity in ['slight', 'significant']:
                                        climate_bonus += 0.5
                                        climate_advantages.append({
                                            'city': dest['city'].split(',')[0],
                                            'advantage': f"Unusually warm (+{temp_dev:.1f}°C) but perfect for your criteria"
                                        })

                                # REBALANCED overall score - less weight on flight time, more on temperature match
                                flight_reliability_score = 7  # Default
                                if dest.get('recommended_flights'):
                                    best_flight = dest['recommended_flights'][0] if dest['recommended_flights'] else {}
                                    if 'overall_score' in best_flight:
                                        flight_reliability_score = best_flight['overall_score']

                                overall_score = (
                                        (temp_score * 0.40) +  # Increased from 0.30
                                        (flight_score * 0.20) +  # Decreased from 0.25
                                        (humidity_score * 0.10) +  # Decreased from 0.15
                                        (climate_score * 0.20) +  # Same
                                        (flight_reliability_score * 0.10) +  # Same
                                        climate_bonus - climate_penalty
                                )

                                analyzed_destinations.append({
                                    **dest,
                                    'meets_criteria': meets_temp,
                                    'temp_score': temp_score,
                                    'flight_score': flight_score,
                                    'humidity_score': humidity_score,
                                    'climate_score': climate_score,
                                    'flight_reliability_score': flight_reliability_score,
                                    'fr24_bonus': fr24_bonus,
                                    'climate_bonus': climate_bonus,
                                    'climate_penalty': climate_penalty,
                                    'overall_score': min(10, max(0, overall_score))
                                })

                            except Exception as e:
                                continue

                        # Sort by overall score
                        analyzed_destinations.sort(key=lambda x: x['overall_score'], reverse=True)

                        agent_data += f"📊 **ENHANCED ANALYSIS RESULTS:**\n"
                        agent_data += f"- Temperature criteria: {temp_min}-{temp_max}°C\n"
                        agent_data += f"- Destinations meeting criteria: {meeting_criteria}/{len(analyzed_destinations)}\n"
                        agent_data += f"- Climate alerts: {len(climate_alerts)} destinations with unusual weather\n"
                        agent_data += f"- Analysis date: {parsed_query.forecast_date} ({parsed_query.forecast_date_parsed.strftime('%Y-%m-%d')})\n\n"

                        if analyzed_destinations:
                            agent_data += "🏆 **TOP DESTINATIONS (Enhanced Climate-Aware Scoring):**\n"
                            for i, dest in enumerate(analyzed_destinations[:8], 1):  # Show top 8 instead of 5
                                city_name = dest['city'].split(',')[0]
                                status = "✅" if dest['meets_criteria'] else "⚠️"

                                # Enhanced status indicators
                                indicators = []
                                if dest.get('climate_bonus', 0) > 0:
                                    indicators.append("🌟")  # Climate bonus
                                if dest.get('climate_penalty', 0) > 0:
                                    indicators.append("⚡")  # Climate concern
                                if dest.get('fr24_data_available'):
                                    indicators.append("✈️")  # Real flight data

                                indicator_str = "".join(indicators)

                                agent_data += f"{i}. {status} **{city_name}**{indicator_str} (Score: {dest['overall_score']:.1f}/10)\n"
                                agent_data += f"   🌡️ {dest['temp_avg']:.1f}°C | ✈️ {dest.get('flight_time_display', 'N/A')} | 💧 {dest['humidity']:.0f}%\n"
                                agent_data += f"   📍 {dest['weather_desc']} | 🛫 {dest.get('routing', 'N/A')}\n"

                                # Detailed scoring breakdown for top 3
                                if i <= 3:
                                    agent_data += f"   📊 Scores: Temp={dest['temp_score']:.1f} Flight={dest['flight_score']:.1f} Climate={dest['climate_score']:.1f}\n"

                                agent_data += f"\n"

                            metadata = {
                                "analyzed_destinations": analyzed_destinations,
                                "climate_summary": {
                                    'total_alerts': len(climate_alerts),
                                    'climate_advantages': len(climate_advantages),
                                    'climate_concerns': len(climate_concerns),
                                    'meeting_criteria': meeting_criteria
                                },
                                "climate_advantages": climate_advantages,
                                "climate_concerns": climate_concerns
                            }
                            tools_used = ["enhanced_data_analysis", "climate_intelligence", "improved_scoring",
                                          "ai_climate_insights"]
                        else:
                            agent_data += "❌ No valid destinations to analyze"
                            metadata = {}
                            tools_used = ["data_analysis"]
                    else:
                        agent_data = "❌ No flight/weather data available from research phase"
                        metadata = {}
                        tools_used = []
                else:
                    agent_data = "❌ Research phase failed or no data available"
                    metadata = {}
                    tools_used = []

            elif role == AgentRole.SYNTHESIZER:
                # Enhanced Synthesizer with itinerary generation
                metadata = {}

                # Get analysis data
                analysis_messages = [msg for msg in self.conversation_history if msg.agent_role == AgentRole.ANALYZER]

                if analysis_messages and analysis_messages[-1].metadata.get('analyzed_destinations'):
                    destinations = analysis_messages[-1].metadata['analyzed_destinations']
                    climate_summary = analysis_messages[-1].metadata.get('climate_summary', {})

                    agent_data = "**CLIMATE-ENHANCED RECOMMENDATIONS & ITINERARIES:**\n\n"

                    if destinations:
                        top_destinations = destinations[:3]

                        # Enhanced recommendations with itineraries for top 2
                        agent_data += "🏆 **TOP RECOMMENDATIONS:**\n\n"

                        for i, dest in enumerate(top_destinations, 1):
                            city_name = dest['city'].split(',')[0]
                            country_name = dest['city'].split(',')[1].strip() if ',' in dest['city'] else ""
                            status = "🎯" if dest['meets_criteria'] else "⭐"

                            # Enhanced indicators
                            indicators = []
                            if dest.get('climate_bonus', 0) > 0:
                                indicators.append("🌟")
                            if dest.get('fr24_data_available'):
                                indicators.append("✈️")

                            indicator_str = " ".join(indicators)

                            agent_data += f"**{i}. {status} {city_name}, {country_name}** {indicator_str}\n"
                            agent_data += f"   🌡️ Weather: {dest['temp_avg']:.1f}°C, {dest['weather_desc']}\n"
                            agent_data += f"   ✈️ Flight: {dest.get('flight_time_display', 'N/A')} ({dest.get('routing', 'N/A')})\n"
                            agent_data += f"   📊 Overall Score: {dest['overall_score']:.1f}/10\n"

                            # Climate context
                            if 'climate_deviation' in dest:
                                deviation = dest['climate_deviation']
                                if deviation['severity'] != 'normal':
                                    agent_data += f"   🌡️ Climate Note: {deviation['context'][:80]}...\n"

                            # Generate 4-day itinerary for top 2 destinations
                            if i <= 2:
                                agent_data += f"\n   🗓️ **4-DAY WEATHER-APPROPRIATE ITINERARY:**\n"
                                try:
                                    itinerary = await self.generate_weather_appropriate_itinerary(dest, 4)
                                    # Indent the itinerary
                                    indented_itinerary = '\n'.join(f"   {line}" for line in itinerary.split('\n'))
                                    agent_data += f"{indented_itinerary}\n"
                                except Exception as e:
                                    agent_data += f"   ❌ Itinerary generation failed: {str(e)[:100]}\n"

                            agent_data += f"\n"

                        # Enhanced booking strategy
                        agent_data += f"🎯 **CLIMATE-SMART BOOKING STRATEGY:**\n\n"
                        agent_data += f"**Target Details:**\n"
                        temp_min, temp_max = self.safe_get_temp_range(parsed_query)
                        agent_data += f"- Date: {parsed_query.forecast_date_parsed.strftime('%A, %B %d, %Y')}\n"
                        agent_data += f"- Temperature Range: {temp_min}-{temp_max}°C\n"
                        agent_data += f"- Departure: {parsed_query.origin_city}\n"

                        meeting_criteria = climate_summary.get('meeting_criteria', 0)
                        total_analyzed = len(destinations)

                        agent_data += f"\n**Analysis Summary:**\n"
                        agent_data += f"- Destinations analyzed: {total_analyzed}\n"
                        agent_data += f"- Meeting temperature criteria: {meeting_criteria}/{total_analyzed}\n"

                        if climate_summary.get('total_alerts', 0) > 0:
                            agent_data += f"- Climate alerts: {climate_summary['total_alerts']} destinations with unusual weather\n"
                            agent_data += f"- **Booking Tip**: Consider flexible tickets, monitor weather 1 week before travel\n"
                        else:
                            agent_data += f"- Climate status: Stable seasonal weather expected\n"
                            agent_data += f"- **Booking Confidence**: High - typical seasonal patterns predicted\n"

                        if parsed_query.additional_criteria:
                            agent_data += f"- Special requirements: {', '.join(parsed_query.additional_criteria)}\n"

                        # Why these destinations ranked well
                        agent_data += f"\n**Why These Destinations Ranked Well:**\n"
                        for i, dest in enumerate(top_destinations[:2], 1):
                            city_name = dest['city'].split(',')[0]
                            reasons = []

                            if dest['meets_criteria']:
                                temp_min, temp_max = self.safe_get_temp_range(parsed_query)
                                reasons.append(
                                    f"Perfect temperature match ({dest['temp_avg']:.1f}°C within {temp_min}-{temp_max}°C)")

                            if dest.get('fr24_data_available'):
                                reasons.append("Real flight data available")

                            if dest.get('climate_bonus', 0) > 0:
                                reasons.append("Favorable climate conditions")

                            if dest.get('flight_time_hours', 0) < 10:
                                reasons.append("Reasonable flight time")

                            agent_data += f"{i}. **{city_name}**: {'; '.join(reasons)}\n"

                        tools_used = ["climate_enhanced_synthesis", "weather_appropriate_itineraries",
                                      "enhanced_recommendations"]
                        metadata = {
                            "final_recommendations": top_destinations[:3],
                            "climate_summary": climate_summary,
                            "booking_date": parsed_query.forecast_date_parsed.strftime('%Y-%m-%d'),
                            "total_options": len(destinations),
                            "climate_enhanced": True,
                            "itineraries_generated": 2
                        }
                    else:
                        agent_data = "❌ No suitable destinations found. Enhanced suggestions:\n"
                        agent_data += "- Consider adjusting temperature range (current weather patterns may be unusual)\n"
                        agent_data += "- Try alternative dates (climate deviations may be temporary)\n"
                        agent_data += "- Expand regions (some areas may have better climate stability)\n"
                        tools_used = ["climate_aware_fallback"]
                        metadata = {}
                else:
                    agent_data = "❌ No analysis data available for enhanced recommendations"
                    tools_used = ["error_handling"]
                    metadata = {}

            message = AgentMessage(
                agent_role=role,
                content=agent_data,
                timestamp=datetime.now(),
                tools_used=tools_used,
                metadata=metadata
            )

            self.conversation_history.append(message)
            return message

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Agent {role.value} error: {error_details}")

            error_message = AgentMessage(
                agent_role=role,
                content=f"❌ Agent {role.value} encountered error: {str(e)}\n\nDebug info:\n{error_details[-300:]}",
                timestamp=datetime.now(),
                tools_used=[],
                metadata={"error": True, "error_details": error_details}
            )
            self.conversation_history.append(error_message)
            return error_message

    async def run_agentic_workflow(self, user_query: str) -> List[AgentMessage]:
        """Run the complete enhanced agentic workflow with climate intelligence"""

        # Clear previous conversation
        self.conversation_history = []

        # Step 1: Parse the query intelligently
        parsed_query = await self.parse_travel_query(user_query)

        # Step 2: Run enhanced agent workflow
        workflow_messages = []

        # Planning
        plan_msg = await self.call_agent(AgentRole.PLANNER, parsed_query)
        workflow_messages.append(plan_msg)

        # Research (with climate analysis)
        research_msg = await self.call_agent(AgentRole.RESEARCHER, parsed_query, plan_msg.content)
        workflow_messages.append(research_msg)

        # Analysis (with climate intelligence)
        analysis_msg = await self.call_agent(AgentRole.ANALYZER, parsed_query, research_msg.content)
        workflow_messages.append(analysis_msg)

        # Synthesis (with climate-aware recommendations and itineraries)
        synthesis_msg = await self.call_agent(AgentRole.SYNTHESIZER, parsed_query, analysis_msg.content)
        workflow_messages.append(synthesis_msg)

        return workflow_messages


# Initialize system
ai_system = AgenticAISystem(OPENAI_KEY, OPENWEATHER_KEY, FLIGHTRADAR24_KEY) if OPENAI_KEY else None

# Enhanced Shiny UI
app_ui = ui.page_fillable(
    ui.h1("🤖 Enhanced Agentic AI Travel System", class_="text-center mb-4"),
    ui.p(
        "🌡️ Climate Intelligence • ✈️ Flight Analysis • 🎯 Smart Recommendations • 🗓️ Weather-Appropriate Itineraries",
        class_="text-center text-muted mb-4"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Configuration"),
            ui.input_password("api_key", "OpenAI API Key:", placeholder="sk-..."),
            ui.input_action_button("connect", "Connect", class_="btn-primary mb-3"),
            ui.hr(),

            ui.h4("Enhanced Examples"),
            ui.input_action_button("example1", "🌍 Europe: Climate Analysis", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example2", "🏝️ Asia: Warm + Morning Flights",
                                   class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example3", "🌮 Mexico: Beach Weather", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example4", "❄️ Dublin Test: Cooler Weather", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example5", "🌡️ Climate Anomaly Test", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example6", "✈️ FR24 Flight Data Test", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example7", "🗺️ Diverse Cities Test", class_="btn-outline-info btn-sm mb-1"),

            ui.hr(),
            ui.h5("Enhanced Features"),
            ui.p("✅ Historical Climate Baselines", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Weather Deviation Analysis", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Improved Scoring Algorithm", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Diverse City Generation", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ 4-Day Weather Itineraries", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ FlightRadar24 Real Data", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Enhanced Error Handling", style="font-size: 0.85em; margin: 2px 0;"),

            width=300
        ),

        ui.div(
            ui.div(
                ui.input_text_area(
                    "user_query",
                    "Enter your enhanced travel query:",
                    placeholder="e.g., 'Find cooler European destinations with 5-15°C in 10 days from Toronto' or 'Diverse Asian cities with warm weather next Friday, include itineraries'",
                    rows=3,
                    width="100%"
                ),
                ui.input_action_button("run_workflow", "🚀 Run Enhanced AI Agents", class_="btn-success mb-4"),
                class_="mb-4"
            ),

            ui.div(
                ui.output_ui("status_display"),
                class_="mb-3"
            ),

            ui.div(
                ui.h3("Enhanced AI Agent Workflow"),
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
        .itinerary-section {
            background: rgba(255,255,255,0.8);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #17a2b8;
        }
    """)
)


def server(input, output, session):
    workflow_status = reactive.Value(
        f"✅ Enhanced System Ready: OpenAI ✅, Weather API {'✅' if OPENWEATHER_KEY else '❌'}, FR24 API {'✅' if FLIGHTRADAR24_KEY else '❌'}, Climate Intelligence ✅" if ai_system
        else "⚠️ Please connect OpenAI API key for enhanced features"
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
                workflow_status.set(
                    f"✅ Enhanced System Connected | Weather API {'✅' if OPENWEATHER_KEY else '❌'} | FR24 API {'✅' if FLIGHTRADAR24_KEY else '❌'} | Climate Intelligence ✅")
            except Exception as e:
                workflow_status.set(f"❌ Connection failed: {str(e)}")
        else:
            workflow_status.set("⚠️ Please enter OpenAI API key")

    @reactive.Effect
    @reactive.event(input.example1)
    def load_example1():
        ui.update_text_area("user_query",
                            value="Find European destinations with temperatures between 18-25°C in 1 week, include climate analysis and deviation alerts")

    @reactive.Effect
    @reactive.event(input.example2)
    def load_example2():
        ui.update_text_area("user_query",
                            value="Asian cities with warm weather next Friday, morning flights from Toronto, check for unusual climate conditions")

    @reactive.Effect
    @reactive.event(input.example3)
    def load_example3():
        ui.update_text_area("user_query",
                            value="Mexican coastal destinations with 25-30°C weather in 5 days, analyze climate patterns vs historical norms")

    @reactive.Effect
    @reactive.event(input.example4)
    def load_example4():
        ui.update_text_area("user_query",
                            value="Find cooler European destinations with 5-15°C in 10 days, Toronto departure, flag any weather anomalies")

    @reactive.Effect
    @reactive.event(input.example5)
    def load_example5():
        ui.update_text_area("user_query",
                            value="South American cities next month, check for climate deviations from seasonal norms, warm weather preferred")

    @reactive.Effect
    @reactive.event(input.example6)
    def load_example6():
        ui.update_text_area("user_query",
                            value="European destinations tomorrow with FlightRadar24 flight analysis, show real flight data and recommendations")

    @reactive.Effect
    @reactive.event(input.example7)
    def load_example7():
        ui.update_text_area("user_query",
                            value="Diverse European cities in 2 weeks, mix of famous and hidden gems, include 4-day weather-appropriate itineraries")

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

        workflow_status.set("🔄 Enhanced AI agents working with climate intelligence & itineraries...")
        conversation_messages.set([])

        try:
            messages = await ai_system.run_agentic_workflow(input.user_query())
            conversation_messages.set(messages)
            workflow_status.set("✅ Enhanced AI workflow completed with climate analysis & itineraries")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            workflow_status.set(f"❌ Enhanced workflow failed: {str(e)}")
            print(f"Workflow error details: {error_details}")

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
                ui.h4("🌡️ Enhanced Climate Intelligence & Itinerary System Ready"),
                ui.p("This enhanced system provides:", class_="mb-2"),
                ui.div(
                    ui.p("• Historical climate baselines for any city/month", class_="text-muted mb-1"),
                    ui.p("• Weather deviation analysis with traveler impact", class_="text-muted mb-1"),
                    ui.p("• Improved scoring algorithm (addresses Dublin ranking)", class_="text-muted mb-1"),
                    ui.p("• Diverse city generation (famous + hidden gems)", class_="text-muted mb-1"),
                    ui.p("• 4-day weather-appropriate itineraries", class_="text-muted mb-1"),
                    ui.p("• Enhanced error handling & debugging", class_="text-muted mb-1"),
                ),
                ui.p(
                    "Try any travel query - the system now generates diverse destinations and weather-specific itineraries!",
                    class_="text-primary text-center mt-3"),
                style="padding: 40px;"
            )

        agent_info = {
            AgentRole.PLANNER: {"icon": "🎯", "name": "Strategic Planner", "class": "agent-planner"},
            AgentRole.RESEARCHER: {"icon": "🔍", "name": "Enhanced Data Researcher", "class": "agent-researcher"},
            AgentRole.ANALYZER: {"icon": "📊", "name": "Climate Intelligence Analyzer", "class": "agent-analyzer"},
            AgentRole.SYNTHESIZER: {"icon": "🎯", "name": "Climate-Aware Synthesizer", "class": "agent-synthesizer"}
        }

        elements = []
        for msg in messages:
            info = agent_info[msg.agent_role]

            tools_display = ""
            if msg.tools_used:
                tools_display = f"Tools: {', '.join(msg.tools_used)}"

            # Enhanced content formatting for itineraries
            content = msg.content
            if "**DAY" in content:
                # Format itinerary sections with better styling
                content = content.replace("**DAY", "\n🗓️ **DAY")
                content = content.replace("- Morning", "\n   🌅 Morning")
                content = content.replace("- Afternoon", "\n   ☀️ Afternoon")
                content = content.replace("- Evening", "\n   🌆 Evening")
                content = content.replace("- Weather Gear:", "\n   🎒 Weather Gear:")
                content = content.replace("- Backup Plan:", "\n   🏠 Backup Plan:")

            elements.append(
                ui.div(
                    ui.div(
                        ui.span(f"{info['icon']} {info['name']}", class_="agent-name"),
                        ui.span(tools_display, class_="tools-used") if tools_display else "",
                        class_="agent-header"
                    ),
                    ui.div(content, style="white-space: pre-wrap; line-height: 1.6;"),
                    ui.div(f"⏱️ {msg.timestamp.strftime('%H:%M:%S')}", class_="timestamp mt-2"),
                    class_=f"agent-message {info['class']}"
                )
            )

        return ui.div(*elements)


app = App(app_ui, server)

if __name__ == "__main__":
    app.run(debug=True)