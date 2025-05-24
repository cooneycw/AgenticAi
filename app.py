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

    async def call_agent(self, role: AgentRole, parsed_query: TravelQuery, context: str = "") -> AgentMessage:
        """Enhanced autonomous agents with climate intelligence"""
        tools_used = []

        try:
            if role == AgentRole.PLANNER:
                # Planner creates a strategy based on the parsed query
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
                tools_used = ["strategic_analysis", "query_parsing", "climate_planning"]

            elif role == AgentRole.RESEARCHER:
                # Enhanced Researcher with climate analysis
                agent_data = "**ENHANCED RESEARCH IN PROGRESS:**\n\n"
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

                                        # Add traveler impact for significant deviations
                                        if severity in ['significant', 'extreme'] and deviation.get('traveler_impact'):
                                            agent_data += f"   💡 Travel Impact: {deviation['traveler_impact'][:80]}...\n"

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
                                else:
                                    agent_data += f"   📍 Route: {flight_info['origin']} → {city_name}\n"
                                    agent_data += f"   🕐 Distance: {flight_info['distance_km']:.0f}km\n"
                                    agent_data += f"   🛬 Airports: {flight_info.get('origin_airport', 'N/A')} → {flight_info.get('dest_airport', 'N/A')}\n"

                                # Display FR24 API status and data
                                api_status = flight_info.get('api_status', 'unknown')
                                agent_data += f"   📡 FR24 API Status: {api_status}\n"

                                if flight_info.get('fr24_data_available'):
                                    fr24_successful_calls += 1
                                    agent_data += f"   ✅ **REAL FLIGHT DATA RETRIEVED**\n"

                                    # Display raw FR24 API response summary
                                    fr24_raw = flight_info.get('fr24_raw_response', {})
                                    if fr24_raw.get('raw_data'):
                                        raw_data = fr24_raw['raw_data']
                                        result_count = len(raw_data.get('results', []))
                                        agent_data += f"   📊 FR24 Raw Results: {result_count} entries found\n"

                                        # Show sample of raw data structure
                                        if result_count > 0:
                                            sample_result = raw_data['results'][0] if raw_data.get('results') else {}
                                            agent_data += f"   📋 Sample API Response: {str(sample_result)[:100]}...\n"

                                    # Display parsed flight data
                                    parsed_data = flight_info.get('fr24_parsed_data', {})
                                    if parsed_data.get('parsed'):
                                        agent_data += f"   ✈️ Total Flights Found: {parsed_data.get('total_flights_found', 0)}\n"
                                        airlines = parsed_data.get('airlines', [])
                                        if airlines:
                                            agent_data += f"   🏢 Airlines: {', '.join(airlines[:3])}{'...' if len(airlines) > 3 else ''}\n"

                                    # Display recommended flights
                                    recommendations = flight_info.get('recommended_flights', [])
                                    if recommendations:
                                        agent_data += f"   🏆 **TOP FLIGHT RECOMMENDATIONS:**\n"
                                        for i, rec in enumerate(recommendations[:3], 1):
                                            if 'error' not in rec:
                                                agent_data += f"      {i}. {rec['airline']} {rec['flight_number']}\n"
                                                agent_data += f"         Duration: {rec['duration_display']}, Score: {rec['overall_score']:.1f}/10\n"
                                                agent_data += f"         Reliability: {rec['reliability_score']}/10, Aircraft: {rec['aircraft']}\n"
                                            else:
                                                agent_data += f"      ❌ Recommendation error: {rec['error']}\n"
                                    else:
                                        agent_data += f"   ⚠️ No flight recommendations generated\n"

                                    # Display schedule data if available
                                    schedule_data = flight_info.get('fr24_schedule_data')
                                    if schedule_data and schedule_data.get('success'):
                                        agent_data += f"   📅 Schedule data also retrieved from FR24\n"

                                else:
                                    fr24_failed_calls += 1
                                    if api_status == "no_fr24_key":
                                        agent_data += f"   ⚠️ Using geographic estimates (no FR24 API key)\n"
                                    elif "missing_airports" in api_status:
                                        agent_data += f"   ⚠️ Using geographic estimates (airport codes not found)\n"
                                    else:
                                        agent_data += f"   ❌ FR24 API failed, using geographic estimates\n"

                                    # Show fallback data
                                    if flight_info.get('routing'):
                                        agent_data += f"   📍 Estimated: {flight_info['routing']}\n"

                                agent_data += f"\n"
                            else:
                                agent_data += f"   ❌ Flight calculation failed: {flight_info['error'][:60]}\n"
                                fr24_failed_calls += 1
                        except Exception as e:
                            agent_data += f"   ❌ Error processing flights for {city_name}: {str(e)[:50]}\n"
                            fr24_failed_calls += 1
                            continue

                    # FR24 API Summary
                    agent_data += f"✅ Flight analysis complete for {len(flight_results)} destinations\n"
                    if self.flightradar24_key:
                        agent_data += f"📡 **FR24 API SUMMARY:**\n"
                        agent_data += f"   - Successful API calls: {fr24_successful_calls}\n"
                        agent_data += f"   - Failed/fallback calls: {fr24_failed_calls}\n"
                        agent_data += f"   - Real flight data coverage: {fr24_successful_calls}/{fr24_successful_calls + fr24_failed_calls} destinations\n"
                    else:
                        agent_data += f"📡 All flight data estimated (FR24 API not configured)\n"

                    tools_used.extend(["fr24_api_calls", "flight_recommendations", "airline_analysis"])

                    # Enhanced research summary with climate insights
                    agent_data += f"\n**ENHANCED RESEARCH SUMMARY:**\n"
                    agent_data += f"- Target date: {parsed_query.forecast_date_parsed.strftime('%A, %B %d, %Y')}\n"
                    agent_data += f"- Temperature criteria: {parsed_query.temperature_range[0]}-{parsed_query.temperature_range[1]}°C\n"
                    agent_data += f"- Origin: {parsed_query.origin_city}\n"
                    agent_data += f"- Complete data: {len(flight_results)} destinations\n"

                    if flight_results:
                        avg_temp = sum(d['temp_avg'] for d in flight_results) / len(flight_results)
                        avg_flight_time = sum(d.get('flight_time_hours', 8.0) for d in flight_results) / len(flight_results)

                        # Calculate how many destinations meet temperature criteria
                        meeting_temp_criteria = sum(1 for d in flight_results
                                                    if parsed_query.temperature_range[0] <= d['temp_avg'] <=
                                                    parsed_query.temperature_range[1])

                        agent_data += f"- Meeting temperature criteria: {meeting_temp_criteria}/{len(flight_results)}\n"
                        agent_data += f"- Average temperature: {avg_temp:.1f}°C\n"
                        agent_data += f"- Average flight time: {avg_flight_time:.1f} hours\n"

                        # Climate context summary
                        if climate_alerts:
                            extreme_count = sum(1 for alert in climate_alerts if alert['severity'] == 'extreme')
                            significant_count = sum(1 for alert in climate_alerts if alert['severity'] == 'significant')

                            agent_data += f"- Climate anomalies: {len(climate_alerts)} destinations with unusual weather\n"
                            if extreme_count > 0:
                                agent_data += f"  • {extreme_count} with extreme deviations from normal\n"
                            if significant_count > 0:
                                agent_data += f"  • {significant_count} with significant deviations from normal\n"
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
                # Enhanced Analyzer with climate deviation analysis
                metadata = {}

                # Get research data from previous agent
                research_messages = [msg for msg in self.conversation_history if msg.agent_role == AgentRole.RESEARCHER]

                if research_messages and research_messages[-1].metadata.get('research_success'):
                    research_data = research_messages[-1].metadata
                    flight_results = research_data.get('flight_results', [])
                    climate_alerts = research_data.get('climate_alerts', [])

                    if flight_results:
                        temp_min, temp_max = parsed_query.temperature_range

                        agent_data = "**ENHANCED ANALYSIS IN PROGRESS:**\n\n"

                        # Analyze each destination with climate context
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

                                # Enhanced scoring with climate factors and FR24 data
                                temp_target = (temp_min + temp_max) / 2
                                temp_score = 10 if meets_temp else max(0, 10 - abs(dest['temp_avg'] - temp_target) * 2)

                                # Enhanced flight scoring with FR24 data
                                flight_time_hours = dest.get('flight_time_hours', 8.0)  # Default fallback
                                base_flight_score = max(0, 10 - (flight_time_hours / 15))

                                # FR24 enhancement bonus
                                fr24_bonus = 0
                                flight_reliability_score = 5  # Default

                                if dest.get('fr24_data_available') and dest.get('recommended_flights'):
                                    fr24_bonus = 1  # Bonus for having real flight data
                                    # Use the best flight recommendation score
                                    best_flight = dest['recommended_flights'][0] if dest['recommended_flights'] else {}
                                    if 'overall_score' in best_flight:
                                        flight_reliability_score = best_flight['overall_score']

                                flight_score = min(10, base_flight_score + fr24_bonus)

                                humidity_score = 10 if 40 <= dest['humidity'] <= 70 else max(0, 10 - abs(
                                    dest['humidity'] - 55) / 5)

                                # NEW: Climate stability score
                                climate_score = 10  # Default for normal conditions
                                climate_bonus = 0
                                climate_penalty = 0

                                if 'climate_deviation' in dest:
                                    deviation = dest['climate_deviation']
                                    severity = deviation['severity']
                                    temp_dev = deviation['temp_deviation_c']

                                    if severity == 'extreme':
                                        climate_penalty = 3
                                        climate_score = 4
                                    elif severity == 'significant':
                                        climate_penalty = 1.5
                                        climate_score = 6
                                    elif severity == 'slight':
                                        climate_score = 8
                                    else:  # normal
                                        climate_bonus = 0.5  # Reward predictable weather

                                    # Special handling for favorable unusual weather
                                    if meets_temp and temp_dev > 0 and severity in ['slight', 'significant']:
                                        # Warmer than normal but still in range = bonus
                                        climate_bonus += 1
                                        climate_advantages.append({
                                            'city': dest['city'].split(',')[0],
                                            'advantage': f"Unusually warm (+{temp_dev:.1f}°C) but perfect for your criteria"
                                        })
                                    elif not meets_temp and severity in ['significant', 'extreme']:
                                        climate_concerns.append({
                                            'city': dest['city'].split(',')[0],
                                            'concern': deviation.get('traveler_impact', 'Unusual weather conditions'),
                                            'severity': severity
                                        })

                                # Enhanced overall score with climate weighting and FR24 data
                                overall_score = (
                                        (temp_score * 0.30) +
                                        (flight_score * 0.25) +
                                        (humidity_score * 0.15) +
                                        (climate_score * 0.20) +
                                        (flight_reliability_score * 0.10) +
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
                                    'overall_score': min(10, max(0, overall_score))  # Clamp to 0-10
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

                        # Climate intelligence summary
                        if climate_alerts:
                            agent_data += f"🌡️ **CLIMATE INTELLIGENCE:**\n"
                            extreme_alerts = [a for a in climate_alerts if a['severity'] == 'extreme']
                            significant_alerts = [a for a in climate_alerts if a['severity'] == 'significant']

                            if extreme_alerts:
                                agent_data += f"   🚨 {len(extreme_alerts)} destinations with extreme weather deviations\n"
                            if significant_alerts:
                                agent_data += f"   ⚠️ {len(significant_alerts)} destinations with significant weather deviations\n"

                            agent_data += f"   ✅ {len(analyzed_destinations) - len(climate_alerts)} destinations with normal seasonal weather\n\n"

                        if climate_advantages:
                            agent_data += f"🎯 **CLIMATE ADVANTAGES:**\n"
                            for adv in climate_advantages[:3]:
                                agent_data += f"   ✨ {adv['city']}: {adv['advantage']}\n"
                            agent_data += f"\n"

                        if climate_concerns:
                            agent_data += f"⚠️ **CLIMATE CONCERNS:**\n"
                            for concern in climate_concerns[:3]:
                                agent_data += f"   🌡️ {concern['city']}: {concern['concern']} ({concern['severity']})\n"
                            agent_data += f"\n"

                        if analyzed_destinations:
                            agent_data += "🏆 **TOP DESTINATIONS (Climate-Enhanced Scoring):**\n"
                            for i, dest in enumerate(analyzed_destinations[:5], 1):
                                city_name = dest['city'].split(',')[0]
                                status = "✅" if dest['meets_criteria'] else "⚠️"

                                # Climate indicator
                                climate_indicator = ""
                                if dest.get('climate_bonus', 0) > 0:
                                    climate_indicator = " 🌟"  # Climate bonus
                                elif dest.get('climate_penalty', 0) > 0:
                                    climate_indicator = " ⚡"  # Climate concern

                                agent_data += f"{i}. {status} **{city_name}**{climate_indicator} (Score: {dest['overall_score']:.1f}/10)\n"
                                agent_data += f"   🌡️ {dest['temp_avg']:.1f}°C | ✈️ {dest.get('flight_time_display', 'N/A')} | 💧 {dest['humidity']:.0f}%\n"
                                agent_data += f"   📍 {dest['weather_desc']} | 🛫 {dest['routing']}\n"

                                # Add climate context
                                if 'climate_deviation' in dest:
                                    deviation = dest['climate_deviation']
                                    if deviation['severity'] != 'normal':
                                        agent_data += f"   🌡️ Climate: {deviation['context'][:60]}...\n"

                                agent_data += f"\n"

                            # Enhanced AI insights with climate data
                            top_cities = [d['city'].split(',')[0] for d in analyzed_destinations[:3]]
                            climate_summary = {
                                'total_alerts': len(climate_alerts),
                                'climate_advantages': len(climate_advantages),
                                'climate_concerns': len(climate_concerns),
                                'avg_climate_score': sum(
                                    d.get('climate_score', 10) for d in analyzed_destinations[:3]) / 3
                            }

                            system_prompt = """You are an advanced travel analysis agent with climate intelligence. Provide insights on travel destinations considering both immediate travel factors and climate context.

Focus on:
- How climate deviations affect travel value
- Whether unusual weather is good or bad for the traveler's goals
- Optimal timing and backup planning
- Climate-aware travel recommendations

Be specific about the climate implications for this traveler."""

                            analysis_context = f"""
Enhanced Analysis Results:
- Query: {parsed_query.raw_query}
- Criteria: {temp_min}-{temp_max}°C on {parsed_query.forecast_date}
- Destinations meeting temperature: {meeting_criteria}/{len(analyzed_destinations)}
- Top 3 destinations: {', '.join(top_cities)}
- Climate alerts: {climate_summary['total_alerts']} total
- Climate advantages: {climate_summary['climate_advantages']} destinations
- Climate concerns: {climate_summary['climate_concerns']} destinations
- Average climate stability score: {climate_summary['avg_climate_score']:.1f}/10

Climate Context:
{climate_advantages[:2] if climate_advantages else 'No significant climate advantages'}
{climate_concerns[:2] if climate_concerns else 'No major climate concerns'}
"""

                            try:
                                response = self.client.chat.completions.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user",
                                         "content": f"Provide climate-enhanced analysis: {analysis_context}"}
                                    ],
                                    max_tokens=400,
                                    temperature=0.7
                                )

                                agent_data += "🧠 **CLIMATE-ENHANCED AI INSIGHTS:**\n"
                                agent_data += response.choices[0].message.content

                            except Exception as e:
                                agent_data += f"⚠️ AI climate analysis unavailable: {str(e)[:50]}"

                            metadata = {
                                "analyzed_destinations": analyzed_destinations,
                                "climate_summary": climate_summary,
                                "climate_advantages": climate_advantages,
                                "climate_concerns": climate_concerns
                            }
                            tools_used = ["enhanced_data_analysis", "climate_intelligence", "deviation_scoring",
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
                # Enhanced Synthesizer with climate-aware recommendations
                metadata = {}

                # Get analysis data
                analysis_messages = [msg for msg in self.conversation_history if msg.agent_role == AgentRole.ANALYZER]

                if analysis_messages and analysis_messages[-1].metadata.get('analyzed_destinations'):
                    destinations = analysis_messages[-1].metadata['analyzed_destinations']
                    climate_summary = analysis_messages[-1].metadata.get('climate_summary', {})
                    climate_advantages = analysis_messages[-1].metadata.get('climate_advantages', [])
                    climate_concerns = analysis_messages[-1].metadata.get('climate_concerns', [])

                    agent_data = "**CLIMATE-ENHANCED RECOMMENDATIONS:**\n\n"

                    if destinations:
                        top_destinations = destinations[:5]

                        # Prepare enhanced data for AI synthesis
                        summary_data = []
                        for i, dest in enumerate(top_destinations, 1):
                            city_name = dest['city'].split(',')[0]
                            status = "✅ MEETS CRITERIA" if dest['meets_criteria'] else "⚠️ CLOSE MATCH"

                            # Climate context
                            climate_note = ""
                            if 'climate_deviation' in dest:
                                deviation = dest['climate_deviation']
                                if deviation['severity'] in ['significant', 'extreme']:
                                    temp_dev = deviation['temp_deviation_c']
                                    if temp_dev > 0:
                                        climate_note = f" | 🔥 {temp_dev:+.1f}°C warmer than normal"
                                    else:
                                        climate_note = f" | 🧊 {temp_dev:+.1f}°C cooler than normal"
                                elif deviation['severity'] == 'slight':
                                    climate_note = " | 📊 Slightly off seasonal average"
                                else:
                                    climate_note = " | ✅ Typical seasonal weather"

                            summary_data.append(f"""
{i}. {city_name} - {status}{climate_note}
   Current Forecast: {dest['temp_avg']:.1f}°C (Target: {parsed_query.temperature_range[0]}-{parsed_query.temperature_range[1]}°C)
   Flight: {dest.get('flight_time_display', 'N/A')} from {parsed_query.origin_city}
   Weather: {dest['weather_desc']}, Humidity: {dest['humidity']:.0f}%
   Climate Score: {dest.get('climate_score', 10):.1f}/10 | Overall Score: {dest['overall_score']:.1f}/10
   Climate Context: {dest.get('climate_deviation', {}).get('context', 'Normal seasonal conditions')}
""")

                        destinations_summary = "\n".join(summary_data)

                        # Climate intelligence summary for AI
                        climate_context = f"""
Climate Intelligence Summary:
- Total destinations analyzed: {len(destinations)}
- Climate alerts: {climate_summary.get('total_alerts', 0)}
- Destinations with climate advantages: {len(climate_advantages)}
- Destinations with climate concerns: {len(climate_concerns)}
- Average climate stability: {climate_summary.get('avg_climate_score', 10):.1f}/10

Key Climate Insights:
{chr(10).join([f"+ {adv['city']}: {adv['advantage']}" for adv in climate_advantages[:2]])}
{chr(10).join([f"- {concern['city']}: {concern['concern']}" for concern in climate_concerns[:2]])}
"""

                        # Enhanced AI synthesis with climate intelligence
                        system_prompt = """You are an advanced travel recommendation agent with climate intelligence capabilities. Create personalized, climate-aware travel recommendations.

Provide:
1. Top 3 destinations with climate-enhanced reasoning
2. Climate timing strategy and booking advice
3. Weather contingency planning
4. Climate advantages and how to maximize them
5. Specific packing and preparation advice based on climate deviations

Be specific about climate benefits/risks and actionable about climate-related travel decisions."""

                        try:
                            response = self.client.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": f"""Create climate-enhanced travel recommendations:

Query: {parsed_query.raw_query}
Travel Date: {parsed_query.forecast_date}
Origin: {parsed_query.origin_city}

{climate_context}

Analyzed Destinations:
{destinations_summary}"""}
                                ],
                                max_tokens=700,
                                temperature=0.8
                            )

                            agent_data += response.choices[0].message.content

                        except Exception as e:
                            # Enhanced fallback with climate context
                            agent_data += "🏆 **CLIMATE-ENHANCED TOP RECOMMENDATIONS:**\n\n"

                            for i, dest in enumerate(top_destinations[:3], 1):
                                city_name = dest['city'].split(',')[0]
                                country_name = dest['city'].split(',')[1].strip() if ',' in dest['city'] else ""
                                status_emoji = "🎯" if dest['meets_criteria'] else "⭐"

                                # Climate emoji
                                climate_emoji = "🌟" if dest.get('climate_bonus', 0) > 0 else "⚡" if dest.get(
                                    'climate_penalty', 0) > 0 else "🌤️"

                                # FR24 emoji
                                fr24_emoji = "✈️" if dest.get('fr24_data_available') else "📍"

                                agent_data += f"**{i}. {status_emoji} {city_name}, {country_name}** {climate_emoji}{fr24_emoji}\n"
                                agent_data += f"   🌡️ Weather: {dest['temp_avg']:.1f}°C, {dest['weather_desc']}\n"
                                agent_data += f"   ✈️ Flight: {dest.get('flight_time_display', 'N/A')} ({dest.get('routing', 'N/A')})\n"

                                # Enhanced flight recommendations from FR24
                                if dest.get('fr24_data_available') and dest.get('recommended_flights'):
                                    agent_data += f"   🏆 **RECOMMENDED FLIGHTS (FR24 Data):**\n"
                                    for j, flight in enumerate(dest['recommended_flights'][:2], 1):
                                        if 'error' not in flight:
                                            agent_data += f"      {j}. {flight['airline']} {flight['flight_number']}\n"
                                            agent_data += f"         Duration: {flight['duration_display']}, Reliability: {flight['reliability_score']}/10\n"
                                            if flight.get('departure_time') and flight.get('arrival_time'):
                                                agent_data += f"         Times: {flight['departure_time']} → {flight['arrival_time']}\n"
                                elif dest.get('api_status'):
                                    agent_data += f"   📡 Flight Data: {dest['api_status'].replace('_', ' ').title()}\n"

                                # Climate-specific recommendation
                                if 'climate_deviation' in dest:
                                    deviation = dest['climate_deviation']
                                    agent_data += f"   🌡️ Climate: {deviation['context']}\n"
                                    if deviation.get('traveler_impact'):
                                        agent_data += f"   💡 Travel Tip: {deviation['traveler_impact']}\n"

                                agent_data += f"   📊 Overall Score: {dest['overall_score']:.1f}/10\n"
                                agent_data += f"      (Climate: {dest.get('climate_score', 10):.1f}/10, Flight: {dest.get('flight_score', 0):.1f}/10, FR24: {dest.get('flight_reliability_score', 0):.1f}/10)\n\n"

                        # Enhanced booking strategy with climate considerations
                        agent_data += f"\n\n🎯 **CLIMATE-SMART BOOKING STRATEGY:**\n"
                        agent_data += f"- **Target Date**: {parsed_query.forecast_date_parsed.strftime('%A, %B %d, %Y')}\n"
                        agent_data += f"- **Weather Criteria**: {parsed_query.temperature_range[0]}-{parsed_query.temperature_range[1]}°C\n"

                        if climate_summary.get('total_alerts', 0) > 0:
                            agent_data += f"- **Climate Alerts**: {climate_summary['total_alerts']} destinations with unusual weather\n"
                            agent_data += f"- **Recommendation**: Book flexible tickets, monitor weather 1 week before travel\n"
                        else:
                            agent_data += f"- **Climate Status**: Stable seasonal weather expected\n"
                            agent_data += f"- **Booking Confidence**: High - typical seasonal patterns predicted\n"

                        if climate_advantages:
                            agent_data += f"- **Climate Opportunity**: {len(climate_advantages)} destinations with favorable unusual weather\n"

                        agent_data += f"- **Departure**: {parsed_query.origin_city}\n"

                        if parsed_query.additional_criteria:
                            agent_data += f"- **Special Requirements**: {', '.join(parsed_query.additional_criteria)}\n"

                        tools_used = ["climate_enhanced_synthesis", "ai_climate_personalization",
                                      "weather_contingency_planning"]
                        metadata = {
                            "final_recommendations": top_destinations[:3],
                            "climate_summary": climate_summary,
                            "booking_date": parsed_query.forecast_date_parsed.strftime('%Y-%m-%d'),
                            "total_options": len(destinations),
                            "climate_enhanced": True
                        }
                    else:
                        agent_data = "❌ No suitable destinations found. Climate-enhanced suggestions:\n"
                        agent_data += "- Consider adjusting temperature range (current weather patterns unusual)\n"
                        agent_data += "- Try alternative dates (climate deviations may be temporary)\n"
                        agent_data += "- Expand regions (some areas may have better climate stability)\n"
                        tools_used = ["climate_aware_fallback"]
                        metadata = {}
                else:
                    agent_data = "❌ No analysis data available for climate-enhanced recommendations"
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

        # Synthesis (with climate-aware recommendations)
        synthesis_msg = await self.call_agent(AgentRole.SYNTHESIZER, parsed_query, analysis_msg.content)
        workflow_messages.append(synthesis_msg)

        return workflow_messages


# Initialize system
ai_system = AgenticAISystem(OPENAI_KEY, OPENWEATHER_KEY, FLIGHTRADAR24_KEY) if OPENAI_KEY else None

# Enhanced Shiny UI
app_ui = ui.page_fillable(
    ui.h1("🤖 Enhanced Agentic AI Travel System", class_="text-center mb-4"),
    ui.p("🌡️ Climate Intelligence • ✈️ Flight Analysis • 🎯 Smart Recommendations",
         class_="text-center text-muted mb-4"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Configuration"),
            ui.input_password("api_key", "OpenAI API Key:", placeholder="sk-..."),
            ui.input_action_button("connect", "Connect", class_="btn-primary mb-3"),
            ui.hr(),

            ui.h4("Enhanced Examples"),
            ui.input_action_button("example1", "🌍 Europe: Climate + Weather", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example2", "🏝️ Asia: Warm + Morning Flights",
                                   class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example3", "🌮 Mexico: Beach Weather", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example4", "❄️ Cooler Europe: Mild Weather", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example5", "🌡️ Climate Anomaly Test", class_="btn-outline-info btn-sm mb-1"),
            ui.input_action_button("example6", "✈️ FR24 Flight Data Test", class_="btn-outline-info btn-sm mb-1"),

            ui.hr(),
            ui.h5("Enhanced Features"),
            ui.p("✅ Historical Climate Baselines", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Weather Deviation Analysis", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Climate-Enhanced Scoring", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ FlightRadar24 Real Flight Data", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Flight Recommendations & Reliability", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Toronto Default Origin", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Multiple Real APIs", style="font-size: 0.85em; margin: 2px 0;"),

            width=300
        ),

        ui.div(
            ui.div(
                ui.input_text_area(
                    "user_query",
                    "Enter your enhanced travel query:",
                    placeholder="e.g., 'Find warm Asian destinations for next Friday with climate analysis' or 'European cities with mild weather in 3 weeks, check for unusual conditions'",
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
    """)
)


def server(input, output, session):
    workflow_status = reactive.Value(
        f"✅ Enhanced System Ready: OpenAI ✅, Weather API {'✅' if OPENWEATHER_KEY else '❌'}, FR24 API {'✅' if FLIGHTRADAR24_KEY else '❌'}, Climate Intelligence ✅" if ai_system
        else "⚠️ Please connect OpenAI API key for enhanced features"
    )
    conversation_messages = reactive.Value([])

    @reactive.Effect
    @reactive.event(input.example6)
    def load_example6():
        ui.update_text_area("user_query",
                            value="European destinations tomorrow with FlightRadar24 flight analysis, show real flight data and recommendations")

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
    @reactive.event(input.run_workflow)
    async def run_agentic_workflow():
        global ai_system

        if not ai_system:
            workflow_status.set("❌ Please connect to OpenAI API first")
            return

        if not input.user_query():
            workflow_status.set("⚠️ Please enter a query")
            return

        workflow_status.set("🔄 Enhanced AI agents working with climate intelligence...")
        conversation_messages.set([])

        try:
            messages = await ai_system.run_agentic_workflow(input.user_query())
            conversation_messages.set(messages)
            workflow_status.set("✅ Enhanced AI workflow completed with climate analysis")
        except Exception as e:
            workflow_status.set(f"❌ Enhanced workflow failed: {str(e)}")

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
                ui.h4("🌡️ Enhanced Climate Intelligence Ready"),
                ui.p("This enhanced system provides:", class_="mb-2"),
                ui.div(
                    ui.p("• Historical climate baselines for any city/month", class_="text-muted mb-1"),
                    ui.p("• Weather deviation analysis (normal → extreme)", class_="text-muted mb-1"),
                    ui.p("• Climate-enhanced destination scoring", class_="text-muted mb-1"),
                    ui.p("• Traveler impact insights for unusual weather", class_="text-muted mb-1"),
                    ui.p("• Climate-aware booking recommendations", class_="text-muted mb-1"),
                ),
                ui.p("Try any travel query with any region, date, or criteria!",
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