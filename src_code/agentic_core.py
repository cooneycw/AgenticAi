# src_code/agentic_core.py
"""Fixed Agentic Core with Progress Tracking - Phase 2.5

This version fixes critical API issues and adds real-time progress tracking:

1. **Fixed Weather API Issues**: Better URL construction and error handling
2. **Fixed JSON Parsing**: Robust city generation with fallbacks
3. **Real-Time Progress**: Status updates throughout workflow
4. **Enhanced Error Recovery**: Better fallback mechanisms
5. **Progress Callbacks**: UI can track detailed progress

Key fixes:
- Weather API 400 errors resolved with proper URL encoding
- City generation JSON parsing with graceful fallbacks
- Progress tracking for UI status updates
- Enhanced error logging and recovery
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from urllib.parse import quote

import aiohttp
from dateutil.relativedelta import relativedelta
from openai import OpenAI

# Local shared state - now with enhanced models
from config.config import (
    AgentMessage,
    AgentRole,
    ClimateAlert,
    ClimateBaseline,
    TaskResult,
    TravelQuery,
    WeatherDeviation,
    OPENAI_KEY,
    OPENWEATHER_KEY,
    FLIGHTRADAR24_KEY,
    DEVIATION_THRESHOLDS,
)
from src_code.climate_analyzer import EnhancedClimateAnalyzer
from src_code.itinerary_generator import WeatherAppropriateItineraryGenerator
from src_code.flight_analyzer import EnhancedFlightAnalyzer
from src_code.weather_api import get_weather_data_enhanced

__all__ = ["AgenticAISystem"]


# ---------------------------------------------------------------------------
# Progress tracking types
# ---------------------------------------------------------------------------

class ProgressStep:
    """Represents a single progress step."""

    def __init__(self, agent: str, step: str, current: int = 0, total: int = 0, details: str = ""):
        self.agent = agent
        self.step = step
        self.current = current
        self.total = total
        self.details = details

    def __str__(self) -> str:
        if self.total > 0:
            return f"🔄 {self.agent}: {self.step} ({self.current}/{self.total}) - {self.details}"
        else:
            return f"🔄 {self.agent}: {self.step} - {self.details}"


# ---------------------------------------------------------------------------
# Enhanced utility helpers
# ---------------------------------------------------------------------------

def _today() -> datetime:
    """Timezone-agnostic *now* for deterministic unit tests."""
    return datetime.now()


def _nl_date_to_dt(expr: str) -> datetime:
    """Enhanced natural-language → datetime helper with better parsing."""
    expr = expr.lower().strip()
    now = _today()

    # Direct matches
    if expr in {"today"}:
        return now
    if expr in {"tomorrow"}:
        return now + timedelta(days=1)
    if "next week" in expr:
        return now + timedelta(weeks=1)
    if "next month" in expr:
        return now + relativedelta(months=1)

    # Enhanced weekday parsing
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    for w, idx in weekdays.items():
        if w in expr:
            delta = (idx - now.weekday()) % 7
            delta = 7 if delta == 0 else delta  # always *next*
            return now + timedelta(days=delta)

    # Enhanced relative time parsing
    patterns = [
        (r"in (\d+) days?", lambda m: now + timedelta(days=int(m.group(1)))),
        (r"in (\d+) weeks?", lambda m: now + timedelta(weeks=int(m.group(1)))),
        (r"in (\d+) months?", lambda m: now + relativedelta(months=int(m.group(1)))),
        (r"(\d+) days? from now", lambda m: now + timedelta(days=int(m.group(1)))),
        (r"(\d+) weeks? from now", lambda m: now + timedelta(weeks=int(m.group(1)))),
    ]

    for pattern, func in patterns:
        match = re.search(pattern, expr)
        if match:
            return func(match)

    # Month name parsing (e.g., "in January", "next March")
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }
    for month_name, month_num in months.items():
        if month_name in expr:
            target_date = now.replace(month=month_num, day=15)
            if target_date <= now:  # Next year
                target_date = target_date.replace(year=now.year + 1)
            return target_date

    # Default fallback
    return now + timedelta(days=1)


# ---------------------------------------------------------------------------
# Enhanced core system class with progress tracking
# ---------------------------------------------------------------------------

class AgenticAISystem:
    """Enhanced orchestrator with progress tracking and fixed API issues."""

    def __init__(
            self,
            openai_key: Optional[str] = None,
            openweather_key: Optional[str] = None,
            flightradar24_key: Optional[str] = None,
    ) -> None:
        if openai_key is None:
            raise RuntimeError("OpenAI key required for AgenticAISystem")
        self.client = OpenAI(api_key=openai_key)
        self.openweather_key = openweather_key
        self.fr24_key = flightradar24_key

        self.conversation_history: List[AgentMessage] = []

        # Enhanced climate analyzer with rich functionality
        self.climate = EnhancedClimateAnalyzer(self.client)

        # Weather-appropriate itinerary generator
        self.itinerary_generator = WeatherAppropriateItineraryGenerator(self.client)

        # Enhanced flight analyzer with real-time data
        self.flight_analyzer = EnhancedFlightAnalyzer(flightradar24_key)

        # Track climate alerts for this session
        self.climate_alerts: List[ClimateAlert] = []

        # Progress tracking
        self.progress_callback: Optional[Callable[[ProgressStep], None]] = None
        self.current_progress: Optional[ProgressStep] = None

    def set_progress_callback(self, callback: Callable[[ProgressStep], None]) -> None:
        """Set callback function for progress updates."""
        self.progress_callback = callback

    def _update_progress(self, agent: str, step: str, current: int = 0, total: int = 0, details: str = "") -> None:
        """Update progress and notify callback."""
        self.current_progress = ProgressStep(agent, step, current, total, details)
        if self.progress_callback:
            self.progress_callback(self.current_progress)

    # ------------------------------------------------------------------
    # Enhanced helper methods with better error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_temp_range(q: TravelQuery) -> Tuple[float, float]:
        """Enhanced temperature range extraction with better validation."""
        rng = q.temperature_range or (20.0, 27.0)
        if len(rng) < 2:
            return (20.0, 27.0)
        lo, hi = sorted(map(float, rng[:2]))

        # Sanity checks for reasonable temperature ranges
        if hi - lo > 20:  # Range too wide
            mid = (lo + hi) / 2
            lo, hi = mid - 5, mid + 5
        if lo < -30 or hi > 50:  # Extreme temperatures
            return (20.0, 27.0)

        return (lo, hi)

    def _calculate_climate_score(self, weather_data: Dict[str, Any], temp_range: Tuple[float, float]) -> float:
        """Calculate enhanced climate-aware destination score."""
        lo, hi = temp_range
        temp_avg = weather_data.get('temp_avg', 20)

        # Base temperature score (0-10)
        target_temp = (lo + hi) / 2
        temp_distance = abs(temp_avg - target_temp)
        range_size = hi - lo

        if lo <= temp_avg <= hi:
            # Perfect match gets high score, gradually decrease toward edges
            temp_score = 10 - (temp_distance / (range_size / 2)) * 2
            temp_score = max(8, temp_score)  # Minimum 8 if within range
        else:
            # Outside range - score based on distance
            temp_score = max(0, 8 - temp_distance * 1.5)

        # Climate stability bonus/penalty
        climate_modifier = 0
        if 'climate_deviation' in weather_data:
            deviation: WeatherDeviation = weather_data['climate_deviation']

            # Penalty for unusual conditions
            severity_penalties = {
                "normal": 0,
                "slight": -0.5,
                "significant": -1.5,
                "extreme": -3.0
            }
            climate_modifier += severity_penalties.get(deviation.deviation_severity, 0)

            # Bonus for favorable unusual weather (warmer than normal within target range)
            if (lo <= temp_avg <= hi and
                    deviation.is_warmer_than_normal and
                    deviation.deviation_severity in ["slight", "significant"]):
                climate_modifier += 0.5

        return max(0, min(10, temp_score + climate_modifier))

    # ------------------------------------------------------------------
    # Enhanced query parsing with better error handling
    # ------------------------------------------------------------------

    async def parse_travel_query(self, query: str) -> TravelQuery:
        """Enhanced query parsing with robust error handling and validation."""
        self._update_progress("Parser", "Analyzing query", details="Understanding travel requirements")

        system_prompt = """You are a travel query parser. Extract structured information 
and return ONLY valid JSON with these exact fields:
{
    "regions": ["Europe", "Asia"],
    "forecast_date": "tomorrow", 
    "temperature_range": [20, 27],
    "origin_city": "Toronto",
    "additional_criteria": ["morning flights", "direct flights"]
}

CRITICAL: temperature_range MUST be exactly 2 numbers in Celsius!
Return ONLY the JSON object, no other text."""

        try:
            rsp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=250,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse: {query}"},
                ],
            )

            # Enhanced JSON parsing with better error handling
            response_text = rsp.choices[0].message.content.strip()
            print(f"DEBUG: Query parser response: {response_text}")

            # Try to extract JSON if it's wrapped in other text
            if not response_text.startswith('{'):
                # Look for JSON block in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")

            data = json.loads(response_text)

        except Exception as e:
            print(f"Query parsing failed: {e}, using fallback")
            data = {
                "regions": ["Europe"],
                "forecast_date": "tomorrow",
                "temperature_range": [20, 27],
                "origin_city": "Toronto",
                "additional_criteria": []
            }

        # Enhanced temperature range handling with validation
        temp_rng = data.get("temperature_range", [20, 27])
        temp_rng = self._normalize_temperature_range(temp_rng)

        # Enhanced date parsing with validation
        date_str = data.get("forecast_date", "tomorrow")
        try:
            parsed_date = _nl_date_to_dt(date_str)
        except Exception as e:
            print(f"Date parsing failed: {e}, using tomorrow")
            parsed_date = _today() + timedelta(days=1)
            date_str = "tomorrow"

        parsed = TravelQuery(
            regions=data.get("regions", ["Europe"]),
            forecast_date=date_str,
            forecast_date_parsed=parsed_date,
            temperature_range=tuple(temp_rng),
            origin_city=data.get("origin_city", "Toronto"),
            additional_criteria=data.get("additional_criteria", []),
            raw_query=query,
        )
        return parsed

    @staticmethod
    def _normalize_temperature_range(temp_input: Any) -> List[float]:
        """Robust temperature range normalization."""
        if temp_input is None:
            return [20.0, 27.0]
        if isinstance(temp_input, (int, float)):
            # Single number - create range around it
            base = float(temp_input)
            return [base - 3, base + 3]
        if isinstance(temp_input, str):
            # Try to parse string like "20-25" or "20 to 25"
            try:
                # Remove common separators and extract numbers
                nums = re.findall(r'-?\d+\.?\d*', temp_input)
                if len(nums) >= 2:
                    return [float(nums[0]), float(nums[1])]
                elif len(nums) == 1:
                    base = float(nums[0])
                    return [base - 3, base + 3]
            except (ValueError, IndexError):
                pass
            return [20.0, 27.0]
        if isinstance(temp_input, (list, tuple)):
            try:
                floats = [float(x) for x in temp_input if x is not None]
                if len(floats) >= 2:
                    return sorted(floats[:2])
                elif len(floats) == 1:
                    base = floats[0]
                    return [base - 3, base + 3]
            except (ValueError, TypeError):
                pass
            return [20.0, 27.0]
        # Unknown type
        return [20.0, 27.0]

    # ------------------------------------------------------------------
    # Enhanced research methods with fixed API calls
    # ------------------------------------------------------------------

    async def _cities_for_regions(
            self, regions: List[str], n: int = 8
    ) -> List[Dict[str, str]]:
        """Enhanced city generation with better error handling and progress tracking."""
        cities: List[Dict[str, str]] = []

        for region_idx, region in enumerate(regions, 1):
            self._update_progress("Researcher", "Generating cities", region_idx, len(regions),
                                  f"Finding diverse cities in {region}")

            # Normalize region names
            region_normalized = self._normalize_region_name(region)

            temperature = random.uniform(0.7, 0.9)

            # Enhanced prompt for better JSON reliability
            prompt = f"""Generate EXACTLY {n} diverse cities in {region_normalized}. 
Mix famous destinations (40%) with hidden gems (60%). 

Return a valid JSON array with this EXACT format:
[
  {{"city": "Barcelona", "country": "Spain"}},
  {{"city": "Porto", "country": "Portugal"}},
  {{"city": "Krakow", "country": "Poland"}}
]

Requirements:
- EXACTLY {n} cities
- Valid JSON format only
- Mix of population sizes
- Geographic diversity within {region_normalized}
- Include cities with tourist infrastructure"""

            try:
                rsp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=temperature,
                    max_tokens=500,  # Increased for larger responses
                    messages=[
                        {"role": "system",
                         "content": "You are a travel expert. Return only valid JSON arrays of cities."},
                        {"role": "user", "content": prompt}
                    ],
                )

                response_text = rsp.choices[0].message.content.strip()
                print(f"DEBUG: City generation response for {region}: {response_text[:200]}...")

                # Enhanced JSON parsing
                if not response_text.startswith('['):
                    # Look for JSON array in the response
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(0)
                    else:
                        raise ValueError("No JSON array found in response")

                batch = json.loads(response_text)

                # Validate and enhance city data
                valid_cities = 0
                for item in batch:
                    if isinstance(item, dict) and 'city' in item and 'country' in item:
                        item["region"] = region_normalized
                        cities.append(item)
                        valid_cities += 1

                print(f"DEBUG: Successfully parsed {valid_cities} cities from {region}")

            except Exception as e:
                print(f"City generation failed for {region}: {e}")
                # Enhanced fallback with region-specific cities
                fallback_cities = self._get_fallback_cities(region_normalized)
                cities.extend(fallback_cities)
                print(f"DEBUG: Using {len(fallback_cities)} fallback cities for {region}")

        random.shuffle(cities)
        return cities[:n * len(regions)]  # Limit total cities

    @staticmethod
    def _normalize_region_name(region: str) -> str:
        """Normalize region names for consistency."""
        region_lower = region.lower().strip()

        # Common region mappings
        mappings = {
            "eu": "Europe",
            "european": "Europe",
            "asia": "Asia",
            "asian": "Asia",
            "south america": "South America",
            "south american": "South America",
            "north america": "North America",
            "north american": "North America",
            "africa": "Africa",
            "african": "Africa",
            "oceania": "Oceania",
        }

        return mappings.get(region_lower, region.title())

    @staticmethod
    def _get_fallback_cities(region: str) -> List[Dict[str, str]]:
        """Enhanced fallback cities with better regional coverage."""
        fallbacks = {
            "Europe": [
                {"city": "Barcelona", "country": "Spain", "region": "Europe"},
                {"city": "Porto", "country": "Portugal", "region": "Europe"},
                {"city": "Krakow", "country": "Poland", "region": "Europe"},
                {"city": "Tallinn", "country": "Estonia", "region": "Europe"},
                {"city": "Ljubljana", "country": "Slovenia", "region": "Europe"},
                {"city": "Bergen", "country": "Norway", "region": "Europe"},
            ],
            "Asia": [
                {"city": "Kyoto", "country": "Japan", "region": "Asia"},
                {"city": "Hoi An", "country": "Vietnam", "region": "Asia"},
                {"city": "Luang Prabang", "country": "Laos", "region": "Asia"},
                {"city": "Kandy", "country": "Sri Lanka", "region": "Asia"},
                {"city": "Chiang Mai", "country": "Thailand", "region": "Asia"},
                {"city": "Yogyakarta", "country": "Indonesia", "region": "Asia"},
            ],
            "South America": [
                {"city": "Valparaiso", "country": "Chile", "region": "South America"},
                {"city": "Cartagena", "country": "Colombia", "region": "South America"},
                {"city": "Cusco", "country": "Peru", "region": "South America"},
                {"city": "Montevideo", "country": "Uruguay", "region": "South America"},
            ]
        }

        return fallbacks.get(region, [
            {"city": "Madrid", "country": "Spain", "region": region}
        ])

    async def _coords(self, city: str, country: str) -> Optional[Dict[str, float]]:
        """Enhanced coordinate lookup with better error handling."""
        try:
            # Clean up city and country names
            city_clean = city.strip()
            country_clean = country.strip()

            # Proper URL encoding for special characters
            query = f"{city_clean}, {country_clean}"
            encoded_query = quote(query, safe=',')
            url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=1"

            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                        url,
                        headers={"User-Agent": "AgenticAI-Enhanced/1.0"},
                        timeout=aiohttp.ClientTimeout(total=10)
                ) as r:
                    if r.status == 200:
                        data = await r.json()
                        if data and len(data) > 0:
                            return {
                                "lat": float(data[0]["lat"]),
                                "lon": float(data[0]["lon"])
                            }
                    else:
                        print(f"Geocoding HTTP error {r.status} for {city}, {country}")
        except Exception as e:
            print(f"Geocoding failed for {city}, {country}: {e}")
        return None

    async def _openmeteo_day(
            self, lat: float, lon: float, day: datetime
    ) -> TaskResult:
        """DEPRECATED: Use get_weather_data_enhanced instead."""
        # This method is deprecated and replaced by the fixed weather API
        from src_code.weather_api import get_weather_data_enhanced
        return await get_weather_data_enhanced(lat, lon, day, "", self.openweather_key)

    @staticmethod
    def _weather_code_to_description(code: int) -> str:
        """Convert WMO weather code to description."""
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            66: "Light freezing rain", 67: "Heavy freezing rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        return weather_codes.get(code, f"Unknown weather (code {code})")

    # ------------------------------------------------------------------
    # Enhanced agent workflow with progress tracking
    # ------------------------------------------------------------------

    async def _agent(self, role: AgentRole, q: TravelQuery) -> AgentMessage:
        """Enhanced agent dispatch with progress tracking and fixed APIs."""
        tools: List[str] = []
        meta: Dict[str, Any] = {}
        content = ""

        if role == AgentRole.PLANNER:
            self._update_progress("Planner", "Creating strategy",
                                  details="Analyzing requirements and planning approach")

            lo, hi = self._safe_temp_range(q)
            days_ahead = (q.forecast_date_parsed - _today()).days

            content = f"""🎯 **ENHANCED PLANNING STRATEGY**

**Target Criteria:**
- Temperature Range: {lo}°C - {hi}°C  
- Travel Date: {q.forecast_date} ({q.forecast_date_parsed.strftime('%Y-%m-%d')})
- Regions: {', '.join(q.regions)}
- Origin: {q.origin_city}
- Days ahead: {days_ahead}

**Enhanced Research Plan:**
1. Generate diverse cities (famous + hidden gems)
2. Gather coordinates and weather forecasts
3. **NEW: Comprehensive climate analysis**
   - Historical baselines for each city/month
   - Weather deviation analysis with severity scoring
   - Climate alert generation for unusual conditions
4. Enhanced scoring with climate intelligence
5. Climate-aware recommendations with traveler impact

**Climate Intelligence Features:**
✅ Historical climate baselines (30-year normals)
✅ Weather deviation analysis with severity levels
✅ Climate alerts for unusual conditions
✅ Traveler impact assessments
✅ Enhanced destination scoring"""

            tools = ["enhanced_planning", "climate_intelligence_integration"]

        elif role == AgentRole.RESEARCHER:
            content = "🔍 **ENHANCED RESEARCH WITH CLIMATE INTELLIGENCE**\n\n"

            # Reset climate alerts for new research
            self.climate_alerts.clear()

            try:
                # 1. Generate diverse cities
                self._update_progress("Researcher", "Generating cities", details="Finding diverse destinations")
                cities = await self._cities_for_regions(q.regions, 8)
                content += f"📍 Generated {len(cities)} diverse cities across {len(q.regions)} regions\n\n"

                weather_results: List[Dict[str, Any]] = []
                climate_alerts_count = 0

                # 2. Enhanced weather + climate analysis for each city
                content += f"🌡️ **ENHANCED WEATHER ANALYSIS** for {q.forecast_date}...\n\n"

                for i, item in enumerate(cities, 1):
                    city_name = item["city"]
                    country_name = item["country"]

                    self._update_progress("Researcher", "Analyzing weather", i, len(cities),
                                          f"Processing {city_name}, {country_name}")

                    content += f"   🏙️ **{i}. {city_name}, {country_name}**\n"

                    # Get coordinates
                    coords = await self._coords(city_name, country_name)
                    if not coords:
                        content += f"   ❌ Could not geocode location\n\n"
                        continue

                    content += f"   📍 Coordinates: {coords['lat']:.2f}, {coords['lon']:.2f}\n"

                    # Get weather forecast using FIXED weather API (including paid OpenWeatherMap)
                    wx = await get_weather_data_enhanced(
                        coords["lat"], coords["lon"], q.forecast_date_parsed,
                        f"{city_name}, {country_name}", self.openweather_key
                    )
                    if not wx.success:
                        content += f"   ❌ Weather forecast failed: {wx.error[:80]}...\n\n"
                        continue

                    # Enhanced climate analysis
                    baseline = await self.climate.get_historical_climate_baseline(
                        city_name, country_name, q.forecast_date_parsed.month
                    )

                    deviation = await self.climate.analyse_deviation(wx.data, baseline)

                    # Display weather + climate analysis
                    temp_avg = wx.data["temp_avg"]
                    content += f"   🌡️ Forecast: {temp_avg:.1f}°C, {wx.data.get('weather_desc', 'variable')}\n"
                    content += f"   📊 Historical: {baseline.avg_temp_mean:.1f}°C ({baseline.climate_zone})\n"

                    # Climate deviation analysis
                    if deviation.is_climate_alert:
                        alert_icon = deviation.get_alert_icon()
                        temp_icon = deviation.get_temperature_trend_icon()
                        content += f"   {alert_icon} **CLIMATE ALERT**: {deviation.deviation_severity.upper()} deviation\n"
                        content += f"   {temp_icon} {deviation.temperature_deviation_c:+.1f}°C from normal\n"
                        content += f"   💡 Impact: {deviation.traveler_impact[:60]}...\n"

                        # Generate and track climate alert
                        alert = ClimateAlert.from_deviation(city_name, deviation)
                        if alert:
                            self.climate_alerts.append(alert)
                            climate_alerts_count += 1
                    else:
                        severity_icon = "✅" if deviation.deviation_severity == "normal" else "⚡"
                        content += f"   {severity_icon} Climate: {deviation.deviation_severity} ({deviation.temperature_deviation_c:+.1f}°C)\n"

                    # Create enhanced weather data
                    wx_enhanced = {
                        **item,
                        **wx.data,
                        "historical_baseline": baseline,
                        "climate_deviation": deviation,
                        "climate_score": self._calculate_climate_score(
                            {**wx.data, "climate_deviation": deviation},
                            self._safe_temp_range(q)
                        ),
                        "coordinates": coords
                    }
                    weather_results.append(wx_enhanced)
                    content += f"\n"

                # Step 3: Enhanced Flight Analysis with FR24 Integration
                self._update_progress("Researcher", "Analyzing flights",
                                      details="Getting flight data and recommendations")
                content += f"\n✈️ **ENHANCED FLIGHT ANALYSIS** from {q.origin_city}...\n\n"

                if self.flight_analyzer.is_fr24_available():
                    content += f"📡 FlightRadar24 API: Connected ✅ (Real flight data available)\n"
                else:
                    content += f"📡 FlightRadar24 API: Not configured ⚠️ (Using enhanced geographic estimates)\n"
                content += f"\n"

                enhanced_flight_results = []
                fr24_successful_calls = 0
                fr24_failed_calls = 0

                for i, weather_data in enumerate(weather_results, 1):
                    city_name = weather_data['city'].split(',')[0].strip()
                    country_name = weather_data['city'].split(',')[1].strip() if ',' in weather_data['city'] else ""

                    self._update_progress("Researcher", "Flight analysis", i, len(weather_results),
                                          f"Analyzing flights to {city_name}")

                    content += f"   🛫 **{i}. {city_name} Flight Analysis:**\n"

                    try:
                        # Get comprehensive flight recommendations
                        flight_analysis = await self.flight_analyzer.generate_flight_recommendations(
                            q.origin_city, city_name, "", country_name, q.forecast_date_parsed
                        )

                        if 'error' not in flight_analysis:
                            # Merge flight data with weather data
                            enhanced_data = {
                                **weather_data,
                                **flight_analysis,
                                "flight_analysis_success": True
                            }
                            enhanced_flight_results.append(enhanced_data)

                            # Display enhanced flight information
                            origin_airport = flight_analysis['origin_airport']
                            dest_airport = flight_analysis['destination_airport']
                            distance = flight_analysis['distance_km']

                            content += f"   📍 Route: {origin_airport.iata_code} ({origin_airport.city}) → {dest_airport.iata_code} ({dest_airport.city})\n"
                            content += f"   📏 Distance: {distance:.0f} km\n"
                            content += f"   🕐 Estimated time: {flight_analysis['estimated_flight_time_hours']:.1f}h\n"
                            content += f"   🛤️ Routing: {flight_analysis['estimated_routing']}\n"

                            # Display FlightRadar24 status and recommendations
                            if flight_analysis.get('real_flight_data_available'):
                                fr24_successful_calls += 1
                                recommendations = flight_analysis.get('recommendations', [])
                                content += f"   ✅ **REAL FLIGHT DATA**: {len(recommendations)} live recommendations\n"

                                if recommendations:
                                    best_flight = recommendations[0]
                                    content += f"   🥇 Best option: {best_flight.airline} {best_flight.flight_number}\n"
                                    content += f"       ⏱️ Duration: {best_flight.duration_display} | Score: {best_flight.overall_score:.1f}/10\n"
                                    content += f"       🔄 Stops: {best_flight.stops} | Reliability: {best_flight.reliability_score:.1f}/10\n"
                            else:
                                fr24_failed_calls += 1
                                error_msg = flight_analysis.get('fr24_error', 'API unavailable')
                                content += f"   📊 Using enhanced geographic estimates ({error_msg[:50]}...)\n"

                                recommendations = flight_analysis.get('recommendations', [])
                                if recommendations:
                                    best_flight = recommendations[0]
                                    content += f"   📈 Estimated best: {best_flight.airline} route\n"
                                    content += f"       ⏱️ Duration: {best_flight.duration_display} | Estimated score: {best_flight.overall_score:.1f}/10\n"
                        else:
                            content += f"   ❌ Flight analysis failed: {flight_analysis['error'][:60]}...\n"
                            fr24_failed_calls += 1

                            # Add basic flight data to maintain compatibility
                            enhanced_data = {
                                **weather_data,
                                "flight_analysis_success": False,
                                "flight_error": flight_analysis['error']
                            }
                            enhanced_flight_results.append(enhanced_data)

                    except Exception as e:
                        content += f"   ❌ Flight analysis error: {str(e)[:60]}...\n"
                        fr24_failed_calls += 1

                        # Add basic data to maintain compatibility
                        enhanced_data = {
                            **weather_data,
                            "flight_analysis_success": False,
                            "flight_error": str(e)
                        }
                        enhanced_flight_results.append(enhanced_data)

                    content += f"\n"

                # Enhanced research summary with flight intelligence
                content += f"✅ **ENHANCED RESEARCH COMPLETE**\n"
                content += f"- Successfully analyzed: {len(enhanced_flight_results)}/{len(cities)} destinations\n"
                content += f"- Climate alerts generated: {climate_alerts_count}\n"
                content += f"- Flight analysis: {len(enhanced_flight_results)} destinations\n"
                content += f"- Enhanced climate intelligence: ✅ Active\n"
                content += f"- Enhanced flight intelligence: ✅ Active\n"

                if self.flight_analyzer.is_fr24_available():
                    content += f"\n✈️ **FLIGHT ANALYSIS SUMMARY:**\n"
                    content += f"   - Real flight data calls: {fr24_successful_calls} successful, {fr24_failed_calls} fallback\n"
                    content += f"   - FlightRadar24 integration: {'✅ Active' if fr24_successful_calls > 0 else '⚠️ Fallback mode'}\n"
                    content += f"   - Airport database: ✅ {len(self.flight_analyzer.get_available_airports())} airports available\n"
                else:
                    content += f"\n✈️ **FLIGHT ANALYSIS SUMMARY:**\n"
                    content += f"   - Enhanced geographic estimates: ✅ All destinations\n"
                    content += f"   - Airport database: ✅ {len(self.flight_analyzer.get_available_airports())} airports available\n"

                if climate_alerts_count > 0:
                    content += f"\n🚨 **{climate_alerts_count} CLIMATE ALERTS** detected - unusual weather expected!\n"

                meta = {
                    "weather_results": weather_results,
                    "enhanced_flight_results": enhanced_flight_results,
                    "climate_alerts": self.climate_alerts,
                    "cities_processed": len(cities),
                    "successful_analyses": len(enhanced_flight_results),
                    "climate_intelligence_active": True,
                    "flight_intelligence_active": True,
                    "fr24_successful_calls": fr24_successful_calls,
                    "fr24_failed_calls": fr24_failed_calls,
                    "fr24_available": self.flight_analyzer.is_fr24_available()
                }
                tools = ["enhanced_weather_analysis", "climate_intelligence", "deviation_analysis", "fixed_apis",
                         "enhanced_flight_analysis", "fr24_integration"]

            except Exception as e:
                content += f"\n❌ Enhanced research failed: {str(e)}"
                import traceback
                print(f"Research error: {traceback.format_exc()}")
                meta = {"error": str(e), "climate_intelligence_active": False}
                tools = ["error_handling"]

        elif role == AgentRole.ANALYZER:
            self._update_progress("Analyzer", "Climate-aware analysis",
                                  details="Scoring destinations with climate intelligence")

            # Enhanced analysis with climate intelligence
            r_messages = [m for m in self.conversation_history if m.agent_role == AgentRole.RESEARCHER]

            if not r_messages or not r_messages[-1].metadata.get("enhanced_flight_results"):
                content = "❌ No enhanced research data available for analysis"
                tools = ["error_handling"]
            else:
                research_meta = r_messages[-1].metadata
                flight_results = research_meta["enhanced_flight_results"]
                climate_alerts = research_meta.get("climate_alerts", [])
                fr24_available = research_meta.get("fr24_available", False)
                fr24_successful = research_meta.get("fr24_successful_calls", 0)

                content = "📊 **ENHANCED CLIMATE & FLIGHT ANALYSIS**\n\n"

                lo, hi = self._safe_temp_range(q)
                scored_destinations: List[Dict[str, Any]] = []

                # Enhanced scoring for each destination with flight intelligence
                for flight_result in flight_results:
                    temp_avg = flight_result["temp_avg"]
                    meets_criteria = lo <= temp_avg <= hi
                    deviation: WeatherDeviation = flight_result["climate_deviation"]

                    # Enhanced climate-aware scoring
                    climate_score = flight_result.get("climate_score", 5.0)

                    # Enhanced flight scoring
                    flight_score = 5.0  # Default
                    flight_bonus = 0

                    if flight_result.get("flight_analysis_success"):
                        recommendations = flight_result.get("recommendations", [])
                        if recommendations:
                            best_flight = recommendations[0]

                            # Use real flight scoring if available
                            flight_score = best_flight.overall_score

                            # Bonus for real flight data
                            if flight_result.get("real_flight_data_available"):
                                flight_bonus = 1.0

                            # Distance-based adjustments
                            distance = flight_result.get("distance_km", 5000)
                            if distance < 2000:  # Short haul bonus
                                flight_bonus += 0.5
                            elif distance > 8000:  # Long haul penalty
                                flight_score *= 0.9
                    else:
                        # Fallback scoring based on estimated distance
                        distance = flight_result.get("distance_km", 5000)
                        flight_score = max(3, 8 - distance / 2000)  # Simpler distance-based score

                    # Additional scoring factors
                    humidity_score = max(6, 10 - abs(flight_result["humidity"] - 60) / 10)
                    meets_temp_bonus = 2 if meets_criteria else 0

                    # ENHANCED FINAL SCORE with flight intelligence
                    overall_score = (
                            (climate_score * 0.35) +  # Climate importance
                            (flight_score * 0.35) +  # Flight importance (equal to climate)
                            (humidity_score * 0.15) +  # Humidity
                            meets_temp_bonus +  # Temperature match bonus
                            flight_bonus  # Real data bonus
                    )
                    overall_score = max(0, min(10, overall_score))

                    flight_enhanced = {
                        **flight_result,
                        "meets_criteria": meets_criteria,
                        "climate_score": climate_score,
                        "flight_score": flight_score,
                        "humidity_score": humidity_score,
                        "flight_bonus": flight_bonus,
                        "overall_score": overall_score,
                        "climate_alert": deviation.is_climate_alert,
                        "temperature_trend": "warmer" if deviation.is_warmer_than_normal else "cooler" if deviation.is_cooler_than_normal else "normal"
                    }
                    scored_destinations.append(flight_enhanced)

                # Sort by overall score
                scored_destinations.sort(key=lambda x: x["overall_score"], reverse=True)

                # Analysis summary with flight intelligence
                meeting_criteria = sum(1 for d in scored_destinations if d["meets_criteria"])
                avg_score = sum(d["overall_score"] for d in scored_destinations) / len(scored_destinations)
                flight_successes = sum(1 for d in scored_destinations if d.get("flight_analysis_success"))

                content += f"**Enhanced Analysis Summary:**\n"
                content += f"- Meeting temperature criteria: {meeting_criteria}/{len(scored_destinations)}\n"
                content += f"- Average destination score: {avg_score:.1f}/10 (climate + flight enhanced)\n"
                content += f"- Climate alerts active: {len(climate_alerts)}\n"
                content += f"- Flight analysis successful: {flight_successes}/{len(scored_destinations)}\n"

                if fr24_available:
                    content += f"- FlightRadar24 data: {fr24_successful} destinations with real flight data\n"
                else:
                    content += f"- Flight estimates: Enhanced geographic analysis for all destinations\n"

                content += f"\n"

                # Top destinations with flight information
                content += "🏆 **TOP DESTINATIONS (Climate + Flight Enhanced Scoring):**\n\n"
                for i, dest in enumerate(scored_destinations[:5], 1):
                    city_name = dest["city"]
                    temp = dest["temp_avg"]
                    score = dest["overall_score"]
                    meets = "✅" if dest["meets_criteria"] else "⚠️"
                    climate_icon = dest["climate_deviation"].get_alert_icon()

                    # Flight information
                    flight_info = ""
                    if dest.get("flight_analysis_success"):
                        recommendations = dest.get("recommendations", [])
                        if recommendations:
                            best_flight = recommendations[0]
                            flight_info = f" | ✈️ {best_flight.duration_display}"
                            if dest.get("real_flight_data_available"):
                                flight_info += f" ({best_flight.airline})"
                            else:
                                flight_info += f" (est.)"
                        else:
                            flight_info = f" | ✈️ {dest.get('estimated_flight_time_hours', 8):.1f}h (est.)"
                    else:
                        flight_info = " | ✈️ Flight data unavailable"

                    content += f"{i}. {meets} **{city_name}** {climate_icon}\n"
                    content += f"   🌡️ {temp:.1f}°C | 📊 Score: {score:.1f}/10 | {dest['temperature_trend'].title()}{flight_info}\n"

                    if dest["climate_alert"]:
                        content += f"   ⚠️ Climate Alert: {dest['climate_deviation'].deviation_severity}\n"

                    # Flight details for top 3
                    if i <= 3 and dest.get("flight_analysis_success"):
                        recommendations = dest.get("recommendations", [])
                        if recommendations:
                            best_flight = recommendations[0]
                            content += f"   ✈️ Best flight: {best_flight.airline} | Reliability: {best_flight.reliability_score:.1f}/10\n"

                    content += f"\n"

                meta = {
                    "scored_destinations": scored_destinations,
                    "climate_alerts": climate_alerts,
                    "meeting_criteria_count": meeting_criteria,
                    "average_score": avg_score,
                    "flight_analysis_successes": flight_successes,
                    "fr24_available": fr24_available,
                    "fr24_successful_calls": fr24_successful,
                    "climate_enhanced_analysis": True,
                    "flight_enhanced_analysis": True
                }
                tools = ["enhanced_climate_analysis", "deviation_scoring", "climate_alerts", "enhanced_flight_analysis",
                         "fr24_integration"]

        elif role == AgentRole.SYNTHESIZER:
            self._update_progress("Synthesizer", "Generating recommendations",
                                  details="Creating itineraries and final recommendations")

            # Enhanced synthesis with weather-appropriate itineraries
            a_messages = [m for m in self.conversation_history if m.agent_role == AgentRole.ANALYZER]

            if not a_messages or not a_messages[-1].metadata.get("scored_destinations"):
                content = "❌ No enhanced analysis data available for synthesis"
                tools = ["error_handling"]
            else:
                analysis_meta = a_messages[-1].metadata
                destinations = analysis_meta["scored_destinations"]
                climate_alerts = analysis_meta.get("climate_alerts", [])

                content = "🎯 **COMPREHENSIVE RECOMMENDATIONS: CLIMATE + FLIGHT + ITINERARIES**\n\n"

                if destinations:
                    top_3 = destinations[:3]

                    content += "🏆 **TOP RECOMMENDATIONS WITH COMPREHENSIVE ANALYSIS:**\n\n"

                    # Generate detailed itineraries for top 2 destinations
                    itineraries_generated = 0

                    for i, dest in enumerate(top_3, 1):
                        city_name = dest["city"]
                        country_name = dest["country"]
                        temp = dest["temp_avg"]
                        score = dest["overall_score"]
                        deviation = dest["climate_deviation"]
                        baseline = dest.get("historical_baseline")

                        status_icon = "🎯" if dest["meets_criteria"] else "⭐"
                        climate_icon = deviation.get_alert_icon()
                        temp_icon = deviation.get_temperature_trend_icon()

                        content += f"**{i}. {status_icon} {city_name}, {country_name}** {climate_icon}\n"
                        content += f"   🌡️ Weather: {temp:.1f}°C, {dest.get('weather_desc', 'variable conditions')}\n"
                        content += f"   📊 Overall Score: {score:.1f}/10 (climate + flight enhanced)\n"
                        content += f"   {temp_icon} Climate: {deviation.deviation_severity} vs normal\n"

                        # Enhanced flight information display
                        if dest.get("flight_analysis_success"):
                            recommendations = dest.get("recommendations", [])
                            if recommendations:
                                best_flight = recommendations[0]
                                data_quality = "🔴 Live" if dest.get("real_flight_data_available") else "📊 Estimated"
                                content += f"   ✈️ Best Flight: {best_flight.airline} {best_flight.flight_number} | {best_flight.duration_display} | {data_quality}\n"
                                content += f"   🛫 Route: {best_flight.origin_airport} → {best_flight.destination_airport} | Score: {best_flight.overall_score:.1f}/10\n"
                                if best_flight.stops > 0:
                                    content += f"   🔄 Stops: {best_flight.stops} | Reliability: {best_flight.reliability_score:.1f}/10\n"
                                else:
                                    content += f"   🎯 Direct flight | Reliability: {best_flight.reliability_score:.1f}/10\n"
                            else:
                                distance = dest.get("distance_km", 0)
                                est_time = dest.get("estimated_flight_time_hours", 8)
                                content += f"   ✈️ Flight: ~{est_time:.1f}h, {distance:.0f}km | Estimated routing\n"
                        else:
                            flight_error = dest.get("flight_error", "Analysis failed")
                            content += f"   ❌ Flight data: {flight_error[:60]}...\n"

                        if deviation.is_climate_alert:
                            content += f"   ⚠️ **Climate Alert**: {deviation.traveler_impact[:100]}...\n"

                        # Generate detailed itinerary for top 2 destinations
                        if i <= 2:
                            self._update_progress("Synthesizer", "Generating itinerary", i, 2,
                                                  f"Creating weather-appropriate itinerary for {city_name}")

                            content += f"\n   🗓️ **4-DAY WEATHER-APPROPRIATE ITINERARY:**\n"
                            try:
                                # Prepare weather data for itinerary generator
                                weather_data = {
                                    'temp_avg': dest['temp_avg'],
                                    'temp_max': dest.get('temp_max', dest['temp_avg'] + 3),
                                    'temp_min': dest.get('temp_min', dest['temp_avg'] - 3),
                                    'humidity': dest.get('humidity', 60),
                                    'weather_desc': dest.get('weather_desc', 'variable conditions')
                                }

                                itinerary = await self.itinerary_generator.generate_weather_itinerary(
                                    city_name, country_name, weather_data, baseline, deviation, 4
                                )

                                # Indent the itinerary for better formatting
                                indented_lines = []
                                for line in itinerary.split('\n'):
                                    if line.strip():
                                        indented_lines.append(f"   {line}")
                                    else:
                                        indented_lines.append("")

                                content += "\n".join(indented_lines) + "\n"
                                itineraries_generated += 1

                            except Exception as e:
                                content += f"   ❌ Itinerary generation failed: {str(e)[:100]}...\n"
                                print(f"Itinerary generation error for {city_name}: {e}")

                        content += f"\n"

                    # Climate alerts summary (if any)
                    if climate_alerts:
                        content += f"🚨 **CLIMATE ALERTS ({len(climate_alerts)}):**\n\n"
                        for alert in climate_alerts:
                            content += f"⚠️ **{alert.city}**: {alert.message}\n"
                            if alert.recommendations:
                                content += f"   💡 {alert.recommendations[0]}\n"
                        content += f"\n"

                    # Enhanced travel booking strategy with flight intelligence
                    lo, hi = self._safe_temp_range(q)
                    content += f"🎯 **COMPREHENSIVE BOOKING STRATEGY:**\n\n"
                    content += f"**Target Details:**\n"
                    content += f"- Date: {q.forecast_date_parsed.strftime('%A, %B %d, %Y')}\n"
                    content += f"- Temperature Range: {lo}°C - {hi}°C\n"
                    content += f"- Origin: {q.origin_city}\n"
                    content += f"- Itineraries Generated: {itineraries_generated}/2 top destinations\n"

                    meeting_criteria = sum(1 for d in destinations if d["meets_criteria"])
                    flight_successes = sum(1 for d in destinations if d.get("flight_analysis_success"))
                    fr24_available = analysis_meta.get("fr24_available", False)
                    fr24_calls = analysis_meta.get("fr24_successful_calls", 0)

                    content += f"\n**Comprehensive Analysis Summary:**\n"
                    content += f"- Destinations analyzed: {len(destinations)}\n"
                    content += f"- Meeting temperature criteria: {meeting_criteria}/{len(destinations)}\n"
                    content += f"- Flight analysis successful: {flight_successes}/{len(destinations)}\n"

                    if fr24_available:
                        content += f"- Real flight data: {fr24_calls} destinations with live FlightRadar24 data\n"
                        content += f"- **Flight Confidence**: {'High' if fr24_calls > 0 else 'Estimated'} - {'Real-time' if fr24_calls > 0 else 'Geographic'} flight analysis\n"
                    else:
                        content += f"- Flight analysis: Enhanced geographic estimates for all destinations\n"
                        content += f"- **Flight Confidence**: Estimated - Geographic analysis with enhanced airport database\n"

                    if climate_alerts:
                        content += f"- Climate alerts: {len(climate_alerts)} destinations with unusual weather\n"
                        content += f"- **Booking Tip**: Consider flexible tickets, monitor forecasts 1 week before travel\n"
                    else:
                        content += f"- Climate status: Stable seasonal weather expected\n"
                        content += f"- **Booking Confidence**: High - typical seasonal patterns predicted\n"

                    if q.additional_criteria:
                        content += f"- Special requirements: {', '.join(q.additional_criteria)}\n"

                    # Why these destinations excel with flight considerations
                    content += f"\n**Why These Destinations Excel (Climate + Flight Analysis):**\n"
                    for i, dest in enumerate(top_3[:2], 1):
                        city_name = dest["city"]
                        reasons = []

                        if dest["meets_criteria"]:
                            reasons.append(f"Perfect temperature match ({dest['temp_avg']:.1f}°C)")

                        if dest["climate_score"] > 8:
                            reasons.append("Excellent climate conditions")

                        if not dest["climate_alert"]:
                            reasons.append("Stable weather patterns")
                        elif dest["temperature_trend"] == "warmer":
                            reasons.append("Favorably warmer than normal")

                        # Flight-specific reasons
                        if dest.get("flight_analysis_success"):
                            recommendations = dest.get("recommendations", [])
                            if recommendations:
                                best_flight = recommendations[0]
                                if best_flight.overall_score > 8:
                                    reasons.append("Excellent flight options")
                                if best_flight.stops == 0:
                                    reasons.append("Direct flights available")
                                if dest.get("real_flight_data_available"):
                                    reasons.append("Real-time flight data confirmed")

                        if dest.get("data_source") == "open_meteo_forecast":
                            reasons.append("Reliable weather forecast available")

                        content += f"{i}. **{city_name}**: {'; '.join(reasons[:4])}\n"

                    meta = {
                        "final_recommendations": top_3,
                        "climate_alerts": climate_alerts,
                        "total_analyzed": len(destinations),
                        "itineraries_generated": itineraries_generated,
                        "flight_analysis_successes": flight_successes,
                        "fr24_available": fr24_available,
                        "fr24_successful_calls": fr24_calls,
                        "climate_enhanced": True,
                        "flight_enhanced": True,
                        "weather_itineraries": True,
                        "comprehensive_analysis": True
                    }
                    tools = ["comprehensive_synthesis", "weather_itinerary_generation", "enhanced_flight_analysis",
                             "climate_recommendations", "fr24_integration", "traveler_impact_analysis"]
                else:
                    content = "❌ No destinations found meeting criteria\n\n"
                    content += "**Enhanced Suggestions with Flight Intelligence:**\n"
                    content += "- Consider adjusting temperature range (current weather patterns may be unusual)\n"
                    content += "- Try alternative dates (climate deviations may be temporary)\n"
                    content += "- Expand regions (some areas may have better climate stability)\n"
                    content += "- Consider nearby airports (flight routing may offer alternatives)\n"
                    content += "- Use broader criteria (comprehensive analysis shows few perfect matches)\n"
                    meta = {"error": "no_destinations", "climate_enhanced": True, "flight_enhanced": True}
                    tools = ["climate_aware_fallback", "flight_aware_fallback", "enhanced_suggestions"]

        # Create enhanced agent message
        msg = AgentMessage(
            agent_role=role,
            content=content,
            timestamp=_today(),
            tools_used=tools,
            metadata=meta,
        )
        self.conversation_history.append(msg)
        return msg

    # ------------------------------------------------------------------
    # Enhanced public workflow driver with progress tracking
    # ------------------------------------------------------------------

    async def run_agentic_workflow(self, user_query: str) -> List[AgentMessage]:
        """Enhanced workflow with progress tracking and fixed API issues."""
        self.conversation_history.clear()
        self.climate_alerts.clear()

        try:
            # Enhanced query parsing
            q = await self.parse_travel_query(user_query)

            # Run enhanced four-agent workflow with progress tracking
            agents = [
                (AgentRole.PLANNER, "Strategy"),
                (AgentRole.RESEARCHER, "Data Collection"),
                (AgentRole.ANALYZER, "Analysis"),
                (AgentRole.SYNTHESIZER, "Recommendations"),
            ]

            for i, (role, phase) in enumerate(agents, 1):
                self._update_progress("Workflow", f"Agent {i}/4", i, 4, f"{phase} ({role.value})")
                await self._agent(role, q)

            self._update_progress("Workflow", "Complete", 4, 4, "All agents finished")
            return list(self.conversation_history)

        except Exception as e:
            # Enhanced error handling
            error_msg = AgentMessage(
                agent_role=AgentRole.PLANNER,
                content=f"🚨 **WORKFLOW ERROR**\n\nEnhanced workflow failed: {str(e)}\n\nPlease try again with a simpler query.",
                timestamp=_today(),
                tools_used=["error_recovery"],
                metadata={"error": True, "error_type": type(e).__name__, "error_message": str(e)}
            )
            return [error_msg]

    # ------------------------------------------------------------------
    # Enhanced utility methods
    # ------------------------------------------------------------------

    def get_active_climate_alerts(self) -> List[ClimateAlert]:
        """Get currently active climate alerts."""
        return list(self.climate_alerts)

    def get_current_progress(self) -> Optional[ProgressStep]:
        """Get current progress step."""
        return self.current_progress

    def clear_session_data(self) -> None:
        """Clear all session data (useful for testing)."""
        self.conversation_history.clear()
        self.climate_alerts.clear()
        self.climate.clear_cache()
        self.current_progress = None
        # Note: itinerary_generator and flight_analyzer are stateless, no clearing needed


# ---------------------------------------------------------------------------
# Enhanced CLI test runner with progress tracking
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query_txt = (" ".join(sys.argv[1:]) or
                 "Find warm European cities next Friday between 18-24°C from Toronto with climate analysis")

    if not OPENAI_KEY:
        print("❌ OPENAI_KEY required for testing")
        sys.exit(1)

    core = AgenticAISystem(OPENAI_KEY, OPENWEATHER_KEY, FLIGHTRADAR24_KEY)


    def progress_callback(step: ProgressStep):
        print(f"⏳ {step}")


    core.set_progress_callback(progress_callback)


    async def _main():
        print(f"🔍 Testing enhanced workflow with fixed APIs: {query_txt}\n")
        msgs = await core.run_agentic_workflow(query_txt)

        for m in msgs:
            banner = f"\n{'=' * 60}\n[{m.agent_role.value.upper()}] - {m.timestamp.strftime('%H:%M:%S')}\n{'=' * 60}"
            print(banner)
            print(m.content)

            if m.tools_used:
                print(f"\n🛠️ Tools: {', '.join(m.tools_used)}")

        # Show climate alerts if any
        alerts = core.get_active_climate_alerts()
        if alerts:
            print(f"\n🚨 CLIMATE ALERTS ({len(alerts)}):")
            for alert in alerts:
                print(f"   ⚠️ {alert.city}: {alert.message}")


    asyncio.run(_main())