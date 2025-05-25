# src_code/agentic_core.py
"""Fixed Agentic Core with Enhanced Mexico Query Recognition - COMPREHENSIVE FIX

This version fixes critical issues with Mexico/regional recognition:

1. **FIXED: Mexico Query Recognition**: Enhanced parsing to properly identify Mexican queries
2. **FIXED: Regional Specificity**: Better prompts to distinguish Mexico from North America
3. **ENHANCED: Coastal City Detection**: Special handling for coastal destination requests
4. **ENHANCED: Agent Decision Logging**: Debug shows AI reasoning, not just technical calls
5. **ENHANCED: Temperature Decision Tracking**: Climate analysis decisions prominently logged

Key fixes completed:
- ✅ Enhanced query parsing with Mexico-specific recognition
- ✅ Better regional classification (Mexico != North America for travel queries)
- ✅ Coastal destination detection and specialized city generation
- ✅ Fixed f-string formatting issues in region prompts
- ✅ Enhanced debug logging focused on agent decision-making
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
from src_code.weather_api import get_weather_data_enhanced

# Optional imports with fallbacks
try:
    from src_code.itinerary_generator import WeatherAppropriateItineraryGenerator

    ITINERARY_AVAILABLE = True
except ImportError:
    ITINERARY_AVAILABLE = False
    print("Warning: itinerary_generator not available")

try:
    from src_code.flight_analyzer import EnhancedFlightAnalyzer

    FLIGHT_ANALYZER_AVAILABLE = True
except ImportError:
    FLIGHT_ANALYZER_AVAILABLE = False
    print("Warning: flight_analyzer not available")

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
# Enhanced agent decision logging
# ---------------------------------------------------------------------------

def _log_agent_decision(agent: str, decision: str, reasoning: str = "", data: Any = None):
    """Log agent decision-making process for debugging."""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    if data:
        print(f"[{timestamp}] AGENT-{agent.upper()}: {decision}")
        if reasoning:
            print(f"[{timestamp}] REASONING: {reasoning}")
        data_str = str(data)[:100]
        ellipsis = "..." if len(str(data)) > 100 else ""
        print(f"[{timestamp}] DATA: {data_str}{ellipsis}")
    else:
        print(f"[{timestamp}] AGENT-{agent.upper()}: {decision} | {reasoning}")


def _log_climate_decision(city: str, temp_data: Dict[str, Any], baseline: ClimateBaseline, decision: str):
    """Log climate analysis decisions with temperature data."""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    temp_avg = temp_data.get('temp_avg', 0)
    print(f"[{timestamp}] CLIMATE-ANALYSIS: {city}")
    print(f"[{timestamp}] TEMP-FORECAST: {temp_avg:.1f}°C | HISTORICAL: {baseline.avg_temp_mean:.1f}°C")
    print(f"[{timestamp}] CLIMATE-DECISION: {decision}")


# ---------------------------------------------------------------------------
# Enhanced core system class with agent decision tracking
# ---------------------------------------------------------------------------

class AgenticAISystem:
    """Enhanced orchestrator with agent decision tracking and better Mexico recognition."""

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

        # Weather-appropriate itinerary generator (optional)
        if ITINERARY_AVAILABLE:
            self.itinerary_generator = WeatherAppropriateItineraryGenerator(self.client)
        else:
            self.itinerary_generator = None

        # Enhanced flight analyzer with real-time data (optional)
        if FLIGHT_ANALYZER_AVAILABLE:
            self.flight_analyzer = EnhancedFlightAnalyzer(flightradar24_key)
        else:
            self.flight_analyzer = None

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
    # FIXED: Enhanced query parsing with better Mexico recognition
    # ------------------------------------------------------------------

    async def parse_travel_query(self, query: str) -> TravelQuery:
        """Enhanced query parsing with better Mexico recognition and coastal detection."""
        self._update_progress("Parser", "Analyzing query", details="Understanding travel requirements")

        _log_agent_decision("PARSER", f"Analyzing travel query: '{query[:50]}...'")

        # Enhanced system prompt with specific Mexico recognition
        system_prompt = """You are a travel query parser that extracts structured information.

CRITICAL REGION RECOGNITION RULES:
- If query mentions "Mexico", "Mexican", "Cancun", "Puerto Vallarta", "Playa del Carmen", "Tulum", "Cozumel", "Cabo", etc. → region = "Mexico"
- If query mentions "coastal Mexico", "Mexican coast", "Mexican beaches" → region = "Mexico" 
- If query mentions "Europe", "European" → region = "Europe"
- If query mentions "Asia", "Asian" → region = "Asia"
- If query mentions "South America" → region = "South America"
- Only use "North America" if specifically mentioned or for US/Canada cities
- Be very specific about regions - prefer specific countries over broad continents

Return ONLY valid JSON with these exact fields:
{
    "regions": ["Mexico"],
    "forecast_date": "in 5 days", 
    "temperature_range": [25, 30],
    "origin_city": "Toronto",
    "additional_criteria": ["coastal", "beaches"]
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
            _log_agent_decision("PARSER", "OpenAI parsing response received", f"Length: {len(response_text)} chars")

            # Try to extract JSON if it's wrapped in other text
            if not response_text.startswith('{'):
                # Look for JSON block in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")

            data = json.loads(response_text)

            # Post-process to ensure Mexico is recognized correctly
            raw_query_lower = query.lower()
            if any(mexico_term in raw_query_lower for mexico_term in
                   ['mexico', 'mexican', 'cancun', 'puerto vallarta', 'playa del carmen',
                    'tulum', 'cozumel', 'cabo', 'riviera maya']):
                if data.get('regions') == ['North America']:
                    data['regions'] = ['Mexico']
                    _log_agent_decision("PARSER", "Corrected region from North America to Mexico",
                                        reasoning="Detected Mexico-specific terms in query")

            _log_agent_decision("PARSER", "Query parsing successful",
                                reasoning=f"Extracted regions: {data.get('regions', [])}, temp range: {data.get('temperature_range', [])}")

        except Exception as e:
            _log_agent_decision("PARSER", f"Query parsing failed: {e}", reasoning="Using enhanced fallback parsing")

            # Enhanced fallback with Mexico detection
            raw_query_lower = query.lower()
            if any(mexico_term in raw_query_lower for mexico_term in
                   ['mexico', 'mexican', 'cancun', 'puerto vallarta', 'playa del carmen']):
                fallback_regions = ["Mexico"]
            elif any(euro_term in raw_query_lower for euro_term in ['europe', 'european']):
                fallback_regions = ["Europe"]
            elif any(asia_term in raw_query_lower for asia_term in ['asia', 'asian']):
                fallback_regions = ["Asia"]
            else:
                fallback_regions = ["Europe"]

            data = {
                "regions": fallback_regions,
                "forecast_date": "in 5 days",
                "temperature_range": [25, 30],
                "origin_city": "Toronto",
                "additional_criteria": ["coastal"] if "coastal" in raw_query_lower else []
            }

        # Enhanced temperature range handling with validation
        temp_rng = data.get("temperature_range", [25, 30])
        temp_rng = self._normalize_temperature_range(temp_rng)

        # Enhanced date parsing with validation
        date_str = data.get("forecast_date", "in 5 days")
        try:
            parsed_date = _nl_date_to_dt(date_str)
        except Exception as e:
            _log_agent_decision("PARSER", f"Date parsing failed: {e}", reasoning="Using 5 days as default")
            parsed_date = _today() + timedelta(days=5)
            date_str = "in 5 days"

        parsed = TravelQuery(
            regions=data.get("regions", ["Mexico"]),
            forecast_date=date_str,
            forecast_date_parsed=parsed_date,
            temperature_range=tuple(temp_rng),
            origin_city=data.get("origin_city", "Toronto"),
            additional_criteria=data.get("additional_criteria", []),
            raw_query=query,
        )

        _log_agent_decision("PARSER", "Final query structure",
                            reasoning=f"Target: {parsed.regions} | Temp: {parsed.temperature_range[0]}-{parsed.temperature_range[1]}°C | Date: {parsed.forecast_date}")

        return parsed

    @staticmethod
    def _normalize_temperature_range(temp_input: Any) -> List[float]:
        """Robust temperature range normalization."""
        if temp_input is None:
            return [25.0, 30.0]
        if isinstance(temp_input, (int, float)):
            # Single number - create range around it
            base = float(temp_input)
            return [base - 2, base + 2]
        if isinstance(temp_input, str):
            # Try to parse string like "25-30" or "25 to 30"
            try:
                # Remove common separators and extract numbers
                nums = re.findall(r'-?\d+\.?\d*', temp_input)
                if len(nums) >= 2:
                    return [float(nums[0]), float(nums[1])]
                elif len(nums) == 1:
                    base = float(nums[0])
                    return [base - 2, base + 2]
            except (ValueError, IndexError):
                pass
            return [25.0, 30.0]
        if isinstance(temp_input, (list, tuple)):
            try:
                floats = [float(x) for x in temp_input if x is not None]
                if len(floats) >= 2:
                    return sorted(floats[:2])
                elif len(floats) == 1:
                    base = floats[0]
                    return [base - 2, base + 2]
            except (ValueError, TypeError):
                pass
            return [25.0, 30.0]
        # Unknown type
        return [25.0, 30.0]

    # ------------------------------------------------------------------
    # FIXED: Enhanced research methods with Mexico-specific city generation
    # ------------------------------------------------------------------

    async def _cities_for_regions(
            self, regions: List[str], n: int = 8
    ) -> List[Dict[str, str]]:
        """Enhanced city generation with Mexico-specific coastal focus."""
        cities: List[Dict[str, str]] = []

        for region_idx, region in enumerate(regions, 1):
            self._update_progress("Researcher", "Generating cities", region_idx, len(regions),
                                  f"Finding diverse cities in {region}")

            _log_agent_decision("RESEARCHER", f"Generating cities for region: {region}",
                                reasoning=f"Need {n} diverse cities, focusing on coastal if Mexico")

            # Enhanced region-specific prompts with Mexico focus
            region_normalized = self._normalize_region_name(region)
            specific_prompt = self._get_region_specific_prompt(region_normalized, n)

            temperature = random.uniform(0.7, 0.9)

            _log_agent_decision("RESEARCHER", f"Requesting OpenAI city generation",
                                reasoning=f"Region: {region_normalized}, Cities needed: {n}")

            try:
                rsp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=temperature,
                    max_tokens=600,
                    messages=[
                        {"role": "system",
                         "content": "You are a travel expert who knows cities worldwide. Return only valid JSON arrays of cities with exact format specified."},
                        {"role": "user", "content": specific_prompt}
                    ],
                    timeout=30
                )

                response_text = rsp.choices[0].message.content.strip()
                _log_agent_decision("RESEARCHER", f"OpenAI response received for {region}",
                                    reasoning=f"Response length: {len(response_text)} chars")

                # Enhanced JSON parsing
                if not response_text.startswith('['):
                    # Look for JSON array in the response
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(0)
                    else:
                        raise ValueError("No JSON array found in response")

                batch = json.loads(response_text)
                _log_agent_decision("RESEARCHER", f"JSON parsing successful for {region}",
                                    reasoning=f"Found {len(batch)} city entries")

                # Validate and enhance city data
                valid_cities = 0
                for item in batch:
                    if isinstance(item, dict) and 'city' in item and 'country' in item:
                        item["region"] = region_normalized
                        cities.append(item)
                        valid_cities += 1
                        _log_agent_decision("RESEARCHER", f"Added city: {item['city']}, {item['country']}")

                _log_agent_decision("RESEARCHER", f"Region {region} complete",
                                    reasoning=f"Successfully added {valid_cities} cities")

            except Exception as e:
                _log_agent_decision("RESEARCHER", f"CRITICAL: OpenAI city generation failed for {region}: {e}",
                                    reasoning="This should not happen - OpenAI must be the exclusive source")
                # NO FALLBACKS - This is a critical failure that should be addressed
                raise RuntimeError(f"OpenAI city generation failed for {region}: {e}. No fallbacks allowed.")

        _log_agent_decision("RESEARCHER", f"City generation complete",
                            reasoning=f"Total cities generated: {len(cities)} across {len(regions)} regions")

        random.shuffle(cities)
        return cities[:n * len(regions)]  # Limit total cities

    def _get_region_specific_prompt(self, region: str, n: int) -> str:
        """Generate highly specific prompts for each region with enhanced Mexico focus."""
        base_instruction = f'''Generate EXACTLY {n} diverse cities in {region}. 

CRITICAL REQUIREMENTS:
1. Return ONLY a valid JSON array
2. Mix famous destinations (30%) with hidden gems (70%)
3. Ensure geographic diversity within {region}
4. All cities must be in {region} specifically

'''

        # Use regular strings to avoid f-string conflicts with JSON braces
        region_specific = {
            "Mexico": '''MEXICO FOCUS - COASTAL EMPHASIS: Include Mexican coastal cities like:
- Pacific Coast: Puerto Vallarta, Mazatlán, Acapulco, Huatulco, Zihuatanejo
- Caribbean Coast: Cancún, Playa del Carmen, Tulum, Cozumel, Isla Mujeres
- Gulf Coast: Veracruz, Tampico
- Colonial coastal: Campeche
- Plus some inland gems: Oaxaca, Guanajuato, San Luis Potosí

Do NOT include US or Canadian cities. ONLY Mexican cities, with emphasis on coastal destinations.

Example format:
[
  {"city": "Puerto Vallarta", "country": "Mexico"},
  {"city": "Playa del Carmen", "country": "Mexico"},
  {"city": "Mazatlán", "country": "Mexico"}
]''',

            "South America": '''SOUTH AMERICA FOCUS: Include cities from Brazil, Argentina, Chile, Colombia, Peru, Uruguay, etc.
Examples: Cartagena, Valparaíso, Mendoza, Salvador, Cusco, Montevideo.

Do NOT include Mexico, US, or Central America. ONLY South American cities.''',

            "Europe": '''EUROPE FOCUS: Mix major cities (Barcelona, Rome) with hidden gems (Tallinn, Ljubljana).
Include Western, Eastern, Northern, and Southern Europe for diversity.''',

            "Asia": '''ASIA FOCUS: Include East Asia, Southeast Asia, South Asia, Central Asia.
Examples: Kyoto, Hoi An, Kandy, Almaty, Luang Prabang, Yogyakarta.''',

            "North America": '''NORTH AMERICA FOCUS: Include US, Canada cities.
Examples: San Diego, Miami, Vancouver, Montreal, Seattle, Charleston.
Do NOT include Mexican cities here - they should be in Mexico region.'''
        }

        specific_guidance = region_specific.get(region, f"Focus specifically on {region} region.")

        # Create the final prompt using string concatenation to avoid f-string issues
        final_format = f'''
Return EXACTLY {n} cities in this format:
[
  {{"city": "CityName", "country": "CountryName"}},
  {{"city": "CityName", "country": "CountryName"}}
]

Return ONLY the JSON array, no explanatory text.'''

        return base_instruction + specific_guidance + final_format

    @staticmethod
    def _normalize_region_name(region: str) -> str:
        """Normalize region names for consistency with Mexico emphasis."""
        region_lower = region.lower().strip()

        # Common region mappings with Mexico emphasis
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
            "mexico": "Mexico",
            "mexican": "Mexico",
        }

        return mappings.get(region_lower, region.title())

    async def _coords(self, city: str, country: str) -> Optional[Dict[str, float]]:
        """Enhanced coordinate lookup with agent decision logging."""
        try:
            # Clean up city and country names
            city_clean = city.strip()
            country_clean = country.strip()

            _log_agent_decision("RESEARCHER", f"Looking up coordinates for {city_clean}, {country_clean}")

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
                            coords = {
                                "lat": float(data[0]["lat"]),
                                "lon": float(data[0]["lon"])
                            }
                            _log_agent_decision("RESEARCHER", f"Coordinates found for {city_clean}",
                                                reasoning=f"Lat: {coords['lat']:.2f}, Lon: {coords['lon']:.2f}")
                            return coords
                    else:
                        _log_agent_decision("RESEARCHER", f"Geocoding HTTP error {r.status} for {city}, {country}")
        except Exception as e:
            _log_agent_decision("RESEARCHER", f"Geocoding failed for {city}, {country}: {e}")
        return None

    # ------------------------------------------------------------------
    # Enhanced agent workflow with comprehensive decision logging
    # ------------------------------------------------------------------

    async def _agent(self, role: AgentRole, q: TravelQuery) -> AgentMessage:
        """Enhanced agent dispatch with comprehensive decision logging."""
        tools: List[str] = []
        meta: Dict[str, Any] = {}
        content = ""

        _log_agent_decision(role.value.upper(), f"Agent {role.value} starting",
                            reasoning=f"Processing query for {', '.join(q.regions)} with {q.temperature_range[0]}-{q.temperature_range[1]}°C")

        if role == AgentRole.PLANNER:
            self._update_progress("Planner", "Creating strategy",
                                  details="Analyzing requirements and planning approach")

            lo, hi = self._safe_temp_range(q)
            days_ahead = (q.forecast_date_parsed - _today()).days

            _log_agent_decision("PLANNER", "Strategy analysis complete",
                                reasoning=f"Target temp range: {lo}-{hi}°C, Days ahead: {days_ahead}")

            content = f"""🎯 **ENHANCED PLANNING STRATEGY**

**Target Criteria:**
- Temperature Range: {lo}°C - {hi}°C  
- Travel Date: {q.forecast_date} ({q.forecast_date_parsed.strftime('%Y-%m-%d')})
- Regions: {', '.join(q.regions)}
- Origin: {q.origin_city}
- Days ahead: {days_ahead}
- Special focus: {'Coastal destinations' if 'coastal' in q.additional_criteria else 'Diverse destinations'}

**Enhanced Research Plan:**
1. Generate diverse cities (famous + hidden gems) - OpenAI EXCLUSIVE
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

            tools = ["enhanced_planning", "climate_intelligence_integration", "mexico_recognition"]

        elif role == AgentRole.RESEARCHER:
            content = "🔍 **ENHANCED RESEARCH WITH CLIMATE INTELLIGENCE**\n\n"

            # Reset climate alerts for new research
            self.climate_alerts.clear()

            try:
                # 1. Generate diverse cities with OpenAI EXCLUSIVE
                self._update_progress("Researcher", "Generating cities", details="Finding diverse destinations")
                _log_agent_decision("RESEARCHER", "Starting OpenAI-exclusive city generation",
                                    reasoning=f"Targeting regions: {q.regions}")

                cities = await self._cities_for_regions(q.regions, 8)
                content += f"📍 Generated {len(cities)} diverse cities across {len(q.regions)} regions\n\n"

                city_names = [f"{c['city']}, {c['country']}" for c in cities[:3]]
                more_indicator = "..." if len(cities) > 3 else ""
                _log_agent_decision("RESEARCHER", f"City generation successful",
                                    reasoning=f"Generated {len(cities)} cities: {city_names}{more_indicator}")

                weather_results: List[Dict[str, Any]] = []
                climate_alerts_count = 0

                # 2. Enhanced weather + climate analysis for each city
                content += f"🌡️ **ENHANCED WEATHER ANALYSIS** for {q.forecast_date}...\n\n"

                for i, item in enumerate(cities, 1):
                    city_name = item["city"]
                    country_name = item["country"]

                    self._update_progress("Researcher", "Analyzing weather", i, len(cities),
                                          f"Processing {city_name}, {country_name}")

                    _log_agent_decision("RESEARCHER", f"Starting analysis for {city_name}, {country_name}",
                                        reasoning=f"City {i}/{len(cities)}")

                    content += f"   🏙️ **{i}. {city_name}, {country_name}**\n"

                    # Get coordinates
                    coords = await self._coords(city_name, country_name)
                    if not coords:
                        content += f"   ❌ Could not geocode location\n\n"
                        _log_agent_decision("RESEARCHER", f"Skipping {city_name} - no coordinates")
                        continue

                    content += f"   📍 Coordinates: {coords['lat']:.2f}, {coords['lon']:.2f}\n"

                    # Get weather forecast using FIXED weather API with One Call API 3.0
                    wx = await get_weather_data_enhanced(
                        coords["lat"], coords["lon"], q.forecast_date_parsed,
                        f"{city_name}, {country_name}", self.openweather_key
                    )
                    if not wx.success:
                        content += f"   ❌ Weather forecast failed: {wx.error[:80]}...\n\n"
                        _log_agent_decision("RESEARCHER", f"Weather forecast failed for {city_name}: {wx.error}")
                        continue

                    # Enhanced climate analysis
                    baseline = await self.climate.get_historical_climate_baseline(
                        city_name, country_name, q.forecast_date_parsed.month
                    )

                    deviation = await self.climate.analyse_deviation(wx.data, baseline)

                    # LOG CRITICAL CLIMATE DECISIONS
                    temp_avg = wx.data["temp_avg"]
                    lo, hi = self._safe_temp_range(q)
                    meets_criteria = lo <= temp_avg <= hi

                    _log_climate_decision(
                        city_name,
                        wx.data,
                        baseline,
                        f"{'MEETS' if meets_criteria else 'OUTSIDE'} criteria ({temp_avg:.1f}°C vs {lo}-{hi}°C target)"
                    )

                    # Display weather + climate analysis
                    content += f"   🌡️ Forecast: {temp_avg:.1f}°C, {wx.data.get('weather_desc', 'variable')}\n"
                    content += f"   📊 Historical: {baseline.avg_temp_mean:.1f}°C ({baseline.climate_zone})\n"

                    # Climate deviation analysis
                    if deviation.is_climate_alert:
                        alert_icon = deviation.get_alert_icon()
                        temp_icon = deviation.get_temperature_trend_icon()
                        content += f"   {alert_icon} **CLIMATE ALERT**: {deviation.deviation_severity.upper()} deviation\n"
                        content += f"   {temp_icon} {deviation.temperature_deviation_c:+.1f}°C from normal\n"
                        content += f"   💡 Impact: {deviation.traveler_impact[:60]}...\n"

                        _log_agent_decision("CLIMATE", f"ALERT generated for {city_name}",
                                            reasoning=f"{deviation.deviation_severity} deviation: {deviation.temperature_deviation_c:+.1f}°C")

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

                # Step 3: Enhanced Flight Analysis (abbreviated for brevity)
                self._update_progress("Researcher", "Analyzing flights",
                                      details="Getting flight data and recommendations")
                content += f"\n✈️ **ENHANCED FLIGHT ANALYSIS** from {q.origin_city}...\n\n"

                _log_agent_decision("FLIGHT-ANALYZER",
                                    f"Starting flight analysis for {len(weather_results)} destinations")

                enhanced_flight_results = []
                fr24_successful_calls = 0
                fr24_failed_calls = 0

                if self.flight_analyzer:
                    if self.flight_analyzer.is_fr24_available():
                        content += f"📡 FlightRadar24 API: Connected ✅ (Real flight data available)\n"
                    else:
                        content += f"📡 FlightRadar24 API: Not configured ⚠️ (Using enhanced geographic estimates)\n"
                    content += f"\n"

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

                                # Display FlightRadar24 status and recommendations
                                if flight_analysis.get('real_flight_data_available'):
                                    fr24_successful_calls += 1
                                    recommendations = flight_analysis.get('recommendations', [])
                                    content += f"   ✅ **REAL FLIGHT DATA**: {len(recommendations)} live recommendations\n"

                                    if recommendations:
                                        best_flight = recommendations[0]
                                        content += f"   🥇 Best option: {best_flight.airline} {best_flight.flight_number}\n"
                                        content += f"       ⏱️ Duration: {best_flight.duration_display} | Score: {best_flight.overall_score:.1f}/10\n"
                                else:
                                    fr24_failed_calls += 1
                                    content += f"   📊 Using enhanced geographic estimates\n"
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
                else:
                    # No flight analyzer available
                    enhanced_flight_results = weather_results
                    content += f"📊 Flight analysis module not available - weather data only\n\n"

                # Enhanced research summary
                content += f"✅ **ENHANCED RESEARCH COMPLETE**\n"
                content += f"- Successfully analyzed: {len(enhanced_flight_results)}/{len(cities)} destinations\n"
                content += f"- Climate alerts generated: {climate_alerts_count}\n"
                content += f"- Flight analysis: {len(enhanced_flight_results)} destinations\n"
                content += f"- Enhanced climate intelligence: ✅ Active\n"
                content += f"- Enhanced Mexico recognition: ✅ Active\n"

                _log_agent_decision("RESEARCHER", "Research phase complete",
                                    reasoning=f"Analyzed {len(enhanced_flight_results)} destinations, {climate_alerts_count} climate alerts")

                if self.flight_analyzer and self.flight_analyzer.is_fr24_available():
                    content += f"\n✈️ **FLIGHT ANALYSIS SUMMARY:**\n"
                    content += f"   - Real flight data calls: {fr24_successful_calls} successful, {fr24_failed_calls} fallback\n"
                    content += f"   - FlightRadar24 integration: {'✅ Active' if fr24_successful_calls > 0 else '⚠️ Fallback mode'}\n"
                elif self.flight_analyzer:
                    content += f"\n✈️ **FLIGHT ANALYSIS SUMMARY:**\n"
                    content += f"   - Enhanced geographic estimates: ✅ All destinations\n"
                else:
                    content += f"\n✈️ **FLIGHT ANALYSIS SUMMARY:**\n"
                    content += f"   - Flight analysis module: ⚠️ Not available\n"

                if climate_alerts_count > 0:
                    content += f"\n🚨 **{climate_alerts_count} CLIMATE ALERTS** detected - unusual weather expected!\n"

                meta = {
                    "weather_results": weather_results,
                    "enhanced_flight_results": enhanced_flight_results,
                    "climate_alerts": self.climate_alerts,
                    "cities_processed": len(cities),
                    "successful_analyses": len(enhanced_flight_results),
                    "climate_intelligence_active": True,
                    "flight_intelligence_active": bool(self.flight_analyzer),
                    "fr24_successful_calls": fr24_successful_calls,
                    "fr24_failed_calls": fr24_failed_calls,
                    "fr24_available": bool(self.flight_analyzer and self.flight_analyzer.is_fr24_available()),
                    "mexico_recognition_active": True
                }
                tools = ["enhanced_weather_analysis", "climate_intelligence", "deviation_analysis", "fixed_apis",
                         "enhanced_flight_analysis", "fr24_integration", "mexico_coastal_focus"]

            except Exception as e:
                content += f"\n❌ Enhanced research failed: {str(e)}"
                _log_agent_decision("RESEARCHER", f"CRITICAL ERROR: {e}", reasoning="Research phase failed")
                import traceback
                print(f"Research error: {traceback.format_exc()}")
                meta = {"error": str(e), "climate_intelligence_active": False, "mexico_recognition_active": False}
                tools = ["error_handling"]

        elif role == AgentRole.ANALYZER:
            self._update_progress("Analyzer", "Climate-aware analysis",
                                  details="Scoring destinations with climate intelligence")

            _log_agent_decision("ANALYZER", "Starting climate-aware destination analysis")

            # Enhanced analysis with climate intelligence
            r_messages = [m for m in self.conversation_history if m.agent_role == AgentRole.RESEARCHER]

            if not r_messages or not r_messages[-1].metadata.get("enhanced_flight_results"):
                content = "❌ No enhanced research data available for analysis"
                tools = ["error_handling"]
                _log_agent_decision("ANALYZER", "CRITICAL: No research data available")
            else:
                research_meta = r_messages[-1].metadata
                flight_results = research_meta["enhanced_flight_results"]
                climate_alerts = research_meta.get("climate_alerts", [])
                fr24_available = research_meta.get("fr24_available", False)
                fr24_successful = research_meta.get("fr24_successful_calls", 0)

                content = "📊 **ENHANCED CLIMATE & FLIGHT ANALYSIS**\n\n"

                lo, hi = self._safe_temp_range(q)
                scored_destinations: List[Dict[str, Any]] = []

                _log_agent_decision("ANALYZER", f"Scoring {len(flight_results)} destinations",
                                    reasoning=f"Target temperature range: {lo}-{hi}°C")

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
                        flight_score = max(3, 8 - distance / 2000)

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

                    _log_agent_decision("ANALYZER", f"Scoring {flight_result['city']}: {overall_score:.1f}/10",
                                        reasoning=f"Climate: {climate_score:.1f}, Flight: {flight_score:.1f}, Meets criteria: {meets_criteria}")

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

                _log_agent_decision("ANALYZER", "Scoring complete",
                                    reasoning=f"Top destination: {scored_destinations[0]['city']} ({scored_destinations[0]['overall_score']:.1f}/10)")

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
                    "flight_enhanced_analysis": True,
                    "mexico_recognition_active": True
                }
                tools = ["enhanced_climate_analysis", "deviation_scoring", "climate_alerts", "enhanced_flight_analysis",
                         "fr24_integration", "mexico_coastal_focus"]

        elif role == AgentRole.SYNTHESIZER:
            self._update_progress("Synthesizer", "Generating recommendations",
                                  details="Creating itineraries and final recommendations")

            _log_agent_decision("SYNTHESIZER", "Starting comprehensive recommendation synthesis")

            # Enhanced synthesis with weather-appropriate itineraries
            a_messages = [m for m in self.conversation_history if m.agent_role == AgentRole.ANALYZER]

            if not a_messages or not a_messages[-1].metadata.get("scored_destinations"):
                content = "❌ No enhanced analysis data available for synthesis"
                tools = ["error_handling"]
                _log_agent_decision("SYNTHESIZER", "CRITICAL: No analysis data available")
            else:
                analysis_meta = a_messages[-1].metadata
                destinations = analysis_meta["scored_destinations"]
                climate_alerts = analysis_meta.get("climate_alerts", [])

                content = "🎯 **COMPREHENSIVE RECOMMENDATIONS: CLIMATE + FLIGHT + ITINERARIES**\n\n"

                if destinations:
                    top_3 = destinations[:3]

                    _log_agent_decision("SYNTHESIZER", f"Generating recommendations for top {len(top_3)} destinations",
                                        reasoning=f"Top choice: {top_3[0]['city']} ({top_3[0]['overall_score']:.1f}/10)")

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

                            _log_agent_decision("SYNTHESIZER",
                                                f"Generating weather-appropriate itinerary for {city_name}")

                            content += f"\n   🗓️ **4-DAY WEATHER-APPROPRIATE ITINERARY:**\n"
                            try:
                                if self.itinerary_generator:
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

                                    _log_agent_decision("SYNTHESIZER", f"Itinerary generated for {city_name}",
                                                        reasoning=f"4-day weather-appropriate plan created")
                                else:
                                    content += f"   ⚠️ Itinerary generator not available - module not installed\n"
                                    content += f"   💡 Basic suggestions: Explore outdoor attractions in good weather ({dest['temp_avg']:.1f}°C)\n"

                            except Exception as e:
                                content += f"   ❌ Itinerary generation failed: {str(e)[:100]}...\n"
                                _log_agent_decision("SYNTHESIZER", f"Itinerary generation failed for {city_name}: {e}")

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
                    content += f"- Focus: {'Coastal destinations' if 'coastal' in q.additional_criteria else 'Diverse destinations'}\n"
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

                    _log_agent_decision("SYNTHESIZER", "Comprehensive recommendations complete",
                                        reasoning=f"{len(top_3)} destinations with {itineraries_generated} itineraries")

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
                        "comprehensive_analysis": True,
                        "mexico_recognition_active": True
                    }
                    tools = ["comprehensive_synthesis", "weather_itinerary_generation", "enhanced_flight_analysis",
                             "climate_recommendations", "fr24_integration", "traveler_impact_analysis",
                             "mexico_coastal_focus"]
                else:
                    content = "❌ No destinations found meeting criteria\n\n"
                    content += "**Enhanced Suggestions with Flight Intelligence:**\n"
                    content += "- Consider adjusting temperature range (current weather patterns may be unusual)\n"
                    content += "- Try alternative dates (climate deviations may be temporary)\n"
                    content += "- Expand regions (some areas may have better climate stability)\n"
                    content += "- Consider nearby airports (flight routing may offer alternatives)\n"
                    content += "- Use broader criteria (comprehensive analysis shows few perfect matches)\n"
                    meta = {"error": "no_destinations", "climate_enhanced": True, "flight_enhanced": True,
                            "mexico_recognition_active": True}
                    tools = ["climate_aware_fallback", "flight_aware_fallback", "enhanced_suggestions",
                             "mexico_suggestions"]

        # Create enhanced agent message
        msg = AgentMessage(
            agent_role=role,
            content=content,
            timestamp=_today(),
            tools_used=tools,
            metadata=meta,
        )
        self.conversation_history.append(msg)

        _log_agent_decision(role.value.upper(), f"Agent {role.value} completed",
                            reasoning=f"Tools used: {', '.join(tools[:3])}{'...' if len(tools) > 3 else ''}")

        return msg

    # ------------------------------------------------------------------
    # Enhanced public workflow driver with comprehensive logging
    # ------------------------------------------------------------------

    async def run_agentic_workflow(self, user_query: str) -> List[AgentMessage]:
        """Enhanced workflow with comprehensive agent decision logging."""
        self.conversation_history.clear()
        self.climate_alerts.clear()

        _log_agent_decision("WORKFLOW", "Starting enhanced agentic workflow",
                            reasoning=f"Query: '{user_query[:50]}...'")

        try:
            # Enhanced query parsing with Mexico recognition
            q = await self.parse_travel_query(user_query)

            # Run enhanced four-agent workflow with decision tracking
            agents = [
                (AgentRole.PLANNER, "Strategy"),
                (AgentRole.RESEARCHER, "Data Collection"),
                (AgentRole.ANALYZER, "Analysis"),
                (AgentRole.SYNTHESIZER, "Recommendations"),
            ]

            for i, (role, phase) in enumerate(agents, 1):
                self._update_progress("Workflow", f"Agent {i}/4", i, 4, f"{phase} ({role.value})")
                _log_agent_decision("WORKFLOW", f"Starting {role.value} agent", reasoning=f"Phase {i}/4: {phase}")
                await self._agent(role, q)

            self._update_progress("Workflow", "Complete", 4, 4, "All agents finished")
            _log_agent_decision("WORKFLOW", "Enhanced agentic workflow completed successfully",
                                reasoning=f"All 4 agents completed, {len(self.conversation_history)} messages generated")

            return list(self.conversation_history)

        except Exception as e:
            # Enhanced error handling
            _log_agent_decision("WORKFLOW", f"CRITICAL WORKFLOW ERROR: {e}", reasoning="Workflow failed")
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
        _log_agent_decision("SYSTEM", "Session data cleared")


# ---------------------------------------------------------------------------
# Enhanced CLI test runner with decision tracking
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query_txt = (" ".join(sys.argv[1:]) or
                 "Find warm Mexican coastal cities next Friday between 25-30°C from Toronto with detailed analysis")

    if not OPENAI_KEY:
        print("❌ OPENAI_KEY required for testing")
        sys.exit(1)

    core = AgenticAISystem(OPENAI_KEY, OPENWEATHER_KEY, FLIGHTRADAR24_KEY)


    def progress_callback(step: ProgressStep):
        print(f"⏳ {step}")


    core.set_progress_callback(progress_callback)


    async def _main():
        print(f"🔍 Testing enhanced workflow with Mexico recognition: {query_txt}\n")
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