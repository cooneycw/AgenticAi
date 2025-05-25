# src_code/climate_analyzer.py
"""Enhanced Climate & Weather Intelligence Layer - ROBUST JSON PARSING

This significantly enhanced module provides comprehensive climate analysis:

1. **Rich Historical Baselines**: Detailed AI-powered climate normals with
   extensive metadata (sunshine hours, record temps, seasonal notes)
2. **Advanced Deviation Analysis**: Multi-factor severity scoring with
   traveler-specific impact assessment
3. **Intelligent Context Generation**: AI-powered explanations of climate
   anomalies and practical travel advice
4. **Robust Fallback Systems**: Multiple fallback layers for reliability
5. **Climate Alerts**: Automatic flagging of unusual conditions

MAJOR FIXES:
- **Robust JSON parsing**: Multiple extraction methods with validation
- **Better error handling**: Graceful fallbacks for API failures
- **Rate limiting protection**: Better retry mechanisms
- **Content validation**: Ensures AI responses are usable
- **Enhanced fallbacks**: More reliable backup data generation
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional

from openai import OpenAI

# Local shared models & constants
from config.config import (
    ClimateBaseline,
    WeatherDeviation,
)


# ---------------------------------------------------------------------------
# Enhanced utility functions
# ---------------------------------------------------------------------------

def _month_name(month_idx: int) -> str:
    """Return full English month name for 1-based month index."""
    return datetime(2024, month_idx, 1).strftime("%B")


def _seasonal_fallback_temps(month: int, region_hint: str = "") -> tuple[float, float]:
    """Enhanced seasonal temperature estimates based on month and region."""
    # Basic seasonal patterns
    if month in (12, 1, 2):  # Winter
        base_high, base_low = 10.0, 2.0
    elif month in (3, 4, 5):  # Spring
        base_high, base_low = 18.0, 8.0
    elif month in (6, 7, 8):  # Summer
        base_high, base_low = 26.0, 16.0
    else:  # Fall
        base_high, base_low = 16.0, 6.0

    # Regional adjustments
    region_lower = region_hint.lower()
    if any(warm in region_lower for warm in ["spain", "italy", "greece", "mediterranean"]):
        base_high += 3
        base_low += 2
    elif any(cold in region_lower for cold in ["nordic", "scandinavia", "finland", "norway"]):
        base_high -= 4
        base_low -= 5
    elif any(moderate in region_lower for moderate in ["uk", "ireland", "britain"]):
        base_high -= 1
        base_low += 1

    return base_high, base_low


def _extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Robust JSON extraction from AI response with multiple fallback methods."""
    if not response_text or not response_text.strip():
        print("DEBUG: Empty response from AI")
        return None

    response_text = response_text.strip()
    print(f"DEBUG: AI response length: {len(response_text)} chars")

    # Method 1: Direct JSON parsing
    try:
        result = json.loads(response_text)
        print("DEBUG: Direct JSON parsing successful")
        return result
    except json.JSONDecodeError:
        pass

    # Method 2: Extract JSON from code blocks
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
        r'```\s*(\{.*?\})\s*```',  # JSON in generic code blocks
        r'`(\{.*?\})`',  # JSON in single backticks
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match.strip())
                print(f"DEBUG: JSON extracted from code block with pattern: {pattern[:20]}...")
                return result
            except json.JSONDecodeError:
                continue

    # Method 3: Find the first complete JSON object
    brace_count = 0
    json_start = response_text.find('{')
    if json_start >= 0:
        for i, char in enumerate(response_text[json_start:], json_start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        json_str = response_text[json_start:i + 1]
                        result = json.loads(json_str)
                        print("DEBUG: JSON extracted by brace matching")
                        return result
                    except json.JSONDecodeError:
                        break

    # Method 4: Try to extract key-value pairs manually
    try:
        # Look for key patterns in the response
        if 'avg_temp_high' in response_text and 'avg_temp_low' in response_text:
            print("DEBUG: Attempting manual key extraction")
            manual_data = _extract_climate_data_manually(response_text)
            if manual_data:
                return manual_data
    except Exception as e:
        print(f"DEBUG: Manual extraction failed: {e}")

    print("DEBUG: All JSON extraction methods failed")
    return None


def _extract_climate_data_manually(text: str) -> Optional[Dict[str, Any]]:
    """Manually extract climate data from AI response when JSON parsing fails."""
    try:
        data = {}

        # Temperature patterns
        temp_patterns = {
            'avg_temp_high': r'avg_temp_high["\']?\s*:\s*([0-9.-]+)',
            'avg_temp_low': r'avg_temp_low["\']?\s*:\s*([0-9.-]+)',
            'avg_temp_mean': r'avg_temp_mean["\']?\s*:\s*([0-9.-]+)',
            'avg_humidity': r'avg_humidity["\']?\s*:\s*([0-9.-]+)',
            'avg_precipitation_days': r'avg_precipitation_days["\']?\s*:\s*([0-9]+)',
            'record_high': r'record_high["\']?\s*:\s*([0-9.-]+)',
            'record_low': r'record_low["\']?\s*:\s*([0-9.-]+)',
            'sunshine_hours_per_day': r'sunshine_hours_per_day["\']?\s*:\s*([0-9.-]+)',
            'wind_speed_kmh': r'wind_speed_kmh["\']?\s*:\s*([0-9.-]+)',
        }

        for key, pattern in temp_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data[key] = float(match.group(1))

        # String patterns
        string_patterns = {
            'climate_zone': r'climate_zone["\']?\s*:\s*["\']([^"\']+)["\']',
            'seasonal_notes': r'seasonal_notes["\']?\s*:\s*["\']([^"\']+)["\']',
            'data_reliability': r'data_reliability["\']?\s*:\s*["\']([^"\']+)["\']',
        }

        for key, pattern in string_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data[key] = match.group(1)

        # Weather patterns array (simplified)
        weather_match = re.search(r'typical_weather_patterns["\']?\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if weather_match:
            patterns_text = weather_match.group(1)
            patterns = re.findall(r'["\']([^"\']+)["\']', patterns_text)
            if patterns:
                data['typical_weather_patterns'] = patterns
            else:
                data['typical_weather_patterns'] = ["variable conditions"]
        else:
            data['typical_weather_patterns'] = ["variable conditions"]

        # Validate we got essential data
        if 'avg_temp_high' in data and 'avg_temp_low' in data:
            print(f"DEBUG: Manual extraction successful, found {len(data)} fields")
            return data
        else:
            print("DEBUG: Manual extraction failed - missing essential temperature data")
            return None

    except Exception as e:
        print(f"DEBUG: Manual extraction error: {e}")
        return None


# ---------------------------------------------------------------------------
# Enhanced Climate Analyzer
# ---------------------------------------------------------------------------

class EnhancedClimateAnalyzer:
    """Advanced climate intelligence with comprehensive analysis capabilities.

    This enhanced version provides:
    - Detailed historical climate baselines with metadata
    - Multi-factor weather deviation analysis
    - AI-powered traveler impact assessments
    - Robust error handling and fallback systems
    - Climate alert generation for unusual conditions
    """

    def __init__(self, openai_client: OpenAI):
        self._client = openai_client
        self._climate_cache: Dict[str, ClimateBaseline] = {}
        self._last_api_call = 0
        self._min_api_interval = 1.0  # Minimum seconds between API calls

    # -----------------------------------------------------------------------
    # Enhanced Historical Baseline Generation
    # -----------------------------------------------------------------------

    async def get_historical_climate_baseline(
            self,
            city: str,
            country: str,
            target_month: int,
    ) -> ClimateBaseline:
        """Generate comprehensive historical climate baseline using AI.

        This enhanced version provides much richer climate data including:
        - Long-term temperature and humidity averages
        - Typical weather patterns for the specific month
        - Climate zone classification
        - Record temperatures and seasonal notes
        - Sunshine hours and reliability indicators

        Args:
            city: Target city name
            country: Country name
            target_month: Month (1-12) for climate data

        Returns:
            Comprehensive ClimateBaseline with enhanced metadata
        """
        cache_key = f"{city.lower()}_{country.lower()}_{target_month}"
        if cache_key in self._climate_cache:
            print(f"DEBUG: Using cached climate data for {city}")
            return self._climate_cache[cache_key]

        month_name = _month_name(target_month)

        # Rate limiting
        current_time = time.time()
        if current_time - self._last_api_call < self._min_api_interval:
            time.sleep(self._min_api_interval - (current_time - self._last_api_call))

        # Enhanced system prompt for more reliable JSON output
        system_prompt = """You are a climate data expert. Return ONLY a valid JSON object with historical climate data.

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON - no explanatory text before or after
2. Use these EXACT field names
3. All temperatures in Celsius, humidity as percentage

Required JSON format:
{
    "avg_temp_high": 23.5,
    "avg_temp_low": 16.2, 
    "avg_temp_mean": 19.8,
    "avg_humidity": 62,
    "avg_precipitation_days": 8,
    "typical_weather_patterns": ["sunny mornings", "afternoon clouds"],
    "climate_zone": "Mediterranean",
    "seasonal_notes": "Pleasant travel weather",
    "record_high": 35,
    "record_low": 8,
    "sunshine_hours_per_day": 7.5,
    "wind_speed_kmh": 12,
    "data_reliability": "high"
}

Return ONLY the JSON object. Be realistic and accurate."""

        user_prompt = f"""Historical climate data for {city}, {country} in {month_name}.

Return only the JSON object with all required fields."""

        try:
            print(f"DEBUG: Requesting climate data for {city}, {country} in {month_name}")

            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.1,
                timeout=30  # Add timeout
            )

            self._last_api_call = time.time()
            response_text = response.choices[0].message.content

            if not response_text:
                print(f"DEBUG: Empty response from OpenAI for {city}")
                return self._enhanced_fallback_baseline(city, country, target_month)

            print(f"DEBUG: OpenAI response for {city}: {response_text[:100]}...")

            climate_data = _extract_json_from_response(response_text)

            if not climate_data:
                print(f"DEBUG: JSON extraction failed for {city}, using fallback")
                return self._enhanced_fallback_baseline(city, country, target_month)

            # Validate extracted data
            required_fields = ['avg_temp_high', 'avg_temp_low']
            missing_fields = [field for field in required_fields if field not in climate_data]
            if missing_fields:
                print(f"DEBUG: Missing required fields {missing_fields} for {city}, using fallback")
                return self._enhanced_fallback_baseline(city, country, target_month)

            # Create enhanced baseline with validation
            baseline = ClimateBaseline(
                city=f"{city}, {country}",
                month=target_month,
                avg_temp_high=float(climate_data.get('avg_temp_high', 20.0)),
                avg_temp_low=float(climate_data.get('avg_temp_low', 10.0)),
                avg_temp_mean=float(climate_data.get('avg_temp_mean', 15.0)),
                avg_humidity=float(climate_data.get('avg_humidity', 60.0)),
                avg_precipitation_days=int(climate_data.get('avg_precipitation_days', 5)),
                typical_weather_patterns=climate_data.get('typical_weather_patterns', ["variable conditions"]),
                climate_zone=climate_data.get('climate_zone', "temperate"),
                data_source="ai_enhanced_analysis"
            )

            # Ensure avg_temp_mean is calculated if not provided
            if baseline.avg_temp_mean is None or baseline.avg_temp_mean == 0:
                baseline.avg_temp_mean = (baseline.avg_temp_high + baseline.avg_temp_low) / 2

            # Add enhanced metadata attributes
            baseline.seasonal_notes = climate_data.get('seasonal_notes', f'Typical {month_name} conditions')
            baseline.record_high = float(climate_data.get('record_high', baseline.avg_temp_high + 10))
            baseline.record_low = float(climate_data.get('record_low', baseline.avg_temp_low - 10))
            baseline.sunshine_hours = float(climate_data.get('sunshine_hours_per_day', 6.0))
            baseline.wind_speed_kmh = float(climate_data.get('wind_speed_kmh', 12.0))
            baseline.data_reliability = climate_data.get('data_reliability', 'high')

            # Validate and cache
            self._validate_baseline(baseline)
            self._climate_cache[cache_key] = baseline
            print(f"DEBUG: Successfully generated climate baseline for {city}")
            return baseline

        except Exception as e:
            print(f"DEBUG: Climate baseline lookup failed for {city}: {e}")
            return self._enhanced_fallback_baseline(city, country, target_month)

    # -----------------------------------------------------------------------
    # Enhanced Deviation Analysis
    # -----------------------------------------------------------------------

    async def analyse_deviation(
            self,
            forecast: Dict[str, Any],
            baseline: ClimateBaseline,
    ) -> WeatherDeviation:
        """Comprehensive weather deviation analysis with traveler impact.

        This enhanced version provides:
        - Multi-factor severity assessment
        - Detailed traveler impact analysis
        - AI-generated contextual explanations
        - Confidence scoring for reliability

        Args:
            forecast: Current weather forecast data
            baseline: Historical climate baseline for comparison

        Returns:
            Comprehensive WeatherDeviation with enhanced analysis
        """
        try:
            # Extract forecast values with robust fallbacks
            forecast_temp = forecast.get('temp_avg', forecast.get('temp_max', baseline.avg_temp_mean))
            forecast_humidity = forecast.get('humidity', baseline.avg_humidity)

            # Calculate deviations
            temp_deviation = forecast_temp - baseline.avg_temp_mean
            humidity_deviation = forecast_humidity - baseline.avg_humidity

            # Enhanced severity calculation
            severity = self._calculate_enhanced_severity(temp_deviation, humidity_deviation)

            # Generate AI-powered contextual analysis with better error handling
            try:
                context_analysis = await self._generate_enhanced_context(
                    forecast, baseline, temp_deviation, humidity_deviation, severity
                )
            except Exception as e:
                print(f"DEBUG: Context generation failed: {e}")
                context_analysis = self._create_fallback_context(temp_deviation, severity)

            return WeatherDeviation(
                temperature_deviation_c=temp_deviation,
                humidity_deviation_pct=humidity_deviation,
                deviation_severity=severity,
                deviation_context=context_analysis['context'],
                traveler_impact=context_analysis['impact'],
                confidence_level=context_analysis['confidence']
            )

        except Exception as e:
            print(f"DEBUG: Deviation analysis failed: {e}")
            return WeatherDeviation(
                temperature_deviation_c=0.0,
                humidity_deviation_pct=0.0,
                deviation_severity="normal",
                deviation_context="Weather analysis unavailable",
                traveler_impact="Check current conditions before travel",
                confidence_level="low"
            )

    # -----------------------------------------------------------------------
    # Enhanced Internal Methods
    # -----------------------------------------------------------------------

    def _enhanced_fallback_baseline(
            self, city: str, country: str, month: int
    ) -> ClimateBaseline:
        """Enhanced fallback with better regional estimates."""
        print(f"DEBUG: Creating fallback baseline for {city}, {country}")

        region_hint = f"{city} {country}".lower()
        high_temp, low_temp = _seasonal_fallback_temps(month, region_hint)

        baseline = ClimateBaseline(
            city=f"{city}, {country}",
            month=month,
            avg_temp_high=high_temp,
            avg_temp_low=low_temp,
            avg_temp_mean=(high_temp + low_temp) / 2,
            avg_humidity=65.0,
            avg_precipitation_days=6,
            typical_weather_patterns=["seasonal average conditions"],
            climate_zone="temperate",
            data_source="enhanced_fallback_estimate"
        )

        # Add fallback metadata
        baseline.seasonal_notes = f"Fallback seasonal estimate for {_month_name(month)}"
        baseline.record_high = high_temp + 10
        baseline.record_low = low_temp - 10
        baseline.sunshine_hours = 6.0
        baseline.wind_speed_kmh = 12.0
        baseline.data_reliability = "low"

        print(f"DEBUG: Fallback baseline created: {baseline.avg_temp_mean:.1f}°C average")
        return baseline

    def _calculate_enhanced_severity(self, temp_dev: float, humidity_dev: float) -> str:
        """Enhanced severity calculation with more nuanced thresholds."""
        abs_temp_dev = abs(temp_dev)
        abs_humidity_dev = abs(humidity_dev)

        # Enhanced temperature thresholds
        if abs_temp_dev >= 8:  # 8°C+ deviation is extreme
            temp_severity = "extreme"
        elif abs_temp_dev >= 5:  # 5-8°C deviation is significant
            temp_severity = "significant"
        elif abs_temp_dev >= 3:  # 3-5°C deviation is slight
            temp_severity = "slight"
        else:
            temp_severity = "normal"

        # Enhanced humidity thresholds
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

    def _create_fallback_context(self, temp_deviation: float, severity: str) -> Dict[str, str]:
        """Create fallback context when AI analysis fails."""
        if severity == "extreme":
            impact = "Expect very unusual conditions - pack versatile clothing and monitor forecasts"
        elif severity == "significant":
            impact = "Notable weather differences - bring appropriate gear for conditions"
        elif severity == "slight":
            impact = "Minor weather variations - standard seasonal clothing should suffice"
        else:
            impact = "Typical seasonal weather expected - pack normal travel clothes"

        return {
            "context": f"Weather {severity}, {temp_deviation:+.1f}°C from historical average",
            "impact": impact,
            "confidence": "medium"
        }

    async def _generate_enhanced_context(
            self,
            forecast: Dict[str, Any],
            baseline: ClimateBaseline,
            temp_deviation: float,
            humidity_deviation: float,
            severity: str,
    ) -> Dict[str, str]:
        """Generate enhanced AI-powered contextual analysis with better error handling."""
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self._last_api_call < self._min_api_interval:
                time.sleep(self._min_api_interval - (current_time - self._last_api_call))

            city_name = baseline.city
            month_name = _month_name(baseline.month)

            system_prompt = """You are a travel weather expert. Analyze weather deviations and provide practical travel advice.

Return ONLY a valid JSON object with exactly these fields:
{
    "context": "Brief explanation of deviation (max 80 chars)",
    "impact": "Practical travel advice (max 120 chars)", 
    "confidence": "high/medium/low"
}

Return ONLY the JSON object."""

            deviation_summary = f"""Location: {city_name}
Month: {month_name}
Historical Average: {baseline.avg_temp_mean:.1f}°C
Forecast: {forecast.get('temp_avg', 0):.1f}°C
Deviation: {temp_deviation:+.1f}°C
Severity: {severity}"""

            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze: {deviation_summary}"}
                ],
                max_tokens=200,
                temperature=0.3,
                timeout=20
            )

            self._last_api_call = time.time()
            response_text = response.choices[0].message.content

            if not response_text:
                print("DEBUG: Empty context response from AI")
                return self._create_fallback_context(temp_deviation, severity)

            context_data = _extract_json_from_response(response_text)

            if context_data and all(key in context_data for key in ["context", "impact", "confidence"]):
                return {
                    "context": str(context_data["context"])[:80],
                    "impact": str(context_data["impact"])[:120],
                    "confidence": str(context_data["confidence"]) if context_data["confidence"] in ["high", "medium",
                                                                                                    "low"] else "medium"
                }
            else:
                print("DEBUG: Invalid context data from AI, using fallback")
                return self._create_fallback_context(temp_deviation, severity)

        except Exception as e:
            print(f"DEBUG: Context generation failed: {e}")
            return self._create_fallback_context(temp_deviation, severity)

    @staticmethod
    def _validate_baseline(baseline: ClimateBaseline) -> None:
        """Validate baseline data for sanity checks."""
        # Temperature sanity checks
        if not (-50 <= baseline.avg_temp_mean <= 50):
            print(f"DEBUG: Warning - Unusual temperature for {baseline.city}: {baseline.avg_temp_mean}°C")

        # Humidity sanity checks
        if not (0 <= baseline.avg_humidity <= 100):
            baseline.avg_humidity = max(0, min(100, baseline.avg_humidity))

        # Precipitation sanity checks
        if not (0 <= baseline.avg_precipitation_days <= 31):
            baseline.avg_precipitation_days = max(0, min(31, baseline.avg_precipitation_days))

        # Ensure temperature consistency
        if baseline.avg_temp_high < baseline.avg_temp_low:
            baseline.avg_temp_high, baseline.avg_temp_low = baseline.avg_temp_low, baseline.avg_temp_high

    # -----------------------------------------------------------------------
    # Convenience methods for external consumers
    # -----------------------------------------------------------------------

    def is_climate_alert(self, deviation: WeatherDeviation) -> bool:
        """Check if weather deviation warrants a climate alert."""
        return deviation.deviation_severity in ['significant', 'extreme']

    def get_traveler_recommendation(self, deviation: WeatherDeviation) -> str:
        """Get concise traveler recommendation based on deviation."""
        if deviation.deviation_severity == 'extreme':
            return "⚠️ UNUSUAL CONDITIONS - Monitor forecasts closely and pack flexible gear"
        elif deviation.deviation_severity == 'significant':
            return "⚡ Notable weather difference - Check latest forecasts before travel"
        elif deviation.deviation_severity == 'slight':
            return "✅ Slightly different but manageable conditions expected"
        else:
            return "✅ Typical seasonal weather patterns expected"

    def clear_cache(self) -> None:
        """Clear the climate baseline cache (useful for testing)."""
        self._climate_cache.clear()
        print("DEBUG: Climate cache cleared")

    def cache_size(self) -> int:
        """Return current cache size (useful for monitoring)."""
        return len(self._climate_cache)