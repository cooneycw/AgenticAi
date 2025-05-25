# src_code/climate_analyzer.py
"""Enhanced Climate & Weather Intelligence Layer - Phase 1 Upgrade - FIXED

This significantly enhanced module provides comprehensive climate analysis:

1. **Rich Historical Baselines**: Detailed AI-powered climate normals with
   extensive metadata (sunshine hours, record temps, seasonal notes)
2. **Advanced Deviation Analysis**: Multi-factor severity scoring with
   traveler-specific impact assessment
3. **Intelligent Context Generation**: AI-powered explanations of climate
   anomalies and practical travel advice
4. **Robust Fallback Systems**: Multiple fallback layers for reliability
5. **Climate Alerts**: Automatic flagging of unusual conditions

FIXES:
- Better JSON parsing with content extraction
- More robust error handling for API failures
- Improved fallback mechanisms
- Better response validation
"""
from __future__ import annotations

import json
import re
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
    """Extract JSON from AI response that might contain additional text."""
    if not response_text or not response_text.strip():
        return None

    # Try direct parsing first
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Look for JSON block in the response
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested JSON
        r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
        r'```\s*(\{.*?\})\s*```',  # JSON in generic code blocks
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Try extracting the first complete JSON object
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
                        return json.loads(response_text[json_start:i + 1])
                    except json.JSONDecodeError:
                        break

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
            return self._climate_cache[cache_key]

        month_name = _month_name(target_month)

        # Enhanced system prompt for richer climate data
        system_prompt = """You are a climate data expert with access to comprehensive 
historical weather data spanning 30+ years. Provide detailed, accurate climate 
information for specific cities and months.

CRITICAL: Return ONLY a valid JSON object with these EXACT fields:
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

Do NOT include any explanatory text before or after the JSON. Return ONLY the JSON object.
Base this on actual long-term climate patterns. Be precise and realistic."""

        user_prompt = f"""Provide comprehensive historical climate baseline for {city}, {country} in {month_name}.

Return ONLY the JSON object with all required fields. No additional text."""

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.1
            )

            response_text = response.choices[0].message.content
            print(f"DEBUG: Climate AI response for {city}: {response_text[:100]}...")

            climate_data = _extract_json_from_response(response_text)

            if not climate_data:
                print(f"Climate AI returned invalid JSON for {city}: Could not extract JSON from response")
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

            # Add enhanced metadata attributes
            baseline.seasonal_notes = climate_data.get('seasonal_notes', '')
            baseline.record_high = float(climate_data.get('record_high', 30.0))
            baseline.record_low = float(climate_data.get('record_low', 0.0))
            baseline.sunshine_hours = float(climate_data.get('sunshine_hours_per_day', 5.0))
            baseline.wind_speed_kmh = float(climate_data.get('wind_speed_kmh', 10.0))
            baseline.data_reliability = climate_data.get('data_reliability', 'moderate')

            # Validate and cache
            self._validate_baseline(baseline)
            self._climate_cache[cache_key] = baseline
            print(f"DEBUG: Successfully generated climate baseline for {city}")
            return baseline

        except Exception as e:
            print(f"Climate baseline lookup failed for {city}: {e}")
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
                print(f"Context generation failed: {e}")
                context_analysis = {
                    'context': f"Conditions {severity}, {temp_deviation:+.1f}°C from historical average",
                    'impact': "Pack flexible clothing and monitor weather forecasts daily",
                    'confidence': "low"
                }

            return WeatherDeviation(
                temperature_deviation_c=temp_deviation,
                humidity_deviation_pct=humidity_deviation,
                deviation_severity=severity,
                deviation_context=context_analysis['context'],
                traveler_impact=context_analysis['impact'],
                confidence_level=context_analysis['confidence']
            )

        except Exception as e:
            print(f"Deviation analysis failed: {e}")
            return WeatherDeviation(
                temperature_deviation_c=0.0,
                humidity_deviation_pct=0.0,
                deviation_severity="unknown",
                deviation_context=f"Analysis failed: {str(e)}",
                traveler_impact="Unknown impact - check current conditions before travel",
                confidence_level="low"
            )

    # -----------------------------------------------------------------------
    # Enhanced Internal Methods
    # -----------------------------------------------------------------------

    def _enhanced_fallback_baseline(
            self, city: str, country: str, month: int
    ) -> ClimateBaseline:
        """Enhanced fallback with better regional estimates."""
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
            climate_zone="unknown",
            data_source="enhanced_fallback_estimate"
        )

        # Add fallback metadata
        baseline.seasonal_notes = f"Fallback seasonal estimate for {_month_name(month)}"
        baseline.record_high = high_temp + 10
        baseline.record_low = low_temp - 10
        baseline.sunshine_hours = 6.0
        baseline.wind_speed_kmh = 12.0
        baseline.data_reliability = "low"

        print(f"DEBUG: Using fallback baseline for {city}: {baseline.avg_temp_mean:.1f}°C")
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
            city_name = baseline.city
            month_name = _month_name(baseline.month)

            system_prompt = """You are an expert travel weather analyst. Explain weather 
deviations from historical norms in practical, actionable terms for travelers.

CRITICAL: Provide ONLY a valid JSON response with exactly these fields:
{
    "context": "Brief explanation of the deviation and likely causes (max 100 chars)",
    "impact": "Specific advice for travelers - clothing, activities, comfort (max 150 chars)", 
    "confidence": "high/medium/low - based on data quality and deviation magnitude"
}

Return ONLY the JSON object. No additional text before or after."""

            deviation_summary = f"""Location: {city_name}
Month: {month_name}
Historical Average: {baseline.avg_temp_mean:.1f}°C, {baseline.avg_humidity:.0f}% humidity
Current Forecast: {forecast.get('temp_avg', 0):.1f}°C, {forecast.get('humidity', 0):.0f}% humidity
Temperature Deviation: {temp_deviation:+.1f}°C from normal
Humidity Deviation: {humidity_deviation:+.0f}% from normal
Severity: {severity}
Climate Zone: {baseline.climate_zone}"""

            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this weather deviation:\n\n{deviation_summary}"}
                ],
                max_tokens=300,
                temperature=0.3
            )

            response_text = response.choices[0].message.content
            context_data = _extract_json_from_response(response_text)

            if context_data:
                return {
                    "context": str(context_data.get("context", "Weather variation from normal"))[:100],
                    "impact": str(context_data.get("impact", "Check current conditions before travel"))[:150],
                    "confidence": str(context_data.get("confidence", "medium"))
                }
            else:
                # Fallback if JSON extraction fails
                return {
                    "context": f"Weather {severity} vs normal ({temp_deviation:+.1f}°C)",
                    "impact": "Pack flexible clothing and monitor weather forecasts daily",
                    "confidence": "medium"
                }

        except Exception as e:
            print(f"Context generation failed: {e}")
            return {
                "context": f"Conditions {severity}, {temp_deviation:+.1f}°C from historical average",
                "impact": "Pack flexible clothing and monitor weather forecasts daily",
                "confidence": "low"
            }

    @staticmethod
    def _validate_baseline(baseline: ClimateBaseline) -> None:
        """Validate baseline data for sanity checks."""
        # Temperature sanity checks
        if not (-50 <= baseline.avg_temp_mean <= 50):
            print(f"Warning: Unusual temperature for {baseline.city}: {baseline.avg_temp_mean}°C")

        # Humidity sanity checks
        if not (0 <= baseline.avg_humidity <= 100):
            baseline.avg_humidity = max(0, min(100, baseline.avg_humidity))

        # Precipitation sanity checks
        if not (0 <= baseline.avg_precipitation_days <= 31):
            baseline.avg_precipitation_days = max(0, min(31, baseline.avg_precipitation_days))

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

    def cache_size(self) -> int:
        """Return current cache size (useful for monitoring)."""
        return len(self._climate_cache)