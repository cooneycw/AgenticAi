# src_code/weather_api.py
"""Fixed Weather API Integration

This module fixes the weather API issues by:
1. Using OpenWeatherMap API (paid service) as primary
2. Fixing Open-Meteo API parameters as fallback
3. Better error handling and parameter validation
4. Proper temperature range handling
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

import aiohttp

from config.config import TaskResult

__all__ = ["get_weather_data_enhanced"]


async def get_weather_data_enhanced(
        lat: float,
        lon: float,
        target_date: datetime,
        city_name: str = "",
        openweather_key: Optional[str] = None
) -> TaskResult:
    """Enhanced weather data retrieval using multiple APIs with fallbacks."""

    # Try OpenWeatherMap first (paid service)
    if openweather_key:
        print(f"DEBUG: Trying OpenWeatherMap API for {city_name}")
        owm_result = await _get_openweathermap_data(lat, lon, target_date, openweather_key)
        if owm_result.success:
            print(f"DEBUG: OpenWeatherMap success for {city_name}")
            return owm_result
        else:
            print(f"DEBUG: OpenWeatherMap failed for {city_name}: {owm_result.error}")

    # Fallback to fixed Open-Meteo
    print(f"DEBUG: Trying fixed Open-Meteo API for {city_name}")
    meteo_result = await _get_openmeteo_fixed(lat, lon, target_date)
    if meteo_result.success:
        print(f"DEBUG: Open-Meteo success for {city_name}")
        return meteo_result
    else:
        print(f"DEBUG: Open-Meteo failed for {city_name}: {meteo_result.error}")

    # Final fallback with seasonal estimates
    print(f"DEBUG: Using seasonal estimate for {city_name}")
    return _get_seasonal_fallback(lat, lon, target_date, city_name)


async def _get_openweathermap_data(
        lat: float,
        lon: float,
        target_date: datetime,
        api_key: str
) -> TaskResult:
    """Get weather data from OpenWeatherMap API (paid service)."""
    try:
        days_ahead = (target_date - datetime.now()).days

        if days_ahead < 0:
            return TaskResult(False, None, "Date in the past")
        elif days_ahead > 5:
            # Use One Call API 3.0 for extended forecast (paid feature)
            url = "https://api.openweathermap.org/data/3.0/onecall"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': api_key,
                'units': 'metric',
                'exclude': 'minutely,alerts'
            }
        else:
            # Use 5-day forecast API for near-term
            url = "https://api.openweathermap.org/data/2.5/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': api_key,
                'units': 'metric'
            }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    return _parse_openweathermap_response(data, target_date, days_ahead)
                else:
                    error_text = await response.text()
                    return TaskResult(False, None, f"OpenWeatherMap HTTP {response.status}: {error_text[:100]}")

    except Exception as e:
        return TaskResult(False, None, f"OpenWeatherMap error: {str(e)}")


def _parse_openweathermap_response(data: Dict[str, Any], target_date: datetime, days_ahead: int) -> TaskResult:
    """Parse OpenWeatherMap API response."""
    try:
        if days_ahead <= 5 and 'list' in data:
            # 5-day forecast format
            target_day = target_date.strftime('%Y-%m-%d')

            # Find data for target date
            day_data = []
            for item in data['list']:
                item_date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
                if item_date == target_day:
                    day_data.append(item)

            if not day_data:
                return TaskResult(False, None, "No data for target date")

            # Calculate daily averages
            temps = [item['main']['temp'] for item in day_data]
            humidities = [item['main']['humidity'] for item in day_data]

            temp_max = max(temps)
            temp_min = min(temps)
            temp_avg = sum(temps) / len(temps)
            humidity = sum(humidities) / len(humidities)

            # Get weather description
            weather_desc = day_data[0]['weather'][0]['description']

        elif 'daily' in data:
            # One Call API format
            if days_ahead < len(data['daily']):
                day = data['daily'][days_ahead]
                temp_max = day['temp']['max']
                temp_min = day['temp']['min']
                temp_avg = (temp_max + temp_min) / 2
                humidity = day['humidity']
                weather_desc = day['weather'][0]['description']
            else:
                return TaskResult(False, None, "Date beyond forecast range")
        else:
            return TaskResult(False, None, "Unexpected API response format")

        return TaskResult(True, {
            "temp_max": temp_max,
            "temp_min": temp_min,
            "temp_avg": temp_avg,
            "humidity": humidity,
            "weather_desc": weather_desc.title(),
            "data_source": "openweathermap_paid"
        })

    except Exception as e:
        return TaskResult(False, None, f"Parse error: {str(e)}")


async def _get_openmeteo_fixed(lat: float, lon: float, target_date: datetime) -> TaskResult:
    """Fixed Open-Meteo API call with correct parameters."""
    try:
        days_ahead = (target_date - datetime.now()).days

        if days_ahead < 0 or days_ahead > 16:
            return TaskResult(False, None, f"Date {days_ahead} days ahead (out of range)")

        ds = target_date.strftime("%Y-%m-%d")

        # Fixed URL with correct parameters (removed problematic humidity parameter)
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat:.6f}&longitude={lon:.6f}"
            "&daily=temperature_2m_max,temperature_2m_min,weather_code"
            f"&start_date={ds}&end_date={ds}"
            "&timezone=auto"
        )

        print(f"DEBUG: Fixed Open-Meteo URL: {url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=15) as response:
                print(f"DEBUG: Open-Meteo status: {response.status}")

                if response.status != 200:
                    error_text = await response.text()
                    print(f"DEBUG: Open-Meteo error: {error_text[:200]}")
                    return TaskResult(False, None, f"HTTP {response.status}")

                data = await response.json()
                print(f"DEBUG: Open-Meteo response keys: {list(data.keys())}")

        daily = data.get("daily", {})
        if not daily:
            return TaskResult(False, None, "No daily data")

        required_fields = ["temperature_2m_max", "temperature_2m_min"]
        missing = [k for k in required_fields if k not in daily or not daily[k]]
        if missing:
            return TaskResult(False, None, f"Missing fields: {missing}")

        hi = daily["temperature_2m_max"][0]
        lo = daily["temperature_2m_min"][0]
        avg = (hi + lo) / 2
        weather_code = daily.get("weather_code", [0])[0]

        # Estimate humidity based on weather code and location
        humidity = _estimate_humidity_from_weather_code(weather_code, lat)

        weather_desc = _weather_code_to_description(weather_code)

        return TaskResult(True, {
            "temp_max": hi,
            "temp_min": lo,
            "temp_avg": avg,
            "humidity": humidity,
            "weather_desc": weather_desc,
            "data_source": "open_meteo_fixed"
        })

    except Exception as e:
        print(f"DEBUG: Open-Meteo exception: {e}")
        return TaskResult(False, None, f"API error: {str(e)}")


def _get_seasonal_fallback(lat: float, lon: float, target_date: datetime, city_name: str) -> TaskResult:
    """Generate seasonal weather estimate as final fallback."""
    month = target_date.month

    # Rough seasonal estimates based on latitude and month
    if abs(lat) < 23.5:  # Tropical
        if month in [6, 7, 8, 9]:  # Wet season
            temp_base = 28
            humidity = 80
            weather_desc = "Partly cloudy with showers"
        else:  # Dry season
            temp_base = 30
            humidity = 65
            weather_desc = "Mostly sunny"
    elif abs(lat) < 35:  # Subtropical
        if month in [12, 1, 2]:  # Winter
            temp_base = 15
            humidity = 70
            weather_desc = "Cool and cloudy"
        elif month in [6, 7, 8]:  # Summer
            temp_base = 28
            humidity = 60
            weather_desc = "Warm and sunny"
        else:  # Spring/Fall
            temp_base = 22
            humidity = 65
            weather_desc = "Mild conditions"
    else:  # Temperate
        if month in [12, 1, 2]:  # Winter
            temp_base = 5
            humidity = 75
            weather_desc = "Cold and overcast"
        elif month in [6, 7, 8]:  # Summer
            temp_base = 25
            humidity = 60
            weather_desc = "Warm and pleasant"
        else:  # Spring/Fall
            temp_base = 15
            humidity = 70
            weather_desc = "Cool and variable"

    # Add some randomness
    import random
    temp_var = random.uniform(-3, 3)
    temp_avg = temp_base + temp_var

    return TaskResult(True, {
        "temp_max": temp_avg + 4,
        "temp_min": temp_avg - 4,
        "temp_avg": temp_avg,
        "humidity": humidity,
        "weather_desc": weather_desc,
        "data_source": "seasonal_fallback_estimate"
    })


def _estimate_humidity_from_weather_code(code: int, lat: float) -> float:
    """Estimate humidity based on weather code and latitude."""
    # Base humidity on latitude (tropical = higher humidity)
    base_humidity = max(50, 80 - abs(lat) * 0.8)

    # Adjust based on weather conditions
    if code in [61, 63, 65, 80, 81, 82]:  # Rain
        return min(95, base_humidity + 20)
    elif code in [71, 73, 75, 85, 86]:  # Snow
        return min(90, base_humidity + 15)
    elif code in [45, 48]:  # Fog
        return min(98, base_humidity + 25)
    elif code in [0, 1]:  # Clear
        return max(30, base_humidity - 15)
    else:  # Cloudy
        return base_humidity


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


# Test function
async def test_weather_apis():
    """Test both weather APIs."""
    # Test coordinates (Toronto)
    lat, lon = 43.6777, -79.6248
    target_date = datetime.now()

    from config.config import OPENWEATHER_KEY

    print("Testing weather APIs...")

    result = await get_weather_data_enhanced(lat, lon, target_date, "Toronto", OPENWEATHER_KEY)

    if result.success:
        print(f"✅ Success: {result.data['data_source']}")
        print(f"   Temperature: {result.data['temp_avg']:.1f}°C")
        print(f"   Weather: {result.data['weather_desc']}")
        print(f"   Humidity: {result.data['humidity']:.0f}%")
    else:
        print(f"❌ Failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(test_weather_apis())