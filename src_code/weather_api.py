# src_code/weather_api.py
"""Fixed Weather API Integration - ONE CALL API 3.0 IMPLEMENTATION

This module properly implements OpenWeatherMap One Call API 3.0:
1. **Correct One Call API 3.0 endpoints and parameters**
2. **Proper forecast vs historical data handling**
3. **Enhanced error handling and validation**
4. **Fixed Open-Meteo fallback with correct parameters**
5. **Better timeout and retry mechanisms**

FIXED ISSUES:
- Now uses correct One Call API 3.0 endpoint
- Proper handling of daily vs hourly forecast data
- Better error handling for API responses
- Fixed parameter validation and URL construction
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
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
    """Enhanced weather data retrieval using One Call API 3.0 and fallbacks."""

    # Input validation
    if not (-90 <= lat <= 90):
        return TaskResult(False, None, f"Invalid latitude: {lat}")
    if not (-180 <= lon <= 180):
        return TaskResult(False, None, f"Invalid longitude: {lon}")

    # Try OpenWeatherMap One Call API 3.0 first (paid service)
    if openweather_key:
        print(f"DEBUG: Trying OpenWeatherMap One Call API 3.0 for {city_name}")
        try:
            owm_result = await _get_openweathermap_onecall_v3(lat, lon, target_date, openweather_key)
            if owm_result.success:
                print(f"DEBUG: OpenWeatherMap One Call API 3.0 success for {city_name}")
                return owm_result
            else:
                print(f"DEBUG: OpenWeatherMap One Call API 3.0 failed for {city_name}: {owm_result.error}")
        except Exception as e:
            print(f"DEBUG: OpenWeatherMap One Call API 3.0 exception for {city_name}: {e}")

        # Fallback to regular forecast API
        print(f"DEBUG: Trying OpenWeatherMap 5-day forecast API for {city_name}")
        try:
            forecast_result = await _get_openweathermap_forecast(lat, lon, target_date, openweather_key)
            if forecast_result.success:
                print(f"DEBUG: OpenWeatherMap 5-day forecast success for {city_name}")
                return forecast_result
            else:
                print(f"DEBUG: OpenWeatherMap 5-day forecast failed for {city_name}: {forecast_result.error}")
        except Exception as e:
            print(f"DEBUG: OpenWeatherMap 5-day forecast exception for {city_name}: {e}")

    # Fallback to fixed Open-Meteo
    print(f"DEBUG: Trying fixed Open-Meteo API for {city_name}")
    try:
        meteo_result = await _get_openmeteo_fixed(lat, lon, target_date)
        if meteo_result.success:
            print(f"DEBUG: Open-Meteo success for {city_name}")
            return meteo_result
        else:
            print(f"DEBUG: Open-Meteo failed for {city_name}: {meteo_result.error}")
    except Exception as e:
        print(f"DEBUG: Open-Meteo exception for {city_name}: {e}")

    # Final fallback with seasonal estimates
    print(f"DEBUG: Using seasonal estimate for {city_name}")
    return _get_seasonal_fallback(lat, lon, target_date, city_name)


async def _get_openweathermap_onecall_v3(
        lat: float,
        lon: float,
        target_date: datetime,
        api_key: str
) -> TaskResult:
    """Get weather data from OpenWeatherMap One Call API 3.0 (paid service)."""
    try:
        days_ahead = (target_date - datetime.now()).days

        if days_ahead < 0:
            return TaskResult(False, None, "Date in the past")
        elif days_ahead > 8:  # One Call API 3.0 supports up to 8 days
            return TaskResult(False, None, f"Date {days_ahead} days ahead (max 8 for One Call API)")

        # Use One Call API 3.0 - correct endpoint and parameters
        url = "https://api.openweathermap.org/data/3.0/onecall"
        params = {
            'lat': f"{lat:.6f}",
            'lon': f"{lon:.6f}",
            'appid': api_key,
            'units': 'metric',
            'exclude': 'minutely,alerts'  # Exclude minutely data and alerts to reduce response size
        }

        timeout = aiohttp.ClientTimeout(total=30)  # Increased timeout for One Call API
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return _parse_onecall_v3_response(data, target_date, days_ahead)
                elif response.status == 401:
                    error_text = "Invalid API key or One Call API 3.0 not subscribed"
                    return TaskResult(False, None, f"OpenWeatherMap auth error: {error_text}")
                elif response.status == 402:
                    error_text = "Payment required - One Call API 3.0 is a paid service"
                    return TaskResult(False, None, f"OpenWeatherMap payment error: {error_text}")
                elif response.status == 429:
                    error_text = "Rate limit exceeded"
                    return TaskResult(False, None, f"OpenWeatherMap rate limit: {error_text}")
                else:
                    error_text = await response.text()
                    return TaskResult(False, None, f"OpenWeatherMap One Call API HTTP {response.status}: {error_text[:100]}")

    except asyncio.TimeoutError:
        return TaskResult(False, None, "OpenWeatherMap One Call API timeout")
    except aiohttp.ClientError as e:
        return TaskResult(False, None, f"OpenWeatherMap One Call API connection error: {str(e)}")
    except Exception as e:
        return TaskResult(False, None, f"OpenWeatherMap One Call API error: {str(e)}")


def _parse_onecall_v3_response(data: Dict[str, Any], target_date: datetime, days_ahead: int) -> TaskResult:
    """Parse OpenWeatherMap One Call API 3.0 response."""
    try:
        # One Call API 3.0 provides daily forecasts in the 'daily' array
        if 'daily' not in data:
            return TaskResult(False, None, "No daily forecast data in One Call API response")

        daily_forecasts = data['daily']
        if days_ahead >= len(daily_forecasts):
            return TaskResult(False, None, f"Date beyond forecast range (available: {len(daily_forecasts)} days)")

        day = daily_forecasts[days_ahead]

        # Validate required fields
        if 'temp' not in day or 'humidity' not in day:
            return TaskResult(False, None, "Missing required fields in One Call API daily data")

        try:
            # Extract temperature data
            temp_data = day['temp']
            temp_max = float(temp_data['max'])
            temp_min = float(temp_data['min'])
            temp_avg = (temp_max + temp_min) / 2

            # Extract humidity
            humidity = float(day['humidity'])

            # Extract weather description
            weather_desc = "Variable conditions"
            if day.get('weather') and len(day['weather']) > 0:
                weather_desc = day['weather'][0].get('description', 'Variable conditions').title()

            # Additional One Call API 3.0 data we can use
            pressure = day.get('pressure', 1013)
            wind_speed = day.get('wind_speed', 0)
            clouds = day.get('clouds', 50)

            return TaskResult(True, {
                "temp_max": round(temp_max, 1),
                "temp_min": round(temp_min, 1),
                "temp_avg": round(temp_avg, 1),
                "humidity": round(humidity, 0),
                "weather_desc": weather_desc,
                "pressure": round(pressure, 0),
                "wind_speed_mps": round(wind_speed, 1),
                "cloudiness": round(clouds, 0),
                "data_source": "openweathermap_onecall_v3"
            })

        except (KeyError, ValueError, TypeError) as e:
            return TaskResult(False, None, f"Error processing One Call API temperature data: {str(e)}")

    except Exception as e:
        return TaskResult(False, None, f"One Call API parse error: {str(e)}")


async def _get_openweathermap_forecast(
        lat: float,
        lon: float,
        target_date: datetime,
        api_key: str
) -> TaskResult:
    """Get weather data from OpenWeatherMap 5-day forecast API (fallback)."""
    try:
        days_ahead = (target_date - datetime.now()).days

        if days_ahead < 0:
            return TaskResult(False, None, "Date in the past")
        elif days_ahead > 5:
            return TaskResult(False, None, f"Date {days_ahead} days ahead (max 5 for forecast API)")

        # Use 5-day forecast API
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            'lat': f"{lat:.6f}",
            'lon': f"{lon:.6f}",
            'appid': api_key,
            'units': 'metric'
        }

        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return _parse_forecast_response(data, target_date)
                elif response.status == 401:
                    error_text = "Invalid API key"
                    return TaskResult(False, None, f"OpenWeatherMap forecast auth error: {error_text}")
                elif response.status == 429:
                    error_text = "Rate limit exceeded"
                    return TaskResult(False, None, f"OpenWeatherMap forecast rate limit: {error_text}")
                else:
                    error_text = await response.text()
                    return TaskResult(False, None, f"OpenWeatherMap forecast HTTP {response.status}: {error_text[:100]}")

    except asyncio.TimeoutError:
        return TaskResult(False, None, "OpenWeatherMap forecast timeout")
    except aiohttp.ClientError as e:
        return TaskResult(False, None, f"OpenWeatherMap forecast connection error: {str(e)}")
    except Exception as e:
        return TaskResult(False, None, f"OpenWeatherMap forecast error: {str(e)}")


def _parse_forecast_response(data: Dict[str, Any], target_date: datetime) -> TaskResult:
    """Parse OpenWeatherMap 5-day forecast API response."""
    try:
        if 'list' not in data:
            return TaskResult(False, None, "No forecast list in API response")

        target_day = target_date.strftime('%Y-%m-%d')

        # Find data for target date
        day_data = []
        for item in data['list']:
            try:
                item_date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
                if item_date == target_day:
                    day_data.append(item)
            except (KeyError, ValueError, OSError):
                continue

        if not day_data:
            return TaskResult(False, None, "No data for target date in forecast")

        # Calculate daily averages from 3-hour forecasts
        try:
            temps = [item['main']['temp'] for item in day_data if 'main' in item and 'temp' in item['main']]
            humidities = [item['main']['humidity'] for item in day_data if
                          'main' in item and 'humidity' in item['main']]

            if not temps:
                return TaskResult(False, None, "No temperature data found in forecast")

            temp_max = max(temps)
            temp_min = min(temps)
            temp_avg = sum(temps) / len(temps)
            humidity = sum(humidities) / len(humidities) if humidities else 60

            # Get weather description from first forecast item
            weather_desc = "Variable conditions"
            if day_data[0].get('weather') and len(day_data[0]['weather']) > 0:
                weather_desc = day_data[0]['weather'][0].get('description', 'Variable conditions').title()

            return TaskResult(True, {
                "temp_max": round(temp_max, 1),
                "temp_min": round(temp_min, 1),
                "temp_avg": round(temp_avg, 1),
                "humidity": round(humidity, 0),
                "weather_desc": weather_desc,
                "data_source": "openweathermap_forecast"
            })

        except (KeyError, ZeroDivisionError, TypeError):
            return TaskResult(False, None, "Error processing forecast temperature data")

    except Exception as e:
        return TaskResult(False, None, f"Forecast parse error: {str(e)}")


async def _get_openmeteo_fixed(lat: float, lon: float, target_date: datetime) -> TaskResult:
    """Fixed Open-Meteo API call with correct parameters."""
    try:
        days_ahead = (target_date - datetime.now()).days

        if days_ahead < 0 or days_ahead > 16:
            return TaskResult(False, None, f"Date {days_ahead} days ahead (Open-Meteo range: 0-16)")

        ds = target_date.strftime("%Y-%m-%d")

        # Fixed URL with correct parameters
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat:.6f}&longitude={lon:.6f}"
            "&daily=temperature_2m_max,temperature_2m_min,weather_code,relative_humidity_2m_max"
            f"&start_date={ds}&end_date={ds}"
            "&timezone=auto"
        )

        print(f"DEBUG: Fixed Open-Meteo URL: {url}")

        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
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

        try:
            hi = float(daily["temperature_2m_max"][0])
            lo = float(daily["temperature_2m_min"][0])
            avg = (hi + lo) / 2
            weather_code = int(daily.get("weather_code", [0])[0])

            # Get humidity if available, otherwise estimate
            if "relative_humidity_2m_max" in daily and daily["relative_humidity_2m_max"]:
                humidity = float(daily["relative_humidity_2m_max"][0])
            else:
                humidity = _estimate_humidity_from_weather_code(weather_code, lat)

            weather_desc = _weather_code_to_description(weather_code)

            return TaskResult(True, {
                "temp_max": round(hi, 1),
                "temp_min": round(lo, 1),
                "temp_avg": round(avg, 1),
                "humidity": round(humidity, 0),
                "weather_desc": weather_desc,
                "data_source": "open_meteo_fixed"
            })

        except (ValueError, TypeError, IndexError) as e:
            return TaskResult(False, None, f"Data conversion error: {str(e)}")

    except asyncio.TimeoutError:
        return TaskResult(False, None, "Open-Meteo timeout")
    except aiohttp.ClientError as e:
        return TaskResult(False, None, f"Open-Meteo connection error: {str(e)}")
    except Exception as e:
        print(f"DEBUG: Open-Meteo exception: {e}")
        return TaskResult(False, None, f"API error: {str(e)}")


def _get_seasonal_fallback(lat: float, lon: float, target_date: datetime, city_name: str) -> TaskResult:
    """Generate seasonal weather estimate as final fallback."""
    try:
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
            "temp_max": round(temp_avg + 4, 1),
            "temp_min": round(temp_avg - 4, 1),
            "temp_avg": round(temp_avg, 1),
            "humidity": round(humidity, 0),
            "weather_desc": weather_desc,
            "data_source": "seasonal_fallback_estimate"
        })

    except Exception as e:
        # Ultimate fallback
        return TaskResult(True, {
            "temp_max": 20.0,
            "temp_min": 10.0,
            "temp_avg": 15.0,
            "humidity": 60.0,
            "weather_desc": "Variable conditions",
            "data_source": "emergency_fallback"
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


# Test function for debugging
async def test_weather_apis():
    """Test both weather APIs."""
    # Test coordinates (Toronto)
    lat, lon = 43.6777, -79.6248
    target_date = datetime.now() + timedelta(days=3)

    from config.config import OPENWEATHER_KEY

    print("Testing enhanced weather APIs with One Call API 3.0...")

    result = await get_weather_data_enhanced(lat, lon, target_date, "Toronto", OPENWEATHER_KEY)

    if result.success:
        print(f"✅ Success: {result.data['data_source']}")
        print(f"   Temperature: {result.data['temp_avg']:.1f}°C (Range: {result.data['temp_min']:.1f}°C - {result.data['temp_max']:.1f}°C)")
        print(f"   Weather: {result.data['weather_desc']}")
        print(f"   Humidity: {result.data['humidity']:.0f}%")
        if 'pressure' in result.data:
            print(f"   Pressure: {result.data['pressure']:.0f} hPa")
        if 'wind_speed_mps' in result.data:
            print(f"   Wind: {result.data['wind_speed_mps']:.1f} m/s")
    else:
        print(f"❌ Failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(test_weather_apis())