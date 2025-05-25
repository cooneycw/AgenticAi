# src_code/itinerary_generator.py
"""Weather-Appropriate Itinerary Generator - Phase 2

This module generates detailed, weather-specific 4-day itineraries for travel destinations.
It's the key differentiating feature that makes the agentic travel system truly valuable.

Key Features:
1. **Weather-Adaptive Activities**: Indoor/outdoor balance based on conditions
2. **Time-of-Day Optimization**: Best times for outdoor activities
3. **Clothing & Gear Recommendations**: Specific to weather conditions
4. **Backup Plans**: Indoor alternatives for each day
5. **Local Context**: City-specific attractions and weather considerations
6. **Climate Intelligence**: Integrates with enhanced climate analysis

The itineraries are AI-generated with detailed weather context and practical advice.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config.config import WeatherDeviation, ClimateBaseline

__all__ = ["WeatherAppropriateItineraryGenerator"]

# ---------------------------------------------------------------------------
# Activity categories and weather mappings
# ---------------------------------------------------------------------------

WEATHER_ACTIVITY_MAPPING = {
    "sunny": {
        "outdoor": ["parks", "walking tours", "outdoor markets", "rooftop bars", "gardens", "beaches", "hiking"],
        "indoor": ["museums", "galleries", "shopping", "cafes", "restaurants"],
        "best_times": ["morning", "late_afternoon", "evening"],
        "avoid_times": ["midday"]  # too hot/bright
    },
    "partly_cloudy": {
        "outdoor": ["city walks", "architecture tours", "parks", "outdoor dining", "markets", "photography"],
        "indoor": ["museums", "galleries", "cultural sites", "cafes"],
        "best_times": ["all_day"],
        "avoid_times": []
    },
    "cloudy": {
        "outdoor": ["walking tours", "markets", "outdoor dining", "city exploration"],
        "indoor": ["museums", "galleries", "shopping", "cultural experiences", "spas"],
        "best_times": ["all_day"],
        "avoid_times": []
    },
    "rainy": {
        "outdoor": ["covered markets", "short walks between indoor venues"],
        "indoor": ["museums", "galleries", "shopping malls", "restaurants", "spas", "cooking classes", "theaters"],
        "best_times": ["between_showers"],
        "avoid_times": ["heavy_rain_periods"]
    },
    "hot": {
        "outdoor": ["early morning walks", "evening activities", "water activities", "shaded areas"],
        "indoor": ["air-conditioned museums", "shopping centers", "restaurants", "cinemas"],
        "best_times": ["early_morning", "evening"],
        "avoid_times": ["midday", "afternoon"]
    },
    "cold": {
        "outdoor": ["winter sports", "quick sightseeing", "hot drink stops"],
        "indoor": ["museums", "galleries", "warm cafes", "shopping", "thermal baths", "cozy restaurants"],
        "best_times": ["midday"],
        "avoid_times": ["early_morning", "late_evening"]
    }
}

TEMPERATURE_CATEGORIES = {
    "hot": lambda t: t > 28,
    "warm": lambda t: 22 <= t <= 28,
    "mild": lambda t: 15 <= t <= 21,
    "cool": lambda t: 8 <= t <= 14,
    "cold": lambda t: t < 8
}

CLOTHING_RECOMMENDATIONS = {
    "hot": {
        "essentials": ["light cotton clothing", "shorts", "t-shirts", "sundresses"],
        "accessories": ["sunhat", "sunglasses", "sunscreen SPF 30+", "water bottle"],
        "footwear": ["comfortable sandals", "breathable sneakers"],
        "layers": ["light cardigan for AC indoors"]
    },
    "warm": {
        "essentials": ["light pants", "t-shirts", "light blouses", "shorts"],
        "accessories": ["light jacket", "sunglasses", "sunscreen"],
        "footwear": ["comfortable walking shoes", "casual sneakers"],
        "layers": ["light sweater for evenings"]
    },
    "mild": {
        "essentials": ["jeans", "long pants", "light sweaters", "long sleeves"],
        "accessories": ["light jacket", "scarf", "umbrella"],
        "footwear": ["comfortable walking shoes", "boots"],
        "layers": ["cardigan", "light jacket"]
    },
    "cool": {
        "essentials": ["warm pants", "sweaters", "long sleeves", "warm layers"],
        "accessories": ["warm jacket", "scarf", "gloves", "umbrella"],
        "footwear": ["warm boots", "waterproof shoes"],
        "layers": ["fleece", "warm jacket", "thermal underwear"]
    },
    "cold": {
        "essentials": ["thermal layers", "warm pants", "heavy sweaters", "wool clothing"],
        "accessories": ["winter coat", "warm hat", "gloves", "scarf", "warm socks"],
        "footwear": ["insulated boots", "waterproof winter shoes"],
        "layers": ["multiple layers", "thermal base layers", "insulated jacket"]
    }
}


# ---------------------------------------------------------------------------
# Main itinerary generator class
# ---------------------------------------------------------------------------

class WeatherAppropriateItineraryGenerator:
    """Generates detailed, weather-specific travel itineraries using AI.

    This generator creates 4-day itineraries that adapt to:
    - Current weather forecasts and conditions
    - Historical climate patterns and deviations
    - Time-of-day weather optimization
    - Local attractions and cultural context
    - Practical travel considerations (clothing, gear, etc.)
    """

    def __init__(self, openai_client: OpenAI):
        self._client = openai_client

    async def generate_weather_itinerary(
            self,
            city: str,
            country: str,
            weather_data: Dict[str, Any],
            climate_baseline: Optional[ClimateBaseline] = None,
            climate_deviation: Optional[WeatherDeviation] = None,
            days: int = 4
    ) -> str:
        """Generate a comprehensive weather-appropriate itinerary.

        Args:
            city: Destination city name
            country: Country name
            weather_data: Current weather forecast data
            climate_baseline: Historical climate baseline (optional)
            climate_deviation: Weather deviation analysis (optional)
            days: Number of days for itinerary (default 4)

        Returns:
            Formatted itinerary string with weather-specific recommendations
        """
        try:
            # Extract weather information
            temp_avg = weather_data.get('temp_avg', 20)
            temp_max = weather_data.get('temp_max', temp_avg + 3)
            temp_min = weather_data.get('temp_min', temp_avg - 3)
            humidity = weather_data.get('humidity', 60)
            weather_desc = weather_data.get('weather_desc', 'partly cloudy')

            # Determine weather category and clothing needs
            weather_category = self._categorize_weather(temp_avg, weather_desc)
            temp_category = self._categorize_temperature(temp_avg)
            clothing_rec = CLOTHING_RECOMMENDATIONS.get(temp_category, CLOTHING_RECOMMENDATIONS['mild'])

            # Build climate context
            climate_context = self._build_climate_context(
                climate_baseline, climate_deviation, temp_avg
            )

            # Generate AI-powered itinerary
            itinerary = await self._generate_ai_itinerary(
                city, country, weather_data, weather_category,
                temp_category, clothing_rec, climate_context, days
            )

            return itinerary

        except Exception as e:
            # Fallback to basic template
            return self._generate_fallback_itinerary(
                city, country, temp_avg, weather_desc, days
            )

    def _categorize_weather(self, temp: float, description: str) -> str:
        """Categorize weather for activity planning."""
        desc_lower = description.lower()

        if any(term in desc_lower for term in ['rain', 'shower', 'drizzle']):
            return 'rainy'
        elif any(term in desc_lower for term in ['clear', 'sunny']):
            return 'sunny' if temp > 15 else 'cloudy'
        elif any(term in desc_lower for term in ['partly cloudy', 'partly']):
            return 'partly_cloudy'
        elif temp > 28:
            return 'hot'
        elif temp < 8:
            return 'cold'
        else:
            return 'partly_cloudy'  # default

    def _categorize_temperature(self, temp: float) -> str:
        """Categorize temperature for clothing recommendations."""
        for category, condition in TEMPERATURE_CATEGORIES.items():
            if condition(temp):
                return category
        return 'mild'  # fallback

    def _build_climate_context(
            self,
            baseline: Optional[ClimateBaseline],
            deviation: Optional[WeatherDeviation],
            current_temp: float
    ) -> str:
        """Build climate context string for AI prompt."""
        if not baseline or not deviation:
            return f"Current temperature: {current_temp:.1f}°C"

        context_parts = [
            f"Current: {current_temp:.1f}°C",
            f"Historical avg: {baseline.avg_temp_mean:.1f}°C",
            f"Climate: {baseline.climate_zone}"
        ]

        if deviation.is_climate_alert:
            severity = deviation.deviation_severity
            temp_dev = deviation.temperature_deviation_c
            trend = "warmer" if temp_dev > 0 else "cooler"
            context_parts.append(f"ALERT: {severity} - {abs(temp_dev):.1f}°C {trend} than normal")
            context_parts.append(f"Impact: {deviation.traveler_impact}")
        else:
            context_parts.append("Normal seasonal weather expected")

        return " | ".join(context_parts)

    async def _generate_ai_itinerary(
            self,
            city: str,
            country: str,
            weather_data: Dict[str, Any],
            weather_category: str,
            temp_category: str,
            clothing_rec: Dict[str, List[str]],
            climate_context: str,
            days: int
    ) -> str:
        """Generate AI-powered detailed itinerary."""

        temp_avg = weather_data.get('temp_avg', 20)
        humidity = weather_data.get('humidity', 60)
        weather_desc = weather_data.get('weather_desc', 'variable')

        system_prompt = f"""You are an expert local travel guide for {city}, {country} specializing in weather-appropriate itinerary planning.

Create a detailed {days}-day itinerary that adapts to the specific weather conditions and local context.

CRITICAL REQUIREMENTS:
1. **Weather Adaptation**: Activities must match the weather conditions
2. **Time Optimization**: Schedule outdoor activities for best weather windows  
3. **Backup Plans**: Include indoor alternatives for each day
4. **Local Expertise**: Use specific local attractions, not generic suggestions
5. **Practical Advice**: Include travel tips, clothing, and gear recommendations

Format EXACTLY like this:
**DAY 1: [Theme]**
- Morning (9-12): [Specific Activity] - [Weather consideration/tip]
- Afternoon (12-17): [Specific Activity] - [Weather consideration/tip]
- Evening (17-21): [Specific Activity] - [Weather consideration/tip]  
- Weather Gear: [Specific recommendations for this day]
- Backup Plan: [Indoor alternative if weather worsens]

**DAY 2: [Theme]**
[Continue same format...]

Focus on creating a realistic, enjoyable experience for these specific weather conditions."""

        weather_activities = WEATHER_ACTIVITY_MAPPING.get(weather_category, WEATHER_ACTIVITY_MAPPING['partly_cloudy'])

        user_prompt = f"""Create a {days}-day weather-appropriate itinerary for {city}, {country}.

**WEATHER CONDITIONS:**
{climate_context}
Temperature: {temp_avg:.1f}°C (feels like {temp_category})
Conditions: {weather_desc}
Humidity: {humidity}%

**WEATHER CATEGORY:** {weather_category}
**RECOMMENDED ACTIVITIES:**
- Outdoor: {', '.join(weather_activities['outdoor'][:4])}
- Indoor: {', '.join(weather_activities['indoor'][:4])}
- Best times: {', '.join(weather_activities['best_times'])}

**CLOTHING ESSENTIALS:**
- Pack: {', '.join(clothing_rec['essentials'][:3])}
- Accessories: {', '.join(clothing_rec['accessories'][:3])}
- Footwear: {', '.join(clothing_rec['footwear'][:2])}

**SPECIAL CONSIDERATIONS:**
- Plan outdoor activities during optimal weather windows
- Include specific local attractions and neighborhoods
- Consider the unique character and culture of {city}
- Adapt to the {weather_category} conditions and {temp_category} temperatures
- Provide practical travel tips for these weather conditions

Create an authentic, local experience optimized for these specific weather conditions."""

        try:
            response = self._client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for better local knowledge
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1200,  # Generous token limit for detailed itineraries
                temperature=0.7  # Balanced creativity and consistency
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"AI itinerary generation failed: {e}")
            return self._generate_fallback_itinerary(city, country, temp_avg, weather_desc, days)

    def _generate_fallback_itinerary(
            self,
            city: str,
            country: str,
            temp_avg: float,
            weather_desc: str,
            days: int
    ) -> str:
        """Generate a basic fallback itinerary when AI fails."""
        temp_category = self._categorize_temperature(temp_avg)
        clothing = CLOTHING_RECOMMENDATIONS.get(temp_category, CLOTHING_RECOMMENDATIONS['mild'])

        fallback = f"""**WEATHER-APPROPRIATE ITINERARY FOR {city.upper()}, {country.upper()}**

**Weather Summary:** {temp_avg:.1f}°C, {weather_desc}

**DAY 1: City Center Exploration**
- Morning (9-12): Walking tour of historic center - Dress in layers for {temp_category} weather
- Afternoon (12-17): Local museums and cultural sites - Perfect for {weather_desc} conditions  
- Evening (17-21): Traditional restaurant dinner - Comfortable indoor dining
- Weather Gear: {', '.join(clothing['essentials'][:2])}, {', '.join(clothing['accessories'][:2])}
- Backup Plan: Shopping centers and covered markets

**DAY 2: Local Neighborhoods**
- Morning (9-12): Neighborhood walking tour - Best time for outdoor activities
- Afternoon (12-17): Local markets and cafes - Mix of indoor/outdoor
- Evening (17-21): Local entertainment district - Adapt to evening temperatures  
- Weather Gear: {', '.join(clothing['footwear'])}, weather protection
- Backup Plan: Indoor cultural venues and galleries

**DAY 3: Attractions & Activities**
- Morning (9-12): Major attractions - Weather-appropriate timing
- Afternoon (12-17): Parks or indoor attractions - Depends on conditions
- Evening (17-21): Scenic viewpoints or cozy venues - Temperature-dependent choice
- Weather Gear: Comfortable walking shoes, appropriate for {temp_category} temps
- Backup Plan: Museums and indoor entertainment

**DAY 4: Flexible Exploration**
- Morning (9-12): Personal interests - Choose based on weather
- Afternoon (12-17): Last-minute discoveries - Indoor/outdoor as appropriate
- Evening (17-21): Farewell dinner - Celebrate the trip indoors
- Weather Gear: Pack for departure, consider travel day comfort
- Backup Plan: Airport/departure preparation activities

**GENERAL WEATHER TIPS:**
- Temperature range: Pack for {temp_category} conditions
- Essential items: {', '.join(clothing['essentials'][:3])}
- Weather protection: {', '.join(clothing['accessories'][:2])}
- Check local weather daily and adjust plans accordingly"""

        return fallback

    # -----------------------------------------------------------------------
    # Utility methods for external integration
    # -----------------------------------------------------------------------

    def get_weather_activity_suggestions(self, temp: float, weather_desc: str) -> Dict[str, List[str]]:
        """Get weather-appropriate activity suggestions for external use."""
        weather_category = self._categorize_weather(temp, weather_desc)
        return WEATHER_ACTIVITY_MAPPING.get(weather_category, WEATHER_ACTIVITY_MAPPING['partly_cloudy'])

    def get_clothing_recommendations(self, temp: float) -> Dict[str, List[str]]:
        """Get clothing recommendations for a given temperature."""
        temp_category = self._categorize_temperature(temp)
        return CLOTHING_RECOMMENDATIONS.get(temp_category, CLOTHING_RECOMMENDATIONS['mild'])

    def should_prioritize_indoor_activities(self, temp: float, weather_desc: str) -> bool:
        """Determine if indoor activities should be prioritized."""
        weather_category = self._categorize_weather(temp, weather_desc)
        return weather_category in ['rainy', 'hot', 'cold']

    def get_optimal_outdoor_times(self, temp: float, weather_desc: str) -> List[str]:
        """Get optimal times for outdoor activities."""
        weather_category = self._categorize_weather(temp, weather_desc)
        activities = WEATHER_ACTIVITY_MAPPING.get(weather_category, WEATHER_ACTIVITY_MAPPING['partly_cloudy'])
        return activities['best_times']


# ---------------------------------------------------------------------------
# Convenience factory function
# ---------------------------------------------------------------------------

def create_itinerary_generator(openai_client: OpenAI) -> WeatherAppropriateItineraryGenerator:
    """Factory function to create an itinerary generator instance."""
    return WeatherAppropriateItineraryGenerator(openai_client)


# ---------------------------------------------------------------------------
# CLI testing interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import asyncio
    from config.config import OPENAI_KEY

    if not OPENAI_KEY:
        print("❌ OPENAI_KEY required for testing")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_KEY)
    generator = WeatherAppropriateItineraryGenerator(client)

    # Test data
    test_weather = {
        'temp_avg': 22.0,
        'temp_max': 25.0,
        'temp_min': 18.0,
        'humidity': 65,
        'weather_desc': 'partly cloudy'
    }


    async def test_generator():
        print("🗓️ Testing Weather-Appropriate Itinerary Generator\n")

        itinerary = await generator.generate_weather_itinerary(
            "Barcelona", "Spain", test_weather, days=4
        )

        print("Generated 4-day weather-appropriate itinerary:")
        print("=" * 60)
        print(itinerary)
        print("=" * 60)

        # Test utility functions
        activities = generator.get_weather_activity_suggestions(22.0, "partly cloudy")
        clothing = generator.get_clothing_recommendations(22.0)

        print(f"\nWeather activities: {activities['outdoor'][:3]}")
        print(f"Clothing recommendations: {clothing['essentials'][:3]}")


    asyncio.run(test_generator())