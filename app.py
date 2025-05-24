from shiny import App, ui, render, reactive
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import openai
from openai import OpenAI
import requests
import pandas as pd
from dotenv import load_dotenv
import os

# Install required packages:
# pip install shiny openai pandas requests python-dotenv aiohttp

# Create a .env file in your project directory with:
# OPENAI_KEY=sk-your-actual-openai-api-key-here

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from environment
OPENAI_KEY = os.getenv("OPENAI_KEY")
print(f"🔑 Environment check: OpenAI key {'✅ Found' if OPENAI_KEY else '❌ Not found'}")


# Configuration
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


class AgenticAISystem:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)  # Fixed typo: was openapi_key
        self.conversation_history: List[AgentMessage] = []
        self.available_tools = {
            "web_search": self.enhanced_web_search,
            "data_analysis": self.mock_data_analysis,
            "calculation": self.perform_calculation,
            "flight_search": self.real_flight_search,
            "weather_search": self.real_weather_search,
            "city_search": self.real_city_search
        }

    async def enhanced_web_search(self, query: str) -> TaskResult:
        """Enhanced web search - uses OpenAI to gather cities, then real weather APIs"""
        try:
            print(f"🔍 ENHANCED SEARCH for query: {query}")

            # If query is about cities and weather, use AI + real APIs
            if any(keyword in query.lower() for keyword in
                   ["destinations", "cities", "temperature", "humidity", "weather", "climate", "20c", "27c"]):

                # Step 1: Use OpenAI to gather potential cities
                print("🤖 Asking OpenAI to suggest cities...")
                ai_cities_result = await self.get_ai_suggested_cities(query)

                if not ai_cities_result.success:
                    return TaskResult(success=False,
                                      error=f"Failed to get AI city suggestions: {ai_cities_result.error}")

                suggested_cities = ai_cities_result.data
                result = f"**🤖 OPENAI CITY SUGGESTIONS:**\n{suggested_cities}\n\n"
                result += "**🌡️ CHECKING REAL WEATHER DATA:**\n\n"

                # Step 2: Parse cities from AI response and get real weather
                cities_to_check = self.parse_cities_from_ai_response(suggested_cities)

                if not cities_to_check:
                    result += "❌ Could not parse cities from AI response.\n"
                    return TaskResult(success=True, data=result)

                city_count = 0
                for city_info in cities_to_check:
                    if city_count >= 10:  # Limit to top 10
                        break

                    try:
                        # Get real weather data for each AI-suggested city
                        print(f"🌡️ Getting weather for {city_info['city']}, {city_info['region']}")
                        weather_result = await self.get_city_weather(city_info['city'], city_info['region'])
                        if weather_result.success:
                            result += f"{weather_result.data}\n"
                            city_count += 1
                    except Exception as e:
                        print(f"❌ Error getting weather for {city_info}: {e}")
                        continue

                if city_count == 0:
                    result += "❌ Unable to retrieve weather data for AI-suggested cities.\n"

                return TaskResult(success=True, data=result)

            # For flight queries
            elif any(keyword in query.lower() for keyword in ["flight", "fly", "travel", "airport", "airline"]):
                return await self.real_flight_search("toronto", "multiple destinations")

            # For other queries
            else:
                result = f"**Search Results:** This query requires specific web search capabilities for real-time information about: {query}"

            return TaskResult(success=True, data=result)

        except Exception as e:
            print(f"❌ Enhanced web search failed: {e}")
            return TaskResult(success=False, error=f"Search failed: {str(e)}")

    async def get_ai_suggested_cities(self, query: str) -> TaskResult:
        """Use OpenAI to suggest cities based on the query"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a travel expert. Suggest 10-15 cities in North America (USA, Canada, Mexico) that are likely to have temperatures between 20-27°C with mild humidity. 

Format your response as a simple list like this:
• San Diego, California
• Vancouver, British Columbia  
• Seattle, Washington
• Miami, Florida

Focus on cities that are:
1. Large enough to have airports
2. Tourist destinations or major cities
3. Known for good weather
4. Geographically diverse across North America"""},
                    {"role": "user",
                     "content": f"Based on this query: '{query}' - suggest cities in North America that might meet the temperature and humidity criteria."}
                ],
                max_tokens=400,
                temperature=0.7
            )

            ai_suggestions = response.choices[0].message.content
            return TaskResult(success=True, data=ai_suggestions)

        except Exception as e:
            return TaskResult(success=False, error=f"OpenAI city suggestion failed: {str(e)}")

    def parse_cities_from_ai_response(self, ai_response: str) -> list:
        """Parse city names from AI response"""
        cities = []
        lines = ai_response.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                # Remove bullet point and clean up
                city_line = line[1:].strip()
                if ',' in city_line:
                    parts = city_line.split(',')
                    if len(parts) >= 2:
                        city = parts[0].strip()
                        region = parts[1].strip()
                        cities.append({
                            "city": city.lower(),
                            "region": region.lower()
                        })

        # If no bullet points found, try to extract cities differently
        if not cities:
            for line in lines:
                if ',' in line and any(state in line.lower() for state in
                                       ['california', 'florida', 'texas', 'washington', 'oregon', 'british columbia',
                                        'ontario']):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        city = parts[0].strip()
                        region = parts[1].strip()
                        cities.append({
                            "city": city.lower(),
                            "region": region.lower()
                        })

        return cities

    async def get_city_weather(self, city: str, region: str) -> TaskResult:
        """Get real weather data for a specific city"""
        try:
            import aiohttp

            # Get coordinates for the city
            geocode_url = f"https://nominatim.openstreetmap.org/search?q={city}+{region}&format=json&limit=1"

            async with aiohttp.ClientSession() as session:
                async with session.get(geocode_url, headers={'User-Agent': 'AgenticAI-Demo/1.0'}) as response:
                    if response.status == 200:
                        cities = await response.json()
                        if cities:
                            city_data = cities[0]
                            lat = float(city_data.get('lat', 0))
                            lon = float(city_data.get('lon', 0))

                            # Get real weather for this city
                            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=temperature_2m,relativehumidity_2m&timezone=auto"

                            async with session.get(weather_url) as weather_response:
                                if weather_response.status == 200:
                                    weather_data = await weather_response.json()
                                    current = weather_data['current_weather']
                                    hourly = weather_data['hourly']

                                    # Calculate average humidity from next 24 hours
                                    avg_humidity = sum(hourly['relativehumidity_2m'][:24]) / 24
                                    temp_c = current['temperature']

                                    # Check if it meets criteria (20-27°C)
                                    meets_criteria = "✅" if 20 <= temp_c <= 27 else "❌"
                                    humidity_status = "Mild" if 40 <= avg_humidity <= 70 else "High" if avg_humidity > 70 else "Low"

                                    result = f"📍 **{city.title()}, {region}** {meets_criteria}\n"
                                    result += f"   🌡️ Temperature: {temp_c}°C\n"
                                    result += f"   💧 Humidity: {avg_humidity:.0f}% ({humidity_status})\n"
                                    result += f"   🌬️ Wind: {current['windspeed']} km/h\n"

                                    return TaskResult(success=True, data=result)

            return TaskResult(success=False, error=f"Could not get weather data for {city}")

        except Exception as e:
            return TaskResult(success=False, error=f"Weather lookup failed for {city}: {str(e)}")

    async def real_weather_search(self, lat: float, lon: float, city_name: str) -> TaskResult:
        """Real weather API using Open-Meteo (no key required)"""
        try:
            import aiohttp
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=temperature_2m,relativehumidity_2m,apparent_temperature&timezone=auto"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        current = data['current_weather']
                        hourly = data['hourly']

                        # Calculate average humidity from next 24 hours
                        avg_humidity = sum(hourly['relativehumidity_2m'][:24]) / 24

                        result = f"""
**🌡️ Current Weather for {city_name}:**
• Temperature: {current['temperature']}°C
• Feels like: {hourly['apparent_temperature'][0]}°C  
• Humidity: {avg_humidity:.0f}%
• Wind Speed: {current['windspeed']} km/h
• Conditions: {"☀️ Clear" if current['weathercode'] == 0 else "🌤️ Partly Cloudy" if current['weathercode'] < 3 else "☁️ Cloudy"}
                        """
                        return TaskResult(success=True, data=result)
                    else:
                        return TaskResult(success=False, error=f"Weather API error: {response.status}")
        except Exception as e:
            return TaskResult(success=False, error=f"Weather search failed: {str(e)}")

    async def real_city_search(self, query: str) -> TaskResult:
        """Real city search with weather data using OpenStreetMap + Open-Meteo APIs"""
        try:
            import aiohttp

            # If it's a simple city name, search for it
            if any(city in query.lower() for city in
                   ["san diego", "santa barbara", "los angeles", "vancouver", "seattle", "portland", "miami",
                    "phoenix"]):
                # Extract city name
                city_name = query.lower().strip()
                for known_city in ["san diego", "santa barbara", "los angeles", "vancouver", "seattle", "portland",
                                   "miami", "phoenix"]:
                    if known_city in city_name:
                        city_name = known_city
                        break

                # Get coordinates for the city
                geocode_url = f"https://nominatim.openstreetmap.org/search?q={city_name}&format=json&limit=1"

                async with aiohttp.ClientSession() as session:
                    async with session.get(geocode_url, headers={'User-Agent': 'AgenticAI-Demo/1.0'}) as response:
                        if response.status == 200:
                            cities = await response.json()
                            if cities:
                                city_data = cities[0]
                                lat = float(city_data.get('lat', 0))
                                lon = float(city_data.get('lon', 0))
                                display_name = city_data.get('display_name', city_name)

                                # Get real weather for this city
                                weather = await self.real_weather_search(lat, lon, city_name.title())

                                if weather.success:
                                    return TaskResult(success=True,
                                                      data=f"📍 **{city_name.title()}** ({display_name.split(',')[1] if ',' in display_name else 'North America'})\n{weather.data}")
                                else:
                                    return TaskResult(success=True,
                                                      data=f"📍 **{city_name.title()}**: Location found but weather data unavailable")

            # Fallback for general searches
            url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=3&countrycodes=us,ca,mx"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={'User-Agent': 'AgenticAI-Demo/1.0'}) as response:
                    if response.status == 200:
                        cities = await response.json()
                        if cities:
                            result = "**🏙️ Cities Found:**\n"
                            for city in cities[:3]:  # Limit to top 3 results
                                name = city.get('display_name', '').split(',')[0]
                                result += f"• {name}\n"
                            return TaskResult(success=True, data=result)
                        else:
                            return TaskResult(success=False, error="No cities found")
                    else:
                        return TaskResult(success=False, error=f"City search API error: {response.status}")
        except Exception as e:
            return TaskResult(success=False, error=f"City search failed: {str(e)}")

    async def real_flight_search(self, origin_city: str, destination_city: str) -> TaskResult:
        """Flight search with specific city information"""
        try:
            if destination_city == "multiple destinations":
                result = f"""
**✈️ Flight Search Guide from {origin_city.title()}:**

**💡 General Flight Estimates:**
• Domestic flights (same country): $150-400 CAD
• Cross-border (US-Canada): $250-500 CAD  
• Coast-to-coast: $300-600 CAD

**🔍 Best Booking Sites for Real Prices:**
• Google Flights: flights.google.com
• Kayak: kayak.com
• Skyscanner: skyscanner.com
• Momondo: momondo.com

**💰 Money-Saving Tips:**
• Book 3-8 weeks in advance
• Fly Tuesday/Wednesday for cheaper fares
• Check nearby airports (often 20-40% savings)
• Use incognito mode when searching
• Set price alerts for your route
                """
            else:
                # Specific city flight information
                flight_data = {
                    "san diego": {
                        "price": "$320-450 CAD",
                        "duration": "5h 30m",
                        "stops": "1 stop (usually Phoenix or Denver)",
                        "airlines": "Air Canada, American, United",
                        "airport": "San Diego International (SAN)",
                        "tips": "Tuesday/Wednesday departures save $50+"
                    },
                    "vancouver": {
                        "price": "$280-380 CAD",
                        "duration": "4h 30m",
                        "stops": "Direct flights available",
                        "airlines": "Air Canada, WestJet",
                        "airport": "Vancouver International (YVR)",
                        "tips": "Off-peak hours save money"
                    },
                    "seattle": {
                        "price": "$290-420 CAD",
                        "duration": "5h 45m",
                        "stops": "1 stop (usually Vancouver)",
                        "airlines": "Air Canada, United, Delta",
                        "airport": "Seattle-Tacoma International (SEA)",
                        "tips": "Early morning flights often cheaper"
                    },
                    "miami": {
                        "price": "$380-550 CAD",
                        "duration": "6h 15m",
                        "stops": "1-2 stops (Charlotte, Atlanta)",
                        "airlines": "Air Canada, American, Delta",
                        "airport": "Miami International (MIA)",
                        "tips": "Avoid hurricane season for better prices"
                    },
                    "los angeles": {
                        "price": "$350-500 CAD",
                        "duration": "5h 45m",
                        "stops": "Direct or 1 stop",
                        "airlines": "Air Canada, American, United",
                        "airport": "Los Angeles International (LAX)",
                        "tips": "Mid-week flights significantly cheaper"
                    },
                    "phoenix": {
                        "price": "$330-480 CAD",
                        "duration": "6h 30m",
                        "stops": "1-2 stops (Denver, Chicago)",
                        "airlines": "Air Canada, American, United",
                        "airport": "Phoenix Sky Harbor (PHX)",
                        "tips": "Summer months have lower prices"
                    }
                }

                city_key = destination_city.lower().strip()
                if city_key in flight_data:
                    flight_info = flight_data[city_key]
                    result = f"""
**✈️ Flights: Toronto (YYZ) → {destination_city.title()}**

**💰 Price Range:** {flight_info['price']}
**⏱️ Flight Time:** {flight_info['duration']}
**✈️ Routing:** {flight_info['stops']}
**🏢 Airlines:** {flight_info['airlines']}
**🛬 Airport:** {flight_info['airport']}
**💡 Best Deal Tip:** {flight_info['tips']}

**📅 Current Booking Recommendations:**
• Use Google Flights for real-time prices
• Book 2-8 weeks in advance for best deals
• Consider nearby airports for savings
• Check airline websites directly for exclusive deals

**🎯 Quick Book Links:**
• Google Flights: flights.google.com
• Air Canada: aircanada.com
• Kayak: kayak.com
                    """
                else:
                    result = f"""
**✈️ Flights: Toronto → {destination_city.title()}**

**General Estimates:**
• Flight time: 4-8 hours (depending on stops)
• Price range: $300-600 CAD
• Best booking: Google Flights, Kayak, Skyscanner

**💡 Tips:**
• Book 3-8 weeks in advance
• Check multiple airports nearby
• Compare airlines directly
                    """

            return TaskResult(success=True, data=result)

        except Exception as e:
            return TaskResult(success=False, error=f"Flight search failed: {str(e)}")
        """Simulated web search tool with more specific responses"""
        await asyncio.sleep(1)  # Simulate API delay

        # More specific mock results based on query content
        if "temperature" in query.lower() and "humidity" in query.lower():
            result = """
**Cities with 20-26°C and Mild Humidity:**
• San Diego, CA: 22°C avg, 55% humidity, Pop: 1.4M
• Santa Barbara, CA: 21°C avg, 52% humidity, Pop: 110K
• Vancouver, BC: 23°C avg, 48% humidity, Pop: 675K
• Victoria, BC: 24°C avg, 51% humidity, Pop: 370K
• Portland, OR: 21°C avg, 57% humidity, Pop: 650K
• Seattle, WA: 20°C avg, 62% humidity, Pop: 750K
• Monterey, CA: 22°C avg, 54% humidity, Pop: 30K
            """
        elif "flight" in query.lower() and "toronto" in query.lower():
            result = """
**Flights from Toronto (YYZ):**
• To San Diego: $320-450 CAD, 5h 30m (1 stop)
• To Vancouver: $280-380 CAD, 4h 30m (direct)
• To Portland: $350-480 CAD, 6h 15m (1 stop)
• To Seattle: $290-420 CAD, 5h 45m (1 stop)
**Best deals:** Air Canada, WestJet, Delta
            """
        elif "artificial intelligence" in query.lower():
            result = "AI is rapidly advancing with LLMs and autonomous agents..."
        elif "climate change" in query.lower():
            result = "Global temperatures rising, renewable energy adoption increasing..."
        elif "space exploration" in query.lower():
            result = "Recent Mars missions successful, commercial spaceflight expanding..."
        else:
            result = f"Search results for '{query}': Multiple relevant articles found discussing various aspects..."

        return TaskResult(success=True, data=result)

    async def perform_calculation(self, expression: str) -> TaskResult:
        """Simple calculation tool"""
        try:
            # Simple and safe calculation
            if any(char in expression for char in ['import', 'exec', 'eval', '__']):
                return TaskResult(success=False, error="Invalid expression")
            result = eval(expression.replace('^', '**'))  # Basic math only
            return TaskResult(success=True, data=f"{expression} = {result}")
        except Exception as e:
            return TaskResult(success=False, error=str(e))

    async def mock_data_analysis(self, data_description: str) -> TaskResult:
        """Simulated data analysis tool"""
        await asyncio.sleep(1.5)
        analysis = f"Analysis of {data_description}: Trends show 15% increase over last quarter with seasonal variations..."
        return TaskResult(success=True, data=analysis)

    async def perform_calculation(self, expression: str) -> TaskResult:
        """Simple calculation tool"""
        try:
            # Simple and safe calculation
            if any(char in expression for char in ['import', 'exec', 'eval', '__']):
                return TaskResult(success=False, error="Invalid expression")
            result = eval(expression.replace('^', '**'))  # Basic math only
            return TaskResult(success=True, data=f"{expression} = {result}")
        except Exception as e:
            return TaskResult(success=False, error=str(e))

    async def mock_flight_search(self, origin: str, destination: str) -> TaskResult:
        """Simulated flight search tool"""
        await asyncio.sleep(2)  # Simulate API delay

        # Mock flight data based on common routes from Toronto
        flight_data = {
            ("toronto", "san diego"): {
                "price": "$320-450 CAD",
                "duration": "5h 30m",
                "stops": "1 stop (usually Phoenix or Denver)",
                "airlines": "Air Canada, American, United",
                "best_time": "Tuesday/Wednesday departures"
            },
            ("toronto", "vancouver"): {
                "price": "$280-380 CAD",
                "duration": "4h 30m",
                "stops": "Direct available",
                "airlines": "Air Canada, WestJet",
                "best_time": "Off-peak hours save $50+"
            },
            ("toronto", "portland"): {
                "price": "$350-480 CAD",
                "duration": "6h 15m",
                "stops": "1 stop (Seattle or Vancouver)",
                "airlines": "Air Canada, Delta, Alaska",
                "best_time": "Mid-week departures"
            },
            ("toronto", "seattle"): {
                "price": "$290-420 CAD",
                "duration": "5h 45m",
                "stops": "1 stop (Vancouver or Chicago)",
                "airlines": "Air Canada, United, Delta",
                "best_time": "Early morning flights"
            }
        }

        key = (origin.lower(), destination.lower())
        if key in flight_data:
            flight_info = flight_data[key]
            result = f"""
**Flight Options: {origin.title()} → {destination.title()}**
💰 Price Range: {flight_info['price']}
⏱️ Flight Time: {flight_info['duration']}
✈️ Routing: {flight_info['stops']}
🏢 Airlines: {flight_info['airlines']}
💡 Tip: {flight_info['best_time']}

**Booking Recommendations:**
• Use Google Flights or Kayak for real-time prices
• Book 2-8 weeks in advance for best deals
• Consider nearby airports for savings
            """
        else:
            result = f"Flight search for {origin} to {destination}: Multiple options available. Use flight comparison sites for current pricing."

        return TaskResult(success=True, data=result)
        """Simple calculation tool"""
        try:
            # Simple and safe calculation
            if any(char in expression for char in ['import', 'exec', 'eval', '__']):
                return TaskResult(success=False, error="Invalid expression")
            result = eval(expression.replace('^', '**'))  # Basic math only
            return TaskResult(success=True, data=f"{expression} = {result}")
        except Exception as e:
            return TaskResult(success=False, error=str(e))

    async def call_agent(self, role: AgentRole, prompt: str, context: str = "") -> AgentMessage:
        """Call a specific agent with a prompt"""

        # Define agent personalities and capabilities
        agent_prompts = {
            AgentRole.PLANNER: f"""You are a strategic planning agent. Break down complex tasks into steps.
            Context: {context}
            Task: {prompt}

            Create a clear action plan with numbered steps. Identify which tools might be needed.""",

            AgentRole.RESEARCHER: f"""You are a research agent with access to web search and data gathering tools.
            Context: {context}
            Task: {prompt}

            Conduct thorough research and gather relevant information. Use tools when needed.""",

            AgentRole.ANALYZER: f"""You are an analytical agent specialized in data processing and pattern recognition.
            Context: {context}
            Task: {prompt}

            Analyze the provided information and extract insights. Use analytical tools when helpful.""",

            AgentRole.SYNTHESIZER: f"""You are a synthesis agent that combines information into coherent outputs.
            Context: {context}
            Task: {prompt}

            Create comprehensive summaries and actionable recommendations."""
        }

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": agent_prompts[role]},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            content = response.choices[0].message.content

            # Simulate tool usage based on agent type
            tools_used = []
            if role == AgentRole.RESEARCHER and "search" in prompt.lower():
                tools_used.append("web_search")
            elif role == AgentRole.ANALYZER and ("data" in prompt.lower() or "analyze" in prompt.lower()):
                tools_used.append("data_analysis")

            message = AgentMessage(
                agent_role=role,
                content=content,
                timestamp=datetime.now(),
                tools_used=tools_used,
                metadata={"model": "gpt-3.5-turbo", "tokens": len(content.split())}
            )

            self.conversation_history.append(message)
            return message

        except Exception as e:
            error_message = AgentMessage(
                agent_role=role,
                content=f"Error: {str(e)}",
                timestamp=datetime.now(),
                tools_used=[],
                metadata={"error": True}
            )
            self.conversation_history.append(error_message)
            return error_message

    async def run_agentic_workflow(self, user_query: str) -> List[AgentMessage]:
        """Run a complete agentic workflow"""
        workflow_messages = []

        # Step 1: Planning
        plan_msg = await self.call_agent(
            AgentRole.PLANNER,
            f"Create a plan to address this query: {user_query}"
        )
        workflow_messages.append(plan_msg)

        # Step 2: Research
        research_msg = await self.call_agent(
            AgentRole.RESEARCHER,
            f"Research information related to: {user_query}",
            context=plan_msg.content
        )
        workflow_messages.append(research_msg)

        # Step 3: Analysis
        analysis_msg = await self.call_agent(
            AgentRole.ANALYZER,
            f"Analyze the research findings for: {user_query}",
            context=f"Plan: {plan_msg.content}\nResearch: {research_msg.content}"
        )
        workflow_messages.append(analysis_msg)

        # Step 4: Synthesis
        synthesis_msg = await self.call_agent(
            AgentRole.SYNTHESIZER,
            f"Synthesize findings and provide recommendations for: {user_query}",
            context=f"Research: {research_msg.content}\nAnalysis: {analysis_msg.content}"
        )
        workflow_messages.append(synthesis_msg)

        return workflow_messages


# Initialize the agentic system with environment variable if available
ai_system = AgenticAISystem(OPENAI_KEY) if OPENAI_KEY else None
if ai_system:
    print("🚀 Auto-connected to OpenAI using environment variable")
else:
    print("⚠️ No environment key found - manual connection required")

# Shiny UI
app_ui = ui.page_fillable(
    ui.h1("🤖 Agentic AI System Demo", class_="text-center mb-4"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Configuration"),
            ui.input_password("api_key", "OpenAI API Key:", placeholder="sk-..."),
            ui.input_action_button("connect", "Connect", class_="btn-primary mb-3"),
            ui.hr(),

            ui.h4("Quick Examples"),
            ui.input_action_button("example1", "🤖 AI Research", class_="btn-outline-secondary btn-sm mb-2"),
            ui.input_action_button("example2", "📊 Market Analysis", class_="btn-outline-secondary btn-sm mb-2"),
            ui.input_action_button("example3", "🏥 Technical Planning", class_="btn-outline-secondary btn-sm mb-2"),
            ui.input_action_button("example4", "🌴 Travel Planning", class_="btn-outline-info btn-sm mb-2"),

            width=300
        ),

        ui.div(
            ui.div(
                ui.input_text_area(
                    "user_query",
                    "Enter your query for the agentic AI system:",
                    placeholder="e.g., 'Analyze the current state of artificial intelligence and its implications for businesses'",
                    rows=3,
                    width="100%"
                ),
                ui.input_action_button("run_workflow", "Run Agentic Workflow", class_="btn-success mb-4"),
                class_="mb-4"
            ),

            ui.div(
                ui.output_ui("status_display"),
                class_="mb-3"
            ),

            ui.div(
                ui.h3("Agent Conversation Flow"),
                ui.output_ui("conversation_display"),
                class_="conversation-container"
            )
        )
    ),

    # Custom CSS
    ui.tags.style("""
        .conversation-container {
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: #f8f9fa;
        }
        .city-selector-container {
            background-color: #e8f4fd;
            border: 1px solid #bee5eb;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .agent-message {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid;
        }
        .agent-planner { border-left-color: #007bff; background-color: #e3f2fd; }
        .agent-researcher { border-left-color: #28a745; background-color: #e8f5e9; }
        .agent-analyzer { border-left-color: #ffc107; background-color: #fff8e1; }
        .agent-synthesizer { border-left-color: #dc3545; background-color: #ffebee; }
        .agent-header {
            font-weight: bold;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .tools-used {
            font-size: 0.8em;
            color: #666;
            font-style: italic;
        }
        .timestamp {
            font-size: 0.8em;
            color: #888;
        }
    """)
)


def server(input, output, session):
    # Reactive values
    workflow_status = reactive.Value(
        "✅ Auto-connected using environment variable" if ai_system
        else "⚠️ No environment key found - please connect manually"
    )
    conversation_messages = reactive.Value([])
    found_cities = reactive.Value([])  # Store cities found from search
    selected_cities = reactive.Value([])  # Store user-selected cities

    @reactive.Effect
    @reactive.event(input.connect)
    def connect_ai_system():
        global ai_system
        # Use manual input first, fallback to environment variable
        api_key = input.api_key() or OPENAI_KEY

        if api_key:
            try:
                ai_system = AgenticAISystem(api_key)
                source = "manual input" if input.api_key() else "environment variable"
                workflow_status.set(f"✅ Connected to OpenAI API (using {source})")
            except Exception as e:
                workflow_status.set(f"❌ Connection failed: {str(e)}")
        else:
            workflow_status.set("⚠️ Please enter your OpenAI API key or set OPENAI_KEY in your .env file")

    @reactive.Effect
    @reactive.event(input.example1)
    def load_example1():
        ui.update_text_area("user_query",
                            value="Analyze the current state of artificial intelligence research and its potential impact on various industries")

    @reactive.Effect
    @reactive.event(input.example2)
    def load_example2():
        ui.update_text_area("user_query",
                            value="Conduct a market analysis for renewable energy technologies and identify investment opportunities")

    @reactive.Effect
    @reactive.event(input.example3)
    def load_example3():
        ui.update_text_area("user_query",
                            value="Create a technical roadmap for implementing machine learning in a healthcare organization")

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

        workflow_status.set("🔄 Running agentic workflow...")
        conversation_messages.set([])

        try:
            # Run the workflow
            messages = await ai_system.run_agentic_workflow(input.user_query())
            conversation_messages.set(messages)
            workflow_status.set("✅ Workflow completed successfully")

        except Exception as e:
            workflow_status.set(f"❌ Workflow failed: {str(e)}")

    @output
    @render.ui
    def status_display():
        status = workflow_status.get()
        if status:
            return ui.div(
                ui.p(status, class_="mb-0"),
                class_="alert alert-info"
            )
        return ui.div()

    @output
    @render.ui
    def conversation_display():
        messages = conversation_messages.get()
        if not messages:
            return ui.div(
                ui.p("No conversation yet. Enter a query and run the workflow to see agents in action!",
                     class_="text-muted text-center"),
                style="padding: 40px;"
            )

        conversation_elements = []

        for i, msg in enumerate(messages):
            # Agent icon and color mapping
            agent_info = {
                AgentRole.PLANNER: {"icon": "📋", "name": "Planner Agent", "class": "agent-planner"},
                AgentRole.RESEARCHER: {"icon": "🔍", "name": "Research Agent", "class": "agent-researcher"},
                AgentRole.ANALYZER: {"icon": "📊", "name": "Analysis Agent", "class": "agent-analyzer"},
                AgentRole.SYNTHESIZER: {"icon": "🔧", "name": "Synthesis Agent", "class": "agent-synthesizer"}
            }

            info = agent_info[msg.agent_role]

            # Tools used display
            tools_display = ""
            if msg.tools_used:
                tools_display = f" | Tools: {', '.join(msg.tools_used)}"

            conversation_elements.append(
                ui.div(
                    ui.div(
                        ui.span(f"{info['icon']} {info['name']}", class_="agent-name"),
                        ui.span(
                            f"{msg.timestamp.strftime('%H:%M:%S')}{tools_display}",
                            class_="timestamp"
                        ),
                        class_="agent-header"
                    ),
                    ui.div(msg.content, style="white-space: pre-wrap; line-height: 1.5;"),
                    class_=f"agent-message {info['class']}"
                )
            )

        return ui.div(*conversation_elements)


# Create the app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run(debug=True)