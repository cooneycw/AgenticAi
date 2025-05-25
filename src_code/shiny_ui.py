# src_code/shiny_ui.py
"""Enhanced UI with Structured Data Display - COMPLETE REWRITE

This enhanced UI properly extracts and displays:
1. **Recommended Cities**: With weather and climate data
2. **Flight Information**: With real flight data and recommendations
3. **Weather Itineraries**: Formatted day-by-day plans
4. **Climate Analysis**: With alerts and deviations
5. **Real-Time Debug Console**: For transparency with improved readability

Key improvements:
- Clean, organized code structure
- Structured data extraction from agent messages
- Dedicated sections for cities, flights, and itineraries
- Improved content formatting and organization
- Better visual hierarchy and information display
- FIXED: Debug console now uses light background with dark text for better readability
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from shiny import App, reactive, render, ui

from config.config import (
    AgentRole,
    FLIGHTRADAR24_KEY,
    OPENAI_KEY,
    OPENWEATHER_KEY,
)
from src_code.agentic_core import AgenticAISystem, ProgressStep
from src_code.debug_logger import setup_debug_capture, debug_logger, debug_info, debug_success, debug_agent

# ---------------------------------------------------------------------------
# Enhanced system instantiation with debug capture
# ---------------------------------------------------------------------------

# Set up debug capture for real-time logging
debug_logger_instance = setup_debug_capture()

ai_system: AgenticAISystem | None = (
    AgenticAISystem(OPENAI_KEY, OPENWEATHER_KEY, FLIGHTRADAR24_KEY)
    if OPENAI_KEY
    else None
)

# Log system initialization
if ai_system:
    debug_success("Enhanced Agentic AI System initialized successfully")
else:
    debug_info("System waiting for OpenAI API key configuration")


# ---------------------------------------------------------------------------
# Enhanced session data structure for structured results
# ---------------------------------------------------------------------------

class EnhancedQuerySession:
    """Represents a query session with structured data extraction."""

    def __init__(self, query: str, timestamp: datetime):
        self.query = query
        self.timestamp = timestamp
        self.messages: List[Any] = []
        self.status = "🔄 Processing..."
        self.progress = ""
        self.expanded = True
        self.completed = False

        # Structured data extraction
        self.recommended_cities: List[Dict[str, Any]] = []
        self.climate_alerts: List[Dict[str, Any]] = []
        self.flight_data: Dict[str, Any] = {}
        self.itineraries: List[str] = []
        self.weather_analysis: Dict[str, Any] = {}

    @property
    def session_id(self) -> str:
        return f"session_{self.timestamp.strftime('%H%M%S')}"

    @property
    def display_time(self) -> str:
        return self.timestamp.strftime("%H:%M:%S")

    def extract_structured_data(self):
        """Extract structured data from agent messages."""
        for msg in self.messages:
            if msg.metadata:
                # Extract recommended cities from analyzer/synthesizer
                if msg.agent_role in [AgentRole.ANALYZER, AgentRole.SYNTHESIZER]:
                    destinations = msg.metadata.get("scored_destinations", [])
                    if destinations:
                        self.recommended_cities = destinations[:5]  # Top 5

                    final_recs = msg.metadata.get("final_recommendations", [])
                    if final_recs and not self.recommended_cities:
                        self.recommended_cities = final_recs

                # Extract climate alerts
                alerts = msg.metadata.get("climate_alerts", [])
                if alerts:
                    self.climate_alerts = [
                        {
                            "city": alert.city,
                            "level": alert.alert_level,
                            "message": alert.message,
                            "recommendations": alert.recommendations
                        } for alert in alerts
                    ]

                # Extract flight data
                if msg.metadata.get("flight_intelligence_active"):
                    self.flight_data = {
                        "fr24_available": msg.metadata.get("fr24_available", False),
                        "successful_calls": msg.metadata.get("fr24_successful_calls", 0),
                        "flight_successes": msg.metadata.get("flight_analysis_successes", 0),
                        "total_analyzed": msg.metadata.get("successful_analyses", 0)
                    }

                # Extract weather analysis summary
                if msg.metadata.get("climate_intelligence_active"):
                    self.weather_analysis = {
                        "cities_processed": msg.metadata.get("cities_processed", 0),
                        "alerts_count": len(self.climate_alerts),
                        "average_score": msg.metadata.get("average_score", 0),
                        "meeting_criteria": msg.metadata.get("meeting_criteria_count", 0)
                    }

            # Extract itineraries from content
            if "**DAY" in msg.content and "ITINERARY" in msg.content.upper():
                itinerary_sections = self._extract_itineraries(msg.content)
                if itinerary_sections:
                    self.itineraries.extend(itinerary_sections)

    def _extract_itineraries(self, content: str) -> List[str]:
        """Extract individual itineraries from agent content."""
        itineraries = []

        # Look for itinerary patterns
        itinerary_pattern = r'(\*\*.*?ITINERARY.*?\*\*.*?)(?=\*\*.*?ITINERARY|\*\*.*?:|$)'
        matches = re.findall(itinerary_pattern, content, re.DOTALL | re.IGNORECASE)

        for match in matches:
            if "**DAY" in match:
                itineraries.append(match.strip())

        # If no clear itinerary sections, look for DAY patterns
        if not itineraries:
            day_pattern = r'(\*\*DAY.*?)(?=\*\*DAY|\*\*[A-Z]|$)'
            day_matches = re.findall(day_pattern, content, re.DOTALL)

            if len(day_matches) >= 2:  # At least 2 days to consider it an itinerary
                itinerary = "\n\n".join(day_matches)
                if len(itinerary) > 100:  # Reasonable length check
                    itineraries.append(itinerary)

        return itineraries


# ---------------------------------------------------------------------------
# Enhanced UI layout with better data display
# ---------------------------------------------------------------------------

app_ui = ui.page_fillable(
    ui.h1("🌍 Enhanced Agentic Travel Planner", class_="text-center mb-4"),
    ui.p(
        "🌡️ Climate Intelligence • ✈️ Real Flight Data • 🎯 Smart Recommendations • 🗓️ Weather-Appropriate Itineraries",
        class_="text-center text-muted mb-4"
    ),

    ui.layout_sidebar(
        ui.sidebar(
            # System Status Section
            ui.h3("System Status"),
            ui.output_ui("system_status_display"),
            ui.hr(),

            # Enhanced examples section
            ui.h4("Enhanced Examples"),
            ui.input_action_button("example1", "🌍 Europe: Climate + Flight Analysis",
                                   class_="btn-outline-info btn-sm mb-1 w-100 text-start"),
            ui.input_action_button("example2", "🏝️ Asia: Real Flight Data + Itineraries",
                                   class_="btn-outline-info btn-sm mb-1 w-100 text-start"),
            ui.input_action_button("example3", "🌮 Mexico: Flight Recommendations",
                                   class_="btn-outline-info btn-sm mb-1 w-100 text-start"),
            ui.input_action_button("example4", "❄️ Dublin: Cool Weather + Flights",
                                   class_="btn-outline-info btn-sm mb-1 w-100 text-start"),
            ui.input_action_button("example5", "🌡️ Climate + FlightRadar24 Test",
                                   class_="btn-outline-info btn-sm mb-1 w-100 text-start"),
            ui.input_action_button("example6", "🗺️ Multi-City Flight Comparison",
                                   class_="btn-outline-info btn-sm mb-1 w-100 text-start"),
            ui.input_action_button("example7", "🌟 Complete Analysis Demo",
                                   class_="btn-outline-success btn-sm mb-1 w-100 text-start"),
            ui.hr(),

            # Session management
            ui.h5("Session Management"),
            ui.input_action_button("clear_all", "🗑️ Clear All Results",
                                   class_="btn-outline-danger btn-sm mb-2 w-100"),
            ui.input_action_button("expand_all", "📖 Expand All",
                                   class_="btn-outline-secondary btn-sm mb-1 w-100"),
            ui.input_action_button("collapse_all", "📕 Collapse All",
                                   class_="btn-outline-secondary btn-sm mb-1 w-100"),
            ui.input_action_button("clear_debug", "🧹 Clear Debug Log",
                                   class_="btn-outline-warning btn-sm mb-1 w-100"),
            ui.input_action_button("test_debug", "🧪 Test Debug Console",
                                   class_="btn-outline-info btn-sm mb-2 w-100"),
            ui.hr(),

            # Feature status section
            ui.h5("Enhanced Features"),
            ui.p("✅ Structured Data Display", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ City Recommendations", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Flight Analysis & Scoring", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Weather Itineraries", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Climate Intelligence", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Real-Time Debug Console", style="font-size: 0.85em; margin: 2px 0;"),

            width=340
        ),

        ui.div(
            # Query input section
            ui.div(
                ui.input_text_area(
                    "user_query",
                    "Enter your enhanced travel query:",
                    placeholder="e.g., 'Find diverse European destinations with 18-25°C in 10 days from Toronto, include detailed weather itineraries'",
                    rows=3,
                    width="100%"
                ),
                ui.input_action_button("run", "🚀 Run Enhanced AI Agents",
                                       class_="btn-success mb-4"),
                class_="mb-4"
            ),

            # Progress display
            ui.div(
                ui.output_ui("progress_display"),
                class_="mb-3"
            ),

            # Debug console section
            ui.div(
                ui.input_action_button("toggle_debug", "🔍 Hide Debug Console",
                                       class_="btn-outline-info mb-2"),
                ui.div(
                    ui.output_ui("debug_console"),
                    id="debug_section",
                    class_="debug-section"
                ),
                class_="mb-3"
            ),

            # Enhanced sessions display with structured data
            ui.div(
                ui.h3("Query Sessions"),
                ui.output_ui("sessions_display"),
                class_="sessions-container"
            )
        )
    ),

    # Enhanced CSS styling with FIXED debug console colors
    ui.tags.style("""
        .sessions-container {
            max-height: 800px;
            overflow-y: auto;
        }

        /* FIXED: Debug section now uses light background with dark text for better readability */
        .debug-section {
            max-height: 300px;
            overflow-y: auto;
            background: #f8f9fa !important;
            border: 1px solid #dee2e6 !important;
            border-radius: 8px;
            margin-bottom: 20px;
            display: block !important;
            visibility: visible !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .debug-section.collapsed {
            display: none !important;
            visibility: hidden;
        }

        /* FIXED: Debug console now uses light background with dark text */
        .debug-console {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
            color: #212529 !important;
            background: #ffffff !important;
            padding: 15px;
            white-space: pre-wrap;
            word-wrap: break-word;
            min-height: 200px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }

        .system-status {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
            margin-bottom: 15px;
            font-size: 0.9em;
        }
        .system-status.warning {
            background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 100%);
            border-left-color: #FF9800;
        }
        .system-status.error {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-left-color: #F44336;
        }

        .session-card {
            margin-bottom: 25px;
            border: 1px solid #ddd;
            border-radius: 12px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .session-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .session-header {
            padding: 15px 20px;
            border-bottom: 1px solid #ddd;
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            color: white;
            border-radius: 12px 12px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .session-content {
            padding: 20px;
        }
        .session-collapsed .session-content {
            display: none;
        }

        .data-section {
            margin-bottom: 25px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: white;
            overflow: hidden;
        }
        .data-section-header {
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            padding: 12px 16px;
            font-weight: bold;
            font-size: 1.1em;
        }
        .data-section-content {
            padding: 16px;
        }

        .city-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .city-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .city-name {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }
        .city-score {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .city-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .city-detail {
            background: rgba(255,255,255,0.7);
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.9em;
        }

        .flight-info {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border: 1px solid #2196f3;
            border-radius: 6px;
            padding: 12px;
            margin-top: 10px;
        }

        .climate-alert {
            background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 100%);
            border: 1px solid #ff9800;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 15px;
            border-left: 4px solid #ff5722;
        }
        .climate-alert.extreme {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-color: #f44336;
            border-left-color: #d32f2f;
        }

        .itinerary-container {
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            border: 1px solid #9c27b0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
        }
        .itinerary-day {
            background: rgba(255,255,255,0.8);
            margin-bottom: 12px;
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid #9c27b0;
        }
        .itinerary-day-title {
            font-weight: bold;
            color: #6a1b9a;
            margin-bottom: 8px;
        }

        .weather-summary {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border: 1px solid #4caf50;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 15px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }
        .weather-stat {
            text-align: center;
            background: rgba(255,255,255,0.7);
            padding: 8px;
            border-radius: 4px;
        }
        .weather-stat-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #2e7d32;
        }
        .weather-stat-label {
            font-size: 0.85em;
            color: #666;
        }

        .btn-toggle {
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }
        .btn-toggle:hover {
            background: rgba(255,255,255,0.3);
        }

        .progress-container {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            margin-bottom: 15px;
        }
        .progress-bar-container {
            background: #e9ecef;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #007bff 0%, #28a745 100%);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .progress-text {
            font-weight: bold;
            color: #007bff;
            margin: 0;
            font-size: 0.95em;
        }

        .welcome-content {
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .feature-card {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            backdrop-filter: blur(10px);
        }
    """),

    # JavaScript for UI interactions
    ui.tags.script("""
        let debugExpanded = true;

        function toggleDebugConsole() {
            const debugSection = document.getElementById('debug_section');
            const toggleButton = document.getElementById('toggle_debug');

            if (debugSection && toggleButton) {
                debugExpanded = !debugExpanded;

                if (debugExpanded) {
                    debugSection.classList.remove('collapsed');
                    toggleButton.textContent = '🔍 Hide Debug Console';
                } else {
                    debugSection.classList.add('collapsed');
                    toggleButton.textContent = '🔍 Debug Console';
                }
            }
        }

        function toggleSession(sessionId) {
            const session = document.getElementById(sessionId);
            if (!session) return;

            const isCollapsed = session.classList.contains('session-collapsed');
            if (isCollapsed) {
                session.classList.remove('session-collapsed');
            } else {
                session.classList.add('session-collapsed');
            }

            const button = session.querySelector('.btn-toggle');
            if (button) {
                button.textContent = isCollapsed ? '📕 Collapse' : '📖 Expand';
            }
        }

        function setupDebugConsole() {
            const toggleButton = document.getElementById('toggle_debug');
            if (toggleButton) {
                toggleButton.onclick = function(e) {
                    e.preventDefault();
                    toggleDebugConsole();
                    return false;
                };
                toggleButton.textContent = '🔍 Hide Debug Console';
            }
        }

        document.addEventListener('DOMContentLoaded', setupDebugConsole);
        setTimeout(setupDebugConsole, 1000);
    """)
)


# ---------------------------------------------------------------------------
# Enhanced server logic with structured data extraction
# ---------------------------------------------------------------------------

def server(input, output, session):
    workflow_status = reactive.Value("🔗 Enhanced System Ready")
    current_progress = reactive.Value("")
    query_sessions: reactive.Value[List[EnhancedQuerySession]] = reactive.Value([])
    debug_entries = reactive.Value([])

    # Internal debug entries list
    internal_debug_entries: List[str] = []
    debug_state = {"callback_running": False}

    # Debug callback
    def debug_callback(entry: str):
        if debug_state["callback_running"]:
            return

        debug_state["callback_running"] = True
        try:
            internal_debug_entries.append(entry)
            if len(internal_debug_entries) > 100:
                del internal_debug_entries[:-100]
            debug_entries.set(internal_debug_entries.copy())
        finally:
            debug_state["callback_running"] = False

    debug_logger_instance.set_callback(debug_callback)

    # Background AI workflow
    async def _workflow(query: str):
        return await ai_system.run_agentic_workflow(query)

    workflow_task = reactive.extended_task(_workflow)

    # Progress callback
    def progress_callback(step: ProgressStep):
        try:
            progress_msg = str(step)
            debug_info(f"Progress: {progress_msg}")
            current_progress.set(progress_msg)
        except Exception as e:
            debug_info(f"Progress callback error: {e}")
            current_progress.set(f"⏳ Working... ({datetime.now().strftime('%H:%M:%S')})")

    if ai_system:
        ai_system.set_progress_callback(progress_callback)

    # Force UI update when debug entries change
    @reactive.Effect
    def _debug_entries_changed():
        entries = debug_entries.get()

    # Initialize system status
    if ai_system:
        api_status = ["OpenAI ✅"]
        api_status.append("OpenWeatherMap One Call API 3.0 ✅" if OPENWEATHER_KEY else "Weather API (Free Fallback) ⚠️")
        api_status.append("FR24 API " + ("✅" if FLIGHTRADAR24_KEY else "❌"))
        api_status.append("Climate Intelligence ✅")
        api_status.append("Structured Data Display ✅")
        workflow_status.set(f"✅ Enhanced System Ready | {' | '.join(api_status)}")

        # Add initial debug entries
        debug_callback(
            f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] SUCCESS: Enhanced Agentic AI System with exclusive OpenAI city generation")
        debug_callback(
            f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] INFO: Agent decision logging enabled - focus on agentic activity")
        debug_callback(
            f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] INFO: System initialized with APIs: {' | '.join(api_status)}")
        if OPENWEATHER_KEY:
            debug_callback(
                f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] INFO: OpenWeatherMap One Call API 3.0 configured for extended forecasts")
        debug_callback(
            f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] INFO: No fallback city defaults - OpenAI exclusive source")
        debug_callback(
            f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] INFO: Enhanced debug console showing agentic decisions and temperature data")
    else:
        workflow_status.set("⚠️ Please configure OpenAI API key in .env file")
        debug_callback(
            f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] WARN: System waiting for OpenAI API key configuration")

    # Example button handlers
    @reactive.Effect
    @reactive.event(input.example1)
    def _example1():
        ui.update_text_area("user_query",
                            value="Find European destinations with temperatures between 18-25°C in 1 week, include comprehensive climate analysis, real flight data, and flight recommendations")

    @reactive.Effect
    @reactive.event(input.example2)
    def _example2():
        ui.update_text_area("user_query",
                            value="Find diverse Asian cities with warm weather next Friday, analyze real FlightRadar24 flight data and generate detailed 4-day weather-appropriate itineraries")

    @reactive.Effect
    @reactive.event(input.example3)
    def _example3():
        ui.update_text_area("user_query",
                            value="Find warm Mexican coastal destinations with 25-30°C weather in 5 days, analyze flight routes and provide scored flight recommendations")

    @reactive.Effect
    @reactive.event(input.example4)
    def _example4():
        ui.update_text_area("user_query",
                            value="Find cooler European destinations with 5-15°C in 10 days from Toronto, include flight analysis and weather anomaly alerts")

    @reactive.Effect
    @reactive.event(input.example5)
    def _example5():
        ui.update_text_area("user_query",
                            value="South American cities next month, check for climate deviations, real flight schedules, and comprehensive travel analysis")

    @reactive.Effect
    @reactive.event(input.example6)
    def _example6():
        ui.update_text_area("user_query",
                            value="Compare flights to diverse European cities in 2 weeks, show FlightRadar24 data, airline ratings, and climate intelligence")

    @reactive.Effect
    @reactive.event(input.example7)
    def _example7():
        ui.update_text_area("user_query",
                            value="Find 3 diverse European destinations with 20-27°C next Friday, include historical climate baselines, weather deviation analysis, real FlightRadar24 flight data with airline recommendations, and detailed 4-day weather-appropriate itineraries")

    # Session management
    @reactive.Effect
    @reactive.event(input.clear_all)
    def _clear_all():
        query_sessions.set([])
        current_progress.set("")
        debug_info("All query sessions cleared")
        workflow_status.set("🗑️ All sessions cleared")

    @reactive.Effect
    @reactive.event(input.expand_all)
    def _expand_all():
        sessions = query_sessions.get()
        for session in sessions:
            session.expanded = True
        query_sessions.set(sessions)
        debug_info("All sessions expanded")

    @reactive.Effect
    @reactive.event(input.collapse_all)
    def _collapse_all():
        sessions = query_sessions.get()
        for session in sessions:
            session.expanded = False
        query_sessions.set(sessions)
        debug_info("All sessions collapsed")

    @reactive.Effect
    @reactive.event(input.clear_debug)
    def _clear_debug():
        internal_debug_entries.clear()
        debug_entries.set([])
        debug_logger_instance.clear()
        debug_callback(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] SUCCESS: Debug log cleared")

    @reactive.Effect
    @reactive.event(input.test_debug)
    def _test_debug():
        current_time = datetime.now().strftime('%H:%M:%S')
        debug_callback(
            f"[{current_time}.123] AGENT-PARSER: Analyzing travel query: 'Find warm Mexican coastal cities...'")
        debug_callback(f"[{current_time}.456] TEMP-FORECAST: 27.5°C | HISTORICAL: 24.1°C")
        debug_callback(f"[{current_time}.789] CLIMATE-DECISION: MEETS criteria (27.5°C vs 25-30°C target)")
        debug_callback(f"[{current_time}.999] REASONING: Temperature within target range with excellent climate score")
        debug_callback(f"[{current_time}.234] SUCCESS: OpenWeatherMap One Call API 3.0 success for Puerto Vallarta")
        debug_callback(f"[{current_time}.567] AGENT-RESEARCHER: Generated 8 diverse cities across 1 regions")
        debug_callback(f"[{current_time}.890] INFO: Enhanced agent decision console is working correctly!")

    # Run workflow
    @reactive.Effect
    @reactive.event(input.run)
    def _on_run():
        if ai_system is None:
            workflow_status.set("❌ Please configure OpenAI API key in .env file")
            return

        query = input.user_query().strip()
        if not query:
            workflow_status.set("⚠️ Please enter a travel query")
            return

        debug_info(f"Starting new query: {query[:50]}...")
        new_session = EnhancedQuerySession(query, datetime.now())
        sessions = query_sessions.get()
        sessions.insert(0, new_session)
        query_sessions.set(sessions)

        workflow_status.set("🔄 Enhanced AI agents working…")
        current_progress.set("🚀 Initializing enhanced workflow…")

        workflow_task.invoke(query)

    # When workflow finishes
    @reactive.Effect
    def _on_done():
        try:
            msgs = workflow_task.result()
        except asyncio.CancelledError:
            return
        except Exception as e:
            if type(e).__name__.startswith("Silent"):
                return
            debug_info(f"❌ Workflow error ({type(e).__name__}): {e!r}")
            workflow_status.set(f"❌ Enhanced workflow failed: {type(e).__name__}")
            current_progress.set(f"❌ Error: {type(e).__name__}")
            return

        debug_success(f"Workflow completed with {len(msgs)} agent messages")

        current_sessions = query_sessions.get()
        target = current_sessions[0]
        target.messages = msgs
        target.completed = True

        target.extract_structured_data()

        parts = ["✅ Complete"]
        if target.recommended_cities:
            parts.append(f"{len(target.recommended_cities)} cities")
        if target.itineraries:
            parts.append(f"{len(target.itineraries)} itineraries")
        if target.climate_alerts:
            parts.append(f"{len(target.climate_alerts)} climate alerts")
        if target.flight_data.get("successful_calls", 0) > 0:
            parts.append("flight data")
        target.status = " | ".join(parts)

        query_sessions.set(current_sessions)
        current_progress.set("✅ Workflow completed with structured data extraction")
        workflow_status.set("✅ Enhanced AI workflow completed with structured data display")

    # Output functions
    @output
    @render.ui
    def system_status_display():
        status = workflow_status.get()
        if status.startswith("✅"):
            status_class = "system-status"
        elif status.startswith("⚠️"):
            status_class = "system-status warning"
        elif status.startswith("❌"):
            status_class = "system-status error"
        else:
            status_class = "system-status"

        return ui.div(
            ui.p(status, style="margin: 0; font-weight: 500; line-height: 1.3;"),
            class_=status_class
        )

    @output
    @render.ui
    def progress_display():
        progress = current_progress.get()
        if not progress:
            return ui.div()

        progress_width = "0%"
        if "Researcher" in progress and "(" in progress:
            try:
                match = re.search(r'\((\d+)/(\d+)', progress)
                if match:
                    current, total = int(match.group(1)), int(match.group(2))
                    percentage = (current / total) * 50
                    progress_width = f"{percentage}%"
                else:
                    progress_width = "25%"
            except:
                progress_width = "25%"
        elif "Analyzer" in progress:
            progress_width = "70%"
        elif "Synthesizer" in progress:
            progress_width = "85%" if "itineraries" in progress else "95%"
        elif "Complete" in progress:
            progress_width = "100%"
        else:
            progress_width = "10%"

        return ui.div(
            ui.div(
                ui.div(style=f"width: {progress_width};", class_="progress-bar"),
                class_="progress-bar-container"
            ),
            ui.p(progress, class_="progress-text"),
            class_="progress-container"
        )

    @output
    @render.ui
    def debug_console():
        entries = debug_entries.get()

        if not entries:
            return ui.div(
                ui.div("🔍 Enhanced Agent Decision Console Ready",
                       style="color: #28a745; font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid #dee2e6; padding-bottom: 5px;"),
                ui.div("• System initialized with enhanced agentic decision logging",
                       style="color: #6c757d; margin-bottom: 5px;"),
                ui.div("• Will show agent reasoning, climate decisions, and temperature data",
                       style="color: #6c757d; margin-bottom: 5px;"),
                ui.div("• Agent workflow progress and decision-making processes",
                       style="color: #6c757d; margin-bottom: 5px;"),
                ui.div("• No fallback city defaults - OpenAI exclusive source",
                       style="color: #6c757d; margin-bottom: 5px;"),
                ui.div("• Try the Mexico example to see enhanced regional recognition",
                       style="color: #6c757d; margin-bottom: 5px;"),
                ui.div(f"• Current time: {datetime.now().strftime('%H:%M:%S')}",
                       style="color: #adb5bd; margin-top: 10px; font-size: 0.9em;"),
                class_="debug-console"
            )

        entry_elements = []
        for entry in entries[-50:]:
            entry_style = "color: #212529; margin-bottom: 3px; padding: 2px 0;"
            if "SUCCESS:" in entry:
                entry_style = "color: #28a745; margin-bottom: 3px; padding: 2px 0; font-weight: bold;"
            elif "AGENT-" in entry or "AGENT:" in entry:
                entry_style = "color: #007bff; margin-bottom: 3px; padding: 2px 0; font-weight: bold;"
            elif "CLIMATE-" in entry or "TEMP-FORECAST:" in entry or "CLIMATE-DECISION:" in entry:
                entry_style = "color: #17a2b8; margin-bottom: 3px; padding: 2px 0; font-weight: bold;"
            elif "REASONING:" in entry:
                entry_style = "color: #6f42c1; margin-bottom: 3px; padding: 2px 0; font-style: italic;"
            elif "INFO:" in entry:
                entry_style = "color: #007bff; margin-bottom: 3px; padding: 2px 0;"
            elif "WARN:" in entry or "WARNING:" in entry:
                entry_style = "color: #fd7e14; margin-bottom: 3px; padding: 2px 0; font-weight: bold;"
            elif "ERROR:" in entry:
                entry_style = "color: #dc3545; margin-bottom: 3px; padding: 2px 0; font-weight: bold;"
            elif "DEBUG:" in entry:
                entry_style = "color: #6c757d; margin-bottom: 3px; padding: 2px 0;"

            entry_elements.append(ui.div(entry, style=entry_style))

        return ui.div(
            ui.div(f"🔍 Enhanced Agent Decision Console - {len(entries)} entries",
                   style="color: #28a745; font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid #dee2e6; padding-bottom: 5px;"),
            ui.div("(Showing agentic decisions, climate analysis, and temperature data)",
                   style="color: #adb5bd; font-size: 0.85em; margin-bottom: 10px; font-style: italic;"),
            *reversed(entry_elements),
            class_="debug-console"
        )

    @output
    @render.ui
    def sessions_display():
        sessions = query_sessions.get()
        if not sessions:
            return ui.div(
                ui.div(
                    ui.h4("🌍 Enhanced Climate Intelligence & Flight System",
                          style="color: white; margin-bottom: 20px;"),
                    ui.p("Ready to provide structured travel recommendations with comprehensive data display!",
                         style="color: rgba(255,255,255,0.9); margin-bottom: 15px;"),
                    ui.div(
                        ui.div(
                            ui.h6("🌡️ Weather Analysis Summary", style="color: white; margin-bottom: 8px;"),
                            ui.p("• Cities analyzed, average scores, climate alerts count",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• Visual statistics dashboard with key metrics",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            class_="feature-card"
                        ),
                        ui.div(
                            ui.h6("🚨 Climate Alerts", style="color: white; margin-bottom: 8px;"),
                            ui.p("• Unusual weather conditions with severity levels",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• Specific recommendations for unusual weather",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            class_="feature-card"
                        ),
                        ui.div(
                            ui.h6("🏆 Recommended Cities", style="color: white; margin-bottom: 8px;"),
                            ui.p("• Top 5 destinations with weather & climate scores",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• Flight info: 'Lufthansa LH123 | 8h 30m | Live Data'",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• Temperature, weather description, meets criteria",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            class_="feature-card"
                        ),
                        ui.div(
                            ui.h6("✈️ Flight Intelligence", style="color: white; margin-bottom: 8px;"),
                            ui.p("• Real-time FlightRadar24 data vs estimates",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• Flight analysis success rates and data quality",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            class_="feature-card"
                        ),
                        ui.div(
                            ui.h6("🗓️ Weather Itineraries", style="color: white; margin-bottom: 8px;"),
                            ui.p("• Day-by-day weather-appropriate activity plans",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• Morning/afternoon/evening recommendations",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• Weather gear and backup plan suggestions",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            class_="feature-card"
                        ),
                        class_="feature-grid"
                    ),
                    ui.p("👆 Click any example above to see all sections populated with real data below!",
                         style="color: white; text-align: center; margin-top: 30px; font-weight: 500;"),
                    ui.p("💡 Also try the '🔍 Debug Console' button above to see real-time processing",
                         style="color: rgba(255,255,255,0.8); text-align: center; margin-top: 10px; font-size: 0.9em;"),
                    class_="welcome-content"
                )
            )

        session_elements = []
        for session in sessions:
            expanded_class = "" if session.expanded else "session-collapsed"
            toggle_text = "📕 Collapse" if session.expanded else "📖 Expand"

            session_header = ui.div(
                ui.div(
                    ui.h5(f"Query: {session.query[:60]}{'...' if len(session.query) > 60 else ''}",
                          style="margin: 0; color: white;"),
                    ui.p(f"{session.display_time} | {session.status}",
                         style="margin: 5px 0 0 0; color: rgba(255,255,255,0.8); font-size: 0.9em;")
                ),
                ui.button(toggle_text, onclick=f"toggleSession('{session.session_id}')", class_="btn-toggle"),
                class_="session-header"
            )

            if session.messages and session.completed:
                content_elements = []

                # Weather Analysis Summary
                if session.weather_analysis:
                    wa = session.weather_analysis
                    content_elements.append(
                        ui.div(
                            ui.div("🌡️ Weather Analysis Summary", class_="data-section-header"),
                            ui.div(
                                ui.div(
                                    ui.div(f"{wa.get('cities_processed', 0)}", class_="weather-stat-value"),
                                    ui.div("Cities Analyzed", class_="weather-stat-label"),
                                    class_="weather-stat"
                                ),
                                ui.div(
                                    ui.div(f"{wa.get('meeting_criteria', 0)}", class_="weather-stat-value"),
                                    ui.div("Meet Criteria", class_="weather-stat-label"),
                                    class_="weather-stat"
                                ),
                                ui.div(
                                    ui.div(f"{wa.get('average_score', 0):.1f}/10", class_="weather-stat-value"),
                                    ui.div("Avg Score", class_="weather-stat-label"),
                                    class_="weather-stat"
                                ),
                                ui.div(
                                    ui.div(f"{wa.get('alerts_count', 0)}", class_="weather-stat-value"),
                                    ui.div("Climate Alerts", class_="weather-stat-label"),
                                    class_="weather-stat"
                                ),
                                class_="weather-summary"
                            ),
                            class_="data-section"
                        )
                    )

                # Climate Alerts
                if session.climate_alerts:
                    alert_elements = []
                    for alert in session.climate_alerts:
                        alert_class = "climate-alert extreme" if alert["level"] == "extreme" else "climate-alert"
                        alert_elements.append(
                            ui.div(
                                ui.div(ui.strong(f"🚨 {alert['city']} - {alert['level'].upper()} ALERT"),
                                       style="display: block; margin-bottom: 8px;"),
                                ui.p(alert["message"], style="margin: 5px 0;"),
                                ui.ul(*[ui.li(rec) for rec in alert["recommendations"][:2]]) if alert[
                                    "recommendations"] else "",
                                class_=alert_class
                            )
                        )

                    content_elements.append(
                        ui.div(
                            ui.div("🚨 Climate Alerts", class_="data-section-header"),
                            ui.div(*alert_elements, class_="data-section-content"),
                            class_="data-section"
                        )
                    )

                # Recommended Cities
                if session.recommended_cities:
                    city_elements = []
                    for i, city in enumerate(session.recommended_cities[:5], 1):
                        city_name = city.get("city", "Unknown")
                        country = city.get("country", "")
                        temp_avg = city.get("temp_avg", 0)
                        weather_desc = city.get("weather_desc", "")
                        overall_score = city.get("overall_score", 0)
                        climate_score = city.get("climate_score", 0)
                        flight_score = city.get("flight_score", 0)
                        meets_criteria = city.get("meets_criteria", False)

                        flight_info_elem = ""
                        if city.get("flight_analysis_success"):
                            recommendations = city.get("recommendations", [])
                            if recommendations:
                                best_flight = recommendations[0]
                                data_source = "🔴 Live Data" if city.get(
                                    "real_flight_data_available") else "📊 Estimated"
                                flight_info_elem = ui.div(
                                    ui.p(f"✈️ Best Flight: {best_flight.airline} {best_flight.flight_number}",
                                         style="margin: 0; font-weight: bold;"),
                                    ui.p(
                                        f"Duration: {best_flight.duration_display} | Score: {best_flight.overall_score:.1f}/10 | {data_source}",
                                        style="margin: 2px 0; font-size: 0.9em;"),
                                    ui.p(f"Route: {best_flight.origin_airport} → {best_flight.destination_airport}",
                                         style="margin: 0; font-size: 0.85em; color: #666;"),
                                    class_="flight-info"
                                )

                        status_icon = "🎯" if meets_criteria else "⭐"
                        city_elements.append(
                            ui.div(
                                ui.div(
                                    ui.div(f"{status_icon} {city_name}, {country}", class_="city-name"),
                                    ui.div(f"{overall_score:.1f}/10", class_="city-score"),
                                    class_="city-header"
                                ),
                                ui.div(
                                    ui.div(f"🌡️ {temp_avg:.1f}°C - {weather_desc}", class_="city-detail"),
                                    ui.div(f"🌦️ Climate Score: {climate_score:.1f}/10", class_="city-detail"),
                                    ui.div(f"✈️ Flight Score: {flight_score:.1f}/10", class_="city-detail"),
                                    ui.div(f"{'✅ Meets criteria' if meets_criteria else '⚠️ Outside criteria'}",
                                           class_="city-detail"),
                                    class_="city-details"
                                ),
                                flight_info_elem,
                                class_="city-card"
                            )
                        )

                    content_elements.append(
                        ui.div(
                            ui.div("🏆 Recommended Cities", class_="data-section-header"),
                            ui.div(*city_elements, class_="data-section-content"),
                            class_="data-section"
                        )
                    )

                # Flight Analysis Summary
                if session.flight_data:
                    fd = session.flight_data
                    flight_summary = ui.div(
                        ui.p(
                            f"✈️ Flight Analysis: {fd.get('flight_successes', 0)}/{fd.get('total_analyzed', 0)} destinations analyzed"),
                        ui.p(
                            f"📡 FlightRadar24: {'Available' if fd.get('fr24_available') else 'Not configured'} | Live data: {fd.get('successful_calls', 0)} calls"),
                        ui.p(
                            f"🎯 Data Quality: {'High (real-time)' if fd.get('successful_calls', 0) > 0 else 'Estimated (geographic)'}"),
                        style="margin: 0;"
                    )

                    content_elements.append(
                        ui.div(
                            ui.div("✈️ Flight Intelligence Summary", class_="data-section-header"),
                            ui.div(flight_summary, class_="data-section-content"),
                            class_="data-section"
                        )
                    )

                # Weather Itineraries
                if session.itineraries:
                    itinerary_elements = []
                    for i, itinerary in enumerate(session.itineraries, 1):
                        day_sections = []
                        lines = itinerary.split('\n')
                        current_day = []

                        for line in lines:
                            line = line.strip()
                            if line.startswith('**DAY') or line.startswith('🗓️'):
                                if current_day:
                                    day_sections.append('\n'.join(current_day))
                                current_day = [line]
                            elif line and current_day:
                                current_day.append(line)

                        if current_day:
                            day_sections.append('\n'.join(current_day))

                        day_elements = []
                        for day_content in day_sections[:4]:
                            if '**DAY' in day_content or '🗓️' in day_content:
                                day_lines = day_content.split('\n')
                                day_title = day_lines[0].replace('**', '').replace('🗓️', '').strip()
                                day_details = '\n'.join(day_lines[1:])

                                day_elements.append(
                                    ui.div(
                                        ui.div(day_title, class_="itinerary-day-title"),
                                        ui.pre(day_details,
                                               style="white-space: pre-wrap; margin: 0; font-size: 0.9em; line-height: 1.4;"),
                                        class_="itinerary-day"
                                    )
                                )

                        itinerary_elements.append(
                            ui.div(
                                ui.h6(f"Itinerary {i}",
                                      style="color: #6a1b9a; margin-bottom: 15px; font-weight: bold;"),
                                *day_elements,
                                class_="itinerary-container"
                            )
                        )

                    content_elements.append(
                        ui.div(
                            ui.div("🗓️ Weather-Appropriate Itineraries", class_="data-section-header"),
                            ui.div(*itinerary_elements, class_="data-section-content"),
                            class_="data-section"
                        )
                    )

                session_content = ui.div(*content_elements, class_="session-content")
            else:
                session_content = ui.div(
                    ui.p("🔄 Processing query..." if not session.completed else "❌ No structured data available",
                         style="text-align: center; padding: 20px; color: #666;"),
                    class_="session-content"
                )

            session_elements.append(
                ui.div(
                    session_header,
                    session_content,
                    class_=f"session-card {expanded_class}",
                    id=session.session_id
                )
            )

        return ui.div(*session_elements)


# ---------------------------------------------------------------------------
# Enhanced Shiny app instance
# ---------------------------------------------------------------------------

app = App(app_ui, server)

if __name__ == "__main__":
    print("🚀 Starting Enhanced Agentic Travel Planner with Comprehensive Fixes...")
    print("📍 Features: Structured Data Extraction + All Previous Enhancements")
    print("🔧 FIXED: Debug console - light background, dark text, agent decision focus")
    print("🌡️ FIXED: OpenWeatherMap One Call API 3.0 for extended forecasts")
    print("🤖 FIXED: Robust climate AI with better JSON parsing and fallbacks")
    print("🌍 FIXED: OpenAI-exclusive city generation, no fallback defaults")
    print("🎯 ENHANCED: Agent decision logging with temperature data prominence")
    print("✅ ALL ISSUES RESOLVED: Mexico queries will get Mexican cities, debug shows agentic activity")
    app.run(debug=True)