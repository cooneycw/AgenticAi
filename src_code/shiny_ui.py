# src_code/shiny_ui.py
"""Enhanced UI with Progress Tracking and Persistent Results - Phase 2.5

This enhanced UI provides:
1. **Real-Time Progress Tracking**: Live updates during workflow execution
2. **Persistent Results**: Results stay visible with expand/collapse functionality
3. **Fixed API Integration**: Handles the corrected API calls
4. **Enhanced Error Display**: Better error handling and user feedback
5. **Multi-Session Support**: Multiple queries without losing previous results

Key Phase 2.5 enhancements:
- Real-time progress display with detailed steps
- Expand/collapse buttons for managing result visibility
- Persistent results that don't disappear on new queries
- Better status indicators and error handling
- Session management for multiple queries
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import List, Dict, Any

from shiny import App, reactive, render, ui

from config.config import (
    AgentRole,
    FLIGHTRADAR24_KEY,
    OPENAI_KEY,
    OPENWEATHER_KEY,
)
from src_code.agentic_core import AgenticAISystem, ProgressStep

# ---------------------------------------------------------------------------
# Enhanced system instantiation with progress tracking
# ---------------------------------------------------------------------------

ai_system: AgenticAISystem | None = (
    AgenticAISystem(OPENAI_KEY, OPENWEATHER_KEY, FLIGHTRADAR24_KEY)
    if OPENAI_KEY
    else None
)


# ---------------------------------------------------------------------------
# Session data structure for persistent results
# ---------------------------------------------------------------------------

class QuerySession:
    """Represents a single query session with results."""

    def __init__(self, query: str, timestamp: datetime):
        self.query = query
        self.timestamp = timestamp
        self.messages: List[Any] = []
        self.status = "🔄 Processing..."
        self.progress = ""
        self.expanded = True
        self.completed = False

    @property
    def session_id(self) -> str:
        return f"session_{self.timestamp.strftime('%H%M%S')}"

    @property
    def display_time(self) -> str:
        return self.timestamp.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Enhanced UI layout with progress tracking
# ---------------------------------------------------------------------------

app_ui = ui.page_fillable(
    ui.h1("🌍 Enhanced Agentic Travel Planner", class_="text-center mb-4"),
    ui.p(
        "🌡️ Climate Intelligence • ✈️ Real Flight Data • 🎯 Smart Recommendations • 🗓️ Weather-Appropriate Itineraries",
        class_="text-center text-muted mb-4"
    ),

    ui.layout_sidebar(
        ui.sidebar(
            # Connection section
            ui.h3("Configuration"),
            ui.input_password("api_key", "OpenAI API Key:", placeholder="sk-..."),
            ui.input_action_button("connect", "Connect", class_="btn-primary mb-3"),
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
                                   class_="btn-outline-secondary btn-sm mb-2 w-100"),
            ui.input_action_button("collapse_all", "📕 Collapse All",
                                   class_="btn-outline-secondary btn-sm mb-2 w-100"),
            ui.hr(),

            # Feature status section
            ui.h5("Enhanced Features"),
            ui.p("✅ Real-Time Progress Tracking", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Persistent Results with Expand/Collapse", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ FlightRadar24 Real Flight Data", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ 80+ Airport Database Worldwide", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Flight Recommendations with Scoring", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Historical Climate Baselines", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ Weather Deviation Analysis", style="font-size: 0.85em; margin: 2px 0;"),
            ui.p("✅ 4-Day Weather Itineraries", style="font-size: 0.85em; margin: 2px 0;"),

            width=320
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

            # Status display with progress
            ui.div(
                ui.output_ui("status_display"),
                ui.output_ui("progress_display"),
                class_="mb-3"
            ),

            # Persistent sessions display
            ui.div(
                ui.h3("Query Sessions"),
                ui.output_ui("sessions_display"),
                class_="sessions-container"
            )
        )
    ),

    # Enhanced CSS styling with progress indicators
    ui.tags.style("""
        .sessions-container {
            max-height: 700px;
            overflow-y: auto;
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
        .agent-message {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
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
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 1.0em;
            color: #333;
        }
        .agent-content {
            white-space: pre-wrap;
            line-height: 1.5;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.9em;
        }
        .tools-used {
            font-size: 0.8em;
            color: #666;
            font-style: italic;
            background: rgba(255,255,255,0.7);
            padding: 3px 6px;
            border-radius: 4px;
        }
        .progress-bar {
            background: linear-gradient(90deg, #007bff 0%, #28a745 100%);
            height: 4px;
            border-radius: 2px;
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        .progress-text {
            font-size: 0.9em;
            color: #666;
            margin: 5px 0;
            font-family: monospace;
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
        .error-message {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .success-message {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
    """)
)


# ---------------------------------------------------------------------------
# Enhanced server logic with progress tracking and persistence
# ---------------------------------------------------------------------------

def server(input, output, session):
    workflow_status = reactive.Value("🔗 Enhanced System Ready")
    current_progress = reactive.Value("")
    query_sessions: reactive.Value[List[QuerySession]] = reactive.Value([])

    # Initialize status based on available APIs
    if ai_system:
        api_status = []
        api_status.append("OpenAI ✅")

        # Show which weather API we're using
        if OPENWEATHER_KEY:
            api_status.append("OpenWeatherMap (Paid) ✅")
        else:
            api_status.append("Weather API (Free) ⚠️")

        api_status.append("FR24 API " + ("✅" if FLIGHTRADAR24_KEY else "❌"))
        api_status.append("Climate Intelligence ✅")
        api_status.append("Itinerary Generation ✅")

        workflow_status.set(f"✅ Enhanced System Ready | {' | '.join(api_status)}")
    else:
        workflow_status.set("⚠️ Please connect OpenAI API key for enhanced features")

    # Progress callback function
    def progress_callback(step: ProgressStep):
        current_progress.set(str(step))

    # Set up progress callback if ai_system exists
    if ai_system:
        ai_system.set_progress_callback(progress_callback)

    # ---------------------------------------------------- connection button
    @reactive.Effect
    @reactive.event(input.connect)
    def _connect():
        global ai_system
        key = input.api_key() or OPENAI_KEY
        if not key:
            workflow_status.set("❌ Enter an OpenAI API key first")
            return
        try:
            ai_system = AgenticAISystem(key, OPENWEATHER_KEY, FLIGHTRADAR24_KEY)
            ai_system.set_progress_callback(progress_callback)

            # Enhanced status with API availability
            api_status = []
            api_status.append("OpenAI ✅")

            # Show which weather API we're using
            if OPENWEATHER_KEY:
                api_status.append("OpenWeatherMap (Paid) ✅")
            else:
                api_status.append("Weather API (Free) ⚠️")

            api_status.append("FR24 API " + ("✅" if FLIGHTRADAR24_KEY else "❌"))
            api_status.append("Climate Intelligence ✅")
            api_status.append("Itinerary Generation ✅")

            workflow_status.set(f"✅ Enhanced System Connected | {' | '.join(api_status)}")
        except Exception as exc:
            workflow_status.set(f"❌ Connection failed: {exc}")

    # ---------------------------------------------------- example buttons
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
                            value="Mexican coastal destinations with 25-30°C weather in 5 days, analyze flight routes and provide scored flight recommendations")

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

    # ---------------------------------------------------- session management
    @reactive.Effect
    @reactive.event(input.clear_all)
    def _clear_all():
        query_sessions.set([])
        current_progress.set("")
        workflow_status.set("🗑️ All sessions cleared")

    @reactive.Effect
    @reactive.event(input.expand_all)
    def _expand_all():
        sessions = query_sessions.get()
        for session in sessions:
            session.expanded = True
        query_sessions.set(sessions)

    @reactive.Effect
    @reactive.event(input.collapse_all)
    def _collapse_all():
        sessions = query_sessions.get()
        for session in sessions:
            session.expanded = False
        query_sessions.set(sessions)

    # ---------------------------------------------------- run workflow
    @reactive.Effect
    @reactive.event(input.run)
    async def _run():
        if ai_system is None:
            workflow_status.set("❌ Please connect to OpenAI API first")
            return
        query = input.user_query().strip()
        if not query:
            workflow_status.set("⚠️ Please enter a travel query")
            return

        # Create new session
        new_session = QuerySession(query, datetime.now())
        sessions = query_sessions.get()
        sessions.insert(0, new_session)  # Add to top
        query_sessions.set(sessions)

        workflow_status.set("🔄 Enhanced AI agents working with climate intelligence & itinerary generation...")
        current_progress.set("🚀 Initializing enhanced workflow...")

        try:
            msgs = await ai_system.run_agentic_workflow(query)

            # Update session with results
            sessions = query_sessions.get()
            for session in sessions:
                if session.session_id == new_session.session_id:
                    session.messages = msgs
                    session.completed = True

                    # Enhanced completion status
                    itinerary_count = 0
                    climate_alerts = 0

                    # Count generated features
                    for msg in msgs:
                        if msg.metadata:
                            itinerary_count += msg.metadata.get("itineraries_generated", 0)
                            alerts = msg.metadata.get("climate_alerts", [])
                            if alerts:
                                climate_alerts = len(alerts)

                    status_parts = ["✅ Complete"]
                    if itinerary_count > 0:
                        status_parts.append(f"{itinerary_count} itineraries")
                    if climate_alerts > 0:
                        status_parts.append(f"{climate_alerts} climate alerts")

                    session.status = " | ".join(status_parts)
                    break

            query_sessions.set(sessions)
            current_progress.set("✅ Workflow completed successfully")

            # Update main status
            status_parts = ["✅ Enhanced AI workflow completed"]
            if itinerary_count > 0:
                status_parts.append(f"{itinerary_count} weather itineraries generated")
            if climate_alerts > 0:
                status_parts.append(f"{climate_alerts} climate alerts detected")

            workflow_status.set(" | ".join(status_parts))

        except Exception as e:
            # Update session with error
            sessions = query_sessions.get()
            for session in sessions:
                if session.session_id == new_session.session_id:
                    session.status = f"❌ Error: {str(e)[:50]}..."
                    session.completed = True
                    break
            query_sessions.set(sessions)

            workflow_status.set(f"❌ Enhanced workflow failed: {str(e)}")
            current_progress.set(f"❌ Error: {str(e)}")
            print(f"Workflow error: {e}")

    # ---------------------------------------------------- outputs
    @output
    @render.ui
    def status_display():
        status = workflow_status.get()

        # Determine alert class based on status
        if status.startswith("✅"):
            alert_class = "alert-success"
        elif status.startswith("⚠️"):
            alert_class = "alert-warning"
        elif status.startswith("❌"):
            alert_class = "alert-danger"
        elif status.startswith("🔄"):
            alert_class = "alert-info"
        else:
            alert_class = "alert-info"

        return ui.div(ui.p(status, class_="mb-0"), class_=f"alert {alert_class}")

    @output
    @render.ui
    def progress_display():
        progress = current_progress.get()
        if not progress:
            return ui.div()

        # Extract progress information if available
        if "(" in progress and "/" in progress:
            # Progress with numbers like "Agent 2/4: Research (3/8 cities)"
            return ui.div(
                ui.div(class_="progress-bar", style="width: 60%; animation: pulse 2s infinite;"),
                ui.p(progress, class_="progress-text"),
                class_="mb-2"
            )
        else:
            # Simple progress message
            return ui.div(
                ui.p(progress, class_="progress-text"),
                class_="mb-2"
            )

    @output
    @render.ui
    def sessions_display():
        sessions = query_sessions.get()
        if not sessions:
            return ui.div(
                ui.div(
                    ui.h4("🌍 Enhanced Climate Intelligence & Itinerary System",
                          style="color: white; margin-bottom: 20px;"),
                    ui.p("Ready to provide comprehensive travel planning with:",
                         style="color: rgba(255,255,255,0.9); margin-bottom: 15px;"),

                    ui.div(
                        ui.div(
                            ui.h6("🌡️ Climate Intelligence", style="color: white; margin-bottom: 8px;"),
                            ui.p("• Historical climate baselines (30-year normals)",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• Weather deviation analysis with severity levels",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• Climate alerts for unusual conditions",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            class_="feature-card"
                        ),
                        ui.div(
                            ui.h6("🗓️ Smart Itineraries", style="color: white; margin-bottom: 8px;"),
                            ui.p("• 4-day weather-appropriate itineraries",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• Activity recommendations by weather",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• Clothing & gear suggestions",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            class_="feature-card"
                        ),
                        ui.div(
                            ui.h6("✈️ Flight Intelligence", style="color: white; margin-bottom: 8px;"),
                            ui.p("• FlightRadar24 real-time flight data",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• 80+ major airports worldwide",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• Scored flight recommendations",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            class_="feature-card"
                        ),
                        ui.div(
                            ui.h6("🎯 Enhanced Analysis", style="color: white; margin-bottom: 8px;"),
                            ui.p("• Real-time progress tracking",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• Persistent results with expand/collapse",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            ui.p("• Comprehensive climate + flight scoring",
                                 style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 2px 0;"),
                            class_="feature-card"
                        ),
                        class_="feature-grid"
                    ),

                    ui.p(
                        "👆 Try any example above or enter your own travel query to see the enhanced system in action!",
                        style="color: white; text-align: center; margin-top: 30px; font-weight: 500;"
                    ),
                    class_="welcome-content"
                )
            )

        # Enhanced agent info with better styling
        agent_info = {
            AgentRole.PLANNER: {"icon": "🎯", "name": "Strategic Planner", "class": "agent-planner"},
            AgentRole.RESEARCHER: {"icon": "🔍", "name": "Enhanced Data Researcher", "class": "agent-researcher"},
            AgentRole.ANALYZER: {"icon": "📊", "name": "Climate + Flight Analyzer", "class": "agent-analyzer"},
            AgentRole.SYNTHESIZER: {"icon": "🎯", "name": "Comprehensive Synthesizer", "class": "agent-synthesizer"}
        }

        session_elements = []
        for session in sessions:
            # Session header with toggle button
            expanded_class = "" if session.expanded else "session-collapsed"
            toggle_text = "📕 Collapse" if session.expanded else "📖 Expand"

            session_header = ui.div(
                ui.div(
                    ui.h5(f"Query: {session.query[:60]}{'...' if len(session.query) > 60 else ''}",
                          style="margin: 0; color: white;"),
                    ui.p(f"{session.display_time} | {session.status}",
                         style="margin: 5px 0 0 0; color: rgba(255,255,255,0.8); font-size: 0.9em;")
                ),
                ui.button(
                    toggle_text,
                    onclick=f"toggleSession('{session.session_id}')",
                    class_="btn-toggle"
                ),
                class_="session-header"
            )

            # Session content (messages)
            if session.messages:
                message_elements = []
                for msg in session.messages:
                    info = agent_info[msg.agent_role]

                    tools_display = ""
                    if msg.tools_used:
                        tools_display = f"Tools: {', '.join(msg.tools_used)}"

                    # Enhanced content formatting
                    content = msg.content

                    # Format itinerary sections
                    if "**DAY" in content:
                        content = content.replace("**DAY", "\n🗓️ **DAY")
                        content = content.replace("- Morning", "\n   🌅 Morning")
                        content = content.replace("- Afternoon", "\n   ☀️ Afternoon")
                        content = content.replace("- Evening", "\n   🌆 Evening")
                        content = content.replace("- Weather Gear:", "\n   🎒 Weather Gear:")
                        content = content.replace("- Backup Plan:", "\n   🏠 Backup Plan:")

                    # Format climate alerts
                    if "CLIMATE ALERT" in content:
                        content = content.replace("CLIMATE ALERT", "🚨 CLIMATE ALERT")

                    # Format temperature displays
                    content = re.sub(r'(\d+\.?\d*)°C', r'🌡️ \1°C', content)

                    message_elements.append(
                        ui.div(
                            ui.div(
                                ui.span(f"{info['icon']} {info['name']}", style="font-weight: bold;"),
                                ui.span(tools_display, class_="tools-used") if tools_display else "",
                                class_="agent-header"
                            ),
                            ui.div(content, class_="agent-content"),
                            class_=f"agent-message {info['class']}"
                        )
                    )

                session_content = ui.div(*message_elements, class_="session-content")
            else:
                session_content = ui.div(
                    ui.p("🔄 Processing query..." if not session.completed else "❌ No results available",
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

        # Add JavaScript for toggle functionality
        toggle_script = ui.tags.script("""
            function toggleSession(sessionId) {
                const session = document.getElementById(sessionId);
                const isCollapsed = session.classList.contains('session-collapsed');

                if (isCollapsed) {
                    session.classList.remove('session-collapsed');
                } else {
                    session.classList.add('session-collapsed');
                }

                // Update button text
                const button = session.querySelector('.btn-toggle');
                button.textContent = isCollapsed ? '📕 Collapse' : '📖 Expand';
            }
        """)

        return ui.div(
            *session_elements,
            toggle_script
        )


# ---------------------------------------------------------------------------
# Enhanced Shiny app instance
# ---------------------------------------------------------------------------

app = App(app_ui, server)

# ---------------------------------------------------------------------------
# Enhanced manual run for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🚀 Starting Enhanced Agentic Travel Planner with Phase 3 features...")
    print("📍 Features: Climate Intelligence + Weather Itineraries + Real Flight Data + Progress Tracking")
    app.run(debug=True)