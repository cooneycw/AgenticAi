# step1_config_models.py
"""Core configuration constants, enums, and data models used across the
refactored Agentic AI Travel application.

This standalone module can be imported by later components. Loading
environment variables here ensures that they are available everywhere
without re‑loading .env multiple times.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import os
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & API keys
# ---------------------------------------------------------------------------

# Load environment variables once, at import time.
load_dotenv()

OPENAI_KEY: Optional[str] = os.getenv("OPENAI_KEY")
OPENWEATHER_KEY: Optional[str] = os.getenv("OPENWEATHER_KEY")
FLIGHTRADAR24_KEY: Optional[str] = os.getenv("FR24_KEY")

# ---------------------------------------------------------------------------
# Core enumerations
# ---------------------------------------------------------------------------

class AgentRole(Enum):
    """Roles played by the autonomous agents in the workflow."""

    PLANNER = "planner"
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    SYNTHESIZER = "synthesizer"


# ---------------------------------------------------------------------------
# Dataclass models used throughout the system
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    """Single message exchanged within the agentic workflow."""

    agent_role: AgentRole
    content: str
    timestamp: datetime
    tools_used: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TaskResult:
    """Generic wrapper for an asynchronous task outcome."""

    success: bool
    data: Any
    error: Optional[str] = None


@dataclass
class TravelQuery:
    """Structured representation of a parsed user travel request."""

    regions: List[str]
    forecast_date: str
    forecast_date_parsed: datetime
    temperature_range: Tuple[float, float]
    origin_city: str
    additional_criteria: List[str]
    raw_query: str


@dataclass
class ClimateBaseline:
    """Historical climate baseline statistics for a specific city/month."""

    city: str
    month: int
    avg_temp_high: float
    avg_temp_low: float
    avg_temp_mean: float
    avg_humidity: float
    avg_precipitation_days: int
    typical_weather_patterns: List[str]
    climate_zone: str
    data_source: str = "ai_historical_analysis"


@dataclass
class WeatherDeviation:
    """Quantified deviation of a forecast from historical norms."""

    temperature_deviation_c: float
    humidity_deviation_pct: float
    deviation_severity: str  # "normal", "slight", "significant", "extreme"
    deviation_context: str
    traveler_impact: str
    confidence_level: str


# ---------------------------------------------------------------------------
# Public re‑exports
# ---------------------------------------------------------------------------

__all__ = [
    "AgentRole",
    "AgentMessage",
    "TaskResult",
    "TravelQuery",
    "ClimateBaseline",
    "WeatherDeviation",
    "OPENAI_KEY",
    "OPENWEATHER_KEY",
    "FLIGHTRADAR24_KEY",
]
