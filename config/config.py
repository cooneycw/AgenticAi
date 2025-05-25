# config/config.py
"""Enhanced configuration constants, enums, and data models - Phase 1 Upgrade

This enhanced configuration module includes:
- Extended ClimateBaseline with rich metadata fields
- Enhanced WeatherDeviation with confidence scoring
- Additional helper methods for climate analysis
- Backward compatibility with existing code

Key enhancements:
- ClimateBaseline now includes sunshine hours, records, reliability scores
- WeatherDeviation has enhanced severity levels and confidence tracking
- Added utility methods for climate alert detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


class DeviationSeverity(Enum):
    """Severity levels for weather deviations from historical norms."""

    NORMAL = "normal"
    SLIGHT = "slight"
    SIGNIFICANT = "significant"
    EXTREME = "extreme"


class ClimateZone(Enum):
    """Common climate zone classifications."""

    MEDITERRANEAN = "Mediterranean"
    CONTINENTAL = "Continental"
    OCEANIC = "Oceanic"
    TROPICAL = "Tropical"
    ARID = "Arid"
    SUBARCTIC = "Subarctic"
    TEMPERATE = "Temperate"
    UNKNOWN = "Unknown"


# ---------------------------------------------------------------------------
# Enhanced dataclass models
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
    """Enhanced historical climate baseline with comprehensive metadata.

    This enhanced version includes:
    - Core temperature and humidity statistics
    - Weather pattern descriptions and climate zone classification
    - Extended metadata for sunshine, wind, and record temperatures
    - Data reliability scoring and source tracking
    """

    # Core required fields
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

    # Enhanced metadata fields (optional, can be added dynamically)
    seasonal_notes: Optional[str] = None
    record_high: Optional[float] = None
    record_low: Optional[float] = None
    sunshine_hours: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    data_reliability: Optional[str] = None  # "high", "medium", "low"

    def __post_init__(self):
        """Validate and normalize data after initialization."""
        # Ensure temperature consistency
        if self.avg_temp_mean is None:
            self.avg_temp_mean = (self.avg_temp_high + self.avg_temp_low) / 2

        # Normalize climate zone
        if isinstance(self.climate_zone, str):
            # Try to match to known climate zones
            zone_lower = self.climate_zone.lower()
            for known_zone in ClimateZone:
                if zone_lower in known_zone.value.lower():
                    self.climate_zone = known_zone.value
                    break

    @property
    def temperature_range(self) -> float:
        """Calculate the typical temperature range for this location/month."""
        return self.avg_temp_high - self.avg_temp_low

    @property
    def is_reliable(self) -> bool:
        """Check if this baseline has high data reliability."""
        return self.data_reliability == "high"

    def get_comfort_rating(self, target_temp: float) -> str:
        """Get comfort rating for a target temperature vs this baseline."""
        if self.avg_temp_low <= target_temp <= self.avg_temp_high:
            return "excellent"
        elif abs(target_temp - self.avg_temp_mean) <= 3:
            return "good"
        elif abs(target_temp - self.avg_temp_mean) <= 6:
            return "fair"
        else:
            return "poor"


@dataclass
class WeatherDeviation:
    """Enhanced weather deviation analysis with traveler impact assessment.

    This enhanced version includes:
    - Quantified temperature and humidity deviations
    - Severity classification with traveler-specific context
    - Confidence scoring for reliability assessment
    - Actionable impact descriptions for trip planning
    """

    temperature_deviation_c: float
    humidity_deviation_pct: float
    deviation_severity: str  # "normal", "slight", "significant", "extreme"
    deviation_context: str
    traveler_impact: str
    confidence_level: str  # "high", "medium", "low"

    def __post_init__(self):
        """Validate and enhance deviation data."""
        # Ensure severity is valid
        valid_severities = [s.value for s in DeviationSeverity]
        if self.deviation_severity not in valid_severities:
            self.deviation_severity = "normal"

        # Ensure confidence level is valid
        valid_confidence = ["high", "medium", "low"]
        if self.confidence_level not in valid_confidence:
            self.confidence_level = "medium"

    @property
    def is_climate_alert(self) -> bool:
        """Check if this deviation warrants a climate alert."""
        return self.deviation_severity in ["significant", "extreme"]

    @property
    def severity_score(self) -> int:
        """Get numeric severity score (0-3) for comparison."""
        severity_map = {
            "normal": 0,
            "slight": 1,
            "significant": 2,
            "extreme": 3
        }
        return severity_map.get(self.deviation_severity, 0)

    @property
    def is_warmer_than_normal(self) -> bool:
        """Check if conditions are warmer than historical average."""
        return self.temperature_deviation_c > 1.0

    @property
    def is_cooler_than_normal(self) -> bool:
        """Check if conditions are cooler than historical average."""
        return self.temperature_deviation_c < -1.0

    def get_alert_icon(self) -> str:
        """Get appropriate icon for UI display."""
        if self.deviation_severity == "extreme":
            return "🚨"
        elif self.deviation_severity == "significant":
            return "⚠️"
        elif self.deviation_severity == "slight":
            return "⚡"
        else:
            return "✅"

    def get_temperature_trend_icon(self) -> str:
        """Get temperature trend icon for UI display."""
        if self.is_warmer_than_normal:
            return "🔥" if self.severity_score >= 2 else "🌡️"
        elif self.is_cooler_than_normal:
            return "🧊" if self.severity_score >= 2 else "❄️"
        else:
            return "🌡️"


# ---------------------------------------------------------------------------
# Enhanced utility classes
# ---------------------------------------------------------------------------

@dataclass
class ClimateAlert:
    """Represents a climate alert for unusual weather conditions."""

    city: str
    deviation: WeatherDeviation
    alert_level: str  # "watch", "warning", "extreme"
    message: str
    recommendations: List[str] = field(default_factory=list)

    @classmethod
    def from_deviation(cls, city: str, deviation: WeatherDeviation) -> Optional['ClimateAlert']:
        """Create a climate alert from a weather deviation if warranted."""
        if not deviation.is_climate_alert:
            return None

        if deviation.deviation_severity == "extreme":
            alert_level = "extreme"
            message = f"Extreme weather conditions expected in {city}"
        else:
            alert_level = "warning"
            message = f"Significant weather deviation expected in {city}"

        recommendations = []
        if deviation.is_warmer_than_normal:
            recommendations.extend([
                "Pack lighter clothing and sun protection",
                "Stay hydrated and seek shade during peak hours",
                "Consider indoor activities during hottest periods"
            ])
        elif deviation.is_cooler_than_normal:
            recommendations.extend([
                "Pack warmer clothing and layers",
                "Check heating availability at accommodations",
                "Plan indoor backup activities"
            ])

        return cls(
            city=city,
            deviation=deviation,
            alert_level=alert_level,
            message=message,
            recommendations=recommendations
        )


# ---------------------------------------------------------------------------
# Public re-exports and constants
# ---------------------------------------------------------------------------

# Climate analysis constants
DEVIATION_THRESHOLDS = {
    "temperature": {
        "slight": 3.0,
        "significant": 5.0,
        "extreme": 8.0
    },
    "humidity": {
        "slight": 10.0,
        "significant": 15.0,
        "extreme": 25.0
    }
}

CONFIDENCE_FACTORS = {
    "data_source": {
        "ai_enhanced_analysis": 0.8,
        "ai_historical_analysis": 0.7,
        "enhanced_fallback_estimate": 0.4,
        "fallback_estimate": 0.2
    },
    "deviation_magnitude": {
        "normal": 0.9,
        "slight": 0.8,
        "significant": 0.7,
        "extreme": 0.6
    }
}

__all__ = [
    # Core enums
    "AgentRole",
    "DeviationSeverity",
    "ClimateZone",

    # Data models
    "AgentMessage",
    "TaskResult",
    "TravelQuery",
    "ClimateBaseline",
    "WeatherDeviation",
    "ClimateAlert",

    # Configuration
    "OPENAI_KEY",
    "OPENWEATHER_KEY",
    "FLIGHTRADAR24_KEY",

    # Constants
    "DEVIATION_THRESHOLDS",
    "CONFIDENCE_FACTORS",
]