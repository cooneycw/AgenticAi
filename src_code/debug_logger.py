# src_code/debug_logger.py
"""Enhanced Debug Logger for Agentic Activity Focus

This module captures debug output and makes it available to the Shiny UI
with enhanced focus on AI agent decision-making processes.

ENHANCED FEATURES:
- Agent decision tracking (AGENT-*, CLIMATE-*, REASONING)
- Temperature/weather decision logging
- Critical agentic activity highlighting
- Better filtering for relevant information
- Agent workflow progress tracking
"""
from __future__ import annotations

import sys
from datetime import datetime
from typing import List, Optional, Callable
from io import StringIO

__all__ = ["DebugLogger", "setup_debug_capture", "debug_info", "debug_success", "debug_warning", "debug_error"]


class DebugLogger:
    """Captures debug output for display in Shiny UI with agent activity focus."""

    def __init__(self, max_entries: int = 150):  # Increased for more agent activity
        self.max_entries = max_entries
        self.entries: List[str] = []
        self.callback: Optional[Callable[[str], None]] = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def set_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback function to notify UI of new debug entries."""
        self.callback = callback

    def add_entry(self, message: str, level: str = "DEBUG") -> None:
        """Add a debug entry with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        entry = f"[{timestamp}] {level}: {message}"

        # Add to entries list
        self.entries.append(entry)

        # Maintain max entries limit
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        # Print to original stdout (for console debugging)
        print(entry, file=self.original_stdout)

        # Notify UI callback if set
        if self.callback:
            try:
                self.callback(entry)
            except Exception as e:
                print(f"Debug callback error: {e}", file=self.original_stdout)

    def get_entries(self) -> List[str]:
        """Get all debug entries."""
        return list(self.entries)

    def get_recent_entries(self, count: int = 30) -> List[str]:  # Increased default
        """Get most recent debug entries."""
        return self.entries[-count:] if self.entries else []

    def clear(self) -> None:
        """Clear all debug entries."""
        self.entries.clear()
        if self.callback:
            self.callback("DEBUG LOG CLEARED")


# Global debug logger instance
debug_logger = DebugLogger()


class EnhancedDebugCapture:
    """Enhanced capture that focuses on agent decision-making processes."""

    def __init__(self, original_stream, logger: DebugLogger, level: str = "DEBUG"):
        self.original_stream = original_stream
        self.logger = logger
        self.level = level

    def write(self, message: str) -> None:
        """Capture write calls with enhanced filtering for agent activity."""
        # Forward to original stream
        self.original_stream.write(message)

        # Process message for debug logger
        message = message.strip()
        if message and not message.isspace():
            # Enhanced filtering for agentic activity
            if self._is_agent_relevant(message):
                self.logger.add_entry(message, self._determine_level(message))

    def _is_agent_relevant(self, message: str) -> bool:
        """Determine if message is relevant to agent decision-making."""
        agent_keywords = [
            # Agent decision tracking
            "AGENT-", "CLIMATE-", "REASONING:", "TEMP-FORECAST:", "CLIMATE-DECISION:",
            "WORKFLOW:", "PARSER:", "RESEARCHER:", "ANALYZER:", "SYNTHESIZER:",

            # Weather and climate decisions
            "OpenWeatherMap One Call API", "Climate AI", "temperature", "°C",
            "MEETS", "OUTSIDE", "criteria", "deviation", "baseline",

            # API status and critical info
            "DEBUG:", "Success:", "Failed:", "Error:", "SUCCESS:", "INFO:",
            "Trying", "success", "failed", "Generated", "Starting", "complete",

            # City and flight decisions
            "cities", "flight", "coordinates", "analysis", "scoring",

            # Progress and workflow
            "Agent", "workflow", "Processing", "Analyzing", "Generating"
        ]

        return any(keyword in message for keyword in agent_keywords)

    def _determine_level(self, message: str) -> str:
        """Determine appropriate log level based on message content."""
        message_upper = message.upper()

        if any(word in message_upper for word in ["AGENT-", "CLIMATE-", "WORKFLOW:"]):
            return "AGENT"
        elif any(word in message_upper for word in ["TEMP-FORECAST:", "CLIMATE-DECISION:"]):
            return "CLIMATE"
        elif any(word in message_upper for word in ["REASONING:"]):
            return "REASONING"
        elif any(word in message_upper for word in ["SUCCESS:", "COMPLETE", "GENERATED"]):
            return "SUCCESS"
        elif any(word in message_upper for word in ["ERROR:", "FAILED", "CRITICAL"]):
            return "ERROR"
        elif any(word in message_upper for word in ["WARNING:", "WARN:"]):
            return "WARN"
        else:
            return "INFO"

    def flush(self) -> None:
        """Flush the original stream."""
        self.original_stream.flush()


def setup_debug_capture() -> DebugLogger:
    """Set up enhanced debug capture focused on agent activity."""
    # Capture stdout (where our print statements go)
    sys.stdout = EnhancedDebugCapture(debug_logger.original_stdout, debug_logger, "INFO")

    return debug_logger


def debug_print(message: str, level: str = "DEBUG") -> None:
    """Enhanced debug print that goes to both console and UI."""
    debug_logger.add_entry(message, level)


# Convenience functions for different log levels
def debug_info(message: str) -> None:
    """Log info level message."""
    debug_print(message, "INFO")


def debug_warning(message: str) -> None:
    """Log warning level message."""
    debug_print(message, "WARN")


def debug_error(message: str) -> None:
    """Log error level message."""
    debug_print(message, "ERROR")


def debug_success(message: str) -> None:
    """Log success level message."""
    debug_print(message, "SUCCESS")


def debug_agent(message: str) -> None:
    """Log agent decision message."""
    debug_print(message, "AGENT")


def debug_climate(message: str) -> None:
    """Log climate analysis message."""
    debug_print(message, "CLIMATE")


# Test function
if __name__ == "__main__":
    setup_debug_capture()

    print("Testing enhanced debug capture for agent activity...")
    debug_info("This is an info message")
    debug_warning("This is a warning")
    debug_error("This is an error")
    debug_success("This is a success message")
    debug_agent("AGENT-RESEARCHER: Starting city generation")
    debug_climate("TEMP-FORECAST: 23.5°C | HISTORICAL: 20.1°C")

    print("AGENT-PLANNER: Planning strategy")
    print("CLIMATE-DECISION: MEETS criteria (23.5°C vs 20-25°C target)")
    print("REASONING: Temperature within target range with good climate score")
    print("OpenWeatherMap One Call API 3.0 success for Barcelona")
    print("Regular message that should be filtered out")
    print("Generated 8 diverse cities across 2 regions")

    print("\nDebug entries captured:")
    for entry in debug_logger.get_entries():
        print(f"  {entry}")