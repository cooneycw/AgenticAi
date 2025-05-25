# src_code/debug_logger.py
"""Debug Logger for Shiny Integration

This module captures debug output and makes it available to the Shiny UI
in a collapsible, real-time debug section.
"""
from __future__ import annotations

import sys
from datetime import datetime
from typing import List, Optional, Callable
from io import StringIO

__all__ = ["DebugLogger", "setup_debug_capture"]


class DebugLogger:
    """Captures debug output for display in Shiny UI."""

    def __init__(self, max_entries: int = 100):
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

    def get_recent_entries(self, count: int = 20) -> List[str]:
        """Get most recent debug entries."""
        return self.entries[-count:] if self.entries else []

    def clear(self) -> None:
        """Clear all debug entries."""
        self.entries.clear()
        if self.callback:
            self.callback("DEBUG LOG CLEARED")


# Global debug logger instance
debug_logger = DebugLogger()


class DebugCapture:
    """Captures print statements and forwards them to debug logger."""

    def __init__(self, original_stream, logger: DebugLogger, level: str = "DEBUG"):
        self.original_stream = original_stream
        self.logger = logger
        self.level = level

    def write(self, message: str) -> None:
        """Capture write calls (print statements)."""
        # Forward to original stream
        self.original_stream.write(message)

        # Process message for debug logger
        message = message.strip()
        if message and not message.isspace():
            # Filter for relevant debug messages
            if any(keyword in message for keyword in [
                "DEBUG:", "OpenWeatherMap", "API", "Trying", "Success", "Failed",
                "Starting", "Completed", "Generated", "Analyzing", "Processing"
            ]):
                self.logger.add_entry(message, self.level)

    def flush(self) -> None:
        """Flush the original stream."""
        self.original_stream.flush()


def setup_debug_capture() -> DebugLogger:
    """Set up debug capture for stdout."""
    # Capture stdout (where our print statements go)
    sys.stdout = DebugCapture(debug_logger.original_stdout, debug_logger, "INFO")

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


# Test function
if __name__ == "__main__":
    setup_debug_capture()

    print("Testing debug capture...")
    debug_info("This is an info message")
    debug_warning("This is a warning")
    debug_error("This is an error")
    debug_success("This is a success message")

    print("DEBUG: This should be captured")
    print("OpenWeatherMap: This should also be captured")
    print("Regular message that might not be captured")

    print("\nDebug entries:")
    for entry in debug_logger.get_entries():
        print(f"  {entry}")