# src_code/climate_analyzer.py
"""Step 2 – Climate & weather intelligence layer

This module isolates all logic related to:
1.   Pulling / synthesising *historical* climate baselines for a given city &
     month using an OpenAI client (or another provider in the future).
2.   Comparing a *forecast* (or seasonal estimate) against the baseline to
     quantify deviations and produce human‑readable context for travellers.

It can be imported by any service that needs climate awareness – in our case
`src_code/agentic_core.py` will be the main consumer.

External deps: ``openai`` (>=1.2), ``python-dateutil`` and standard lib.
Internals: pulls dataclass definitions from ``config.config``.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

from dateutil.relativedelta import relativedelta

# 3rd‑party
from openai import OpenAI  # type: ignore – runtime import guarded by caller

# Local shared models & constants
from config.config import (
    ClimateBaseline,
    WeatherDeviation,
)


# ---------------------------------------------------------------------------
# Helper utilities (private)
# ---------------------------------------------------------------------------

def _month_name(month_idx: int) -> str:
    """Return full English month name for *1‑based* month index."""
    return datetime(2024, month_idx, 1).strftime("%B")


# ---------------------------------------------------------------------------
# Public class – injected with an OpenAI client
# ---------------------------------------------------------------------------

class EnhancedClimateAnalyzer:
    """Compute historical baselines & deviations using OpenAI knowledge.

    Parameters
    ----------
    openai_client
        A pre‑configured ``openai.OpenAI`` client instance. Dependency
        injection keeps this module decoupled from credential management.
    """

    #: cache key -> ``ClimateBaseline``
    _climate_cache: Dict[str, ClimateBaseline]

    def __init__(self, openai_client: OpenAI):
        self._client = openai_client
        self._climate_cache = {}

    # ---------------------------------------------------------------------
    # Baseline Generation
    # ---------------------------------------------------------------------

    async def get_historical_climate_baseline(
        self,
        city: str,
        country: str,
        target_month: int,
    ) -> ClimateBaseline:
        """Return a 30‑year climate normal for *city* in *month*.

        Falls back to a rule‑of‑thumb seasonal estimate if the LLM call fails
        or returns malformed JSON.
        """
        cache_key = f"{city.lower()}_{country.lower()}_{target_month}"
        if cache_key in self._climate_cache:
            return self._climate_cache[cache_key]

        month_name = _month_name(target_month)

        system_prompt = (
            "You are a climate science assistant with access to 30‑year "
            "normals. Return ONLY valid JSON with the exact keys given."
        )

        user_prompt = (
            f"Provide historical climate baseline for {city}, {country} in "
            f"{month_name}."
        )

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            payload = json.loads(response.choices[0].message.content)
        except Exception as exc:  # network or JSON failure
            print(f"[Climate] fallback – {exc}")
            return self._fallback_baseline(city, country, target_month)

        baseline = ClimateBaseline(
            city=f"{city}, {country}",
            month=target_month,
            avg_temp_high=payload.get("avg_temp_high", 20.0),
            avg_temp_low=payload.get("avg_temp_low", 10.0),
            avg_temp_mean=payload.get("avg_temp_mean", 15.0),
            avg_humidity=payload.get("avg_humidity", 60.0),
            avg_precipitation_days=payload.get("avg_precipitation_days", 5),
            typical_weather_patterns=payload.get(
                "typical_weather_patterns", ["variable"]
            ),
            climate_zone=payload.get("climate_zone", "temperate"),
        )

        self._climate_cache[cache_key] = baseline
        return baseline

    # ------------------------------------------------------------------
    # Deviation analysis
    # ------------------------------------------------------------------

    async def analyse_deviation(
        self,
        forecast: Dict[str, Any],
        baseline: ClimateBaseline,
    ) -> WeatherDeviation:
        """Compare forecast vs. baseline to score deviation severity."""

        f_temp = forecast.get("temp_avg", forecast.get("temp_max", 0))
        f_hum = forecast.get("humidity", 0)

        temp_dev = f_temp - baseline.avg_temp_mean
        hum_dev = f_hum - baseline.avg_humidity

        severity = self._severity_level(temp_dev, hum_dev)
        context_blob = await self._generate_deviation_context(
            forecast, baseline, temp_dev, hum_dev, severity
        )

        return WeatherDeviation(
            temperature_deviation_c=temp_dev,
            humidity_deviation_pct=hum_dev,
            deviation_severity=severity,
            deviation_context=context_blob["context"],
            traveler_impact=context_blob["impact"],
            confidence_level=context_blob["confidence"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fallback_baseline(
        self, city: str, country: str, month: int
    ) -> ClimateBaseline:
        """Simple hemispheric seasonal guess when the LLM fails."""
        if month in (12, 1, 2):
            hi, lo = 10.0, 2.0
        elif month in (3, 4, 5):
            hi, lo = 18.0, 8.0
        elif month in (6, 7, 8):
            hi, lo = 26.0, 16.0
        else:
            hi, lo = 16.0, 6.0

        return ClimateBaseline(
            city=f"{city}, {country}",
            month=month,
            avg_temp_high=hi,
            avg_temp_low=lo,
            avg_temp_mean=(hi + lo) / 2,
            avg_humidity=65.0,
            avg_precipitation_days=6,
            typical_weather_patterns=["seasonal average"],
            climate_zone="unknown",
            data_source="fallback",
        )

    # ..................................................................

    @staticmethod
    def _severity_level(temp_dev: float, hum_dev: float) -> str:
        abs_t, abs_h = abs(temp_dev), abs(hum_dev)
        tiers = ["normal", "slight", "significant", "extreme"]

        def tier(x: float, t1: float, t2: float, t3: float) -> int:
            if x >= t3:
                return 3
            if x >= t2:
                return 2
            if x >= t1:
                return 1
            return 0

        t_level = tier(abs_t, 3, 5, 8)
        h_level = tier(abs_h, 10, 15, 25)
        return tiers[max(t_level, h_level)]

    # ..................................................................

    async def _generate_deviation_context(
        self,
        forecast: Dict[str, Any],
        baseline: ClimateBaseline,
        temp_dev: float,
        hum_dev: float,
        severity: str,
    ) -> Dict[str, str]:
        """Ask the LLM to turn numeric deltas into human advice."""
        month_name = _month_name(baseline.month)
        sys_prompt = (
            "You are a travel weather analyst. Respond with JSON {context, impact, confidence}."
        )

        summary = (
            f"City: {baseline.city} – {month_name}\n"
            f"Historical mean: {baseline.avg_temp_mean:.1f}°C / {baseline.avg_humidity:.0f}%\n"
            f"Forecast: {forecast.get('temp_avg', 0):.1f}°C / {forecast.get('humidity', 0):.0f}%\n"
            f"Temp Δ: {temp_dev:+.1f}°C | Hum Δ: {hum_dev:+.0f}% (severity: {severity})"
        )

        try:
            rsp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": summary},
                ],
            )
            return json.loads(rsp.choices[0].message.content)
        except Exception:
            # graceful degradation
            return {
                "context": f"Conditions are {severity} vs. normal.",
                "impact": "Pack flexible clothing & check forecasts daily.",
                "confidence": "low",
            }
