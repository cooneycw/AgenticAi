# src_code/agentic_core.py
"""Step 3 – Agentic core (business logic, no UI).

This module orchestrates the *end‑to‑end* travel‑planning workflow:

1. **Parse** a free‑form user query → `TravelQuery` (LLM‑aided)
2. **Plan – Research – Analyse – Synthesise** with four autonomous agents
3. Delegate climate reasoning to :pymod:`src_code.climate_analyzer`
4. Optional live‑data integrations (Open‑Meteo & FlightRadar24)

It depends only on
* ``config.config``   → dataclasses & API keys
* ``src_code.climate_analyzer.EnhancedClimateAnalyzer``
* ``openai``           (≥ 1.2) for chatting with GPT‑4/4o‑mini
* ``aiohttp``          for HTTP I/O

Nothing here is Shiny‑specific – the UI can import `AgenticAISystem` and
call :py:meth:`run_agentic_workflow` to get a list of ``AgentMessage``
objects that the front‑end renders however it likes.
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from dateutil.relativedelta import relativedelta
from openai import OpenAI

# Local shared state
from config.config import (
    AgentMessage,
    AgentRole,
    ClimateBaseline,
    TaskResult,
    TravelQuery,
    WeatherDeviation,
    OPENAI_KEY,
    OPENWEATHER_KEY,
    FLIGHTRADAR24_KEY,
)
from src_code.climate_analyzer import EnhancedClimateAnalyzer

__all__ = ["AgenticAISystem"]


# ---------------------------------------------------------------------------
# Utility helpers (local)
# ---------------------------------------------------------------------------

def _today() -> datetime:
    """Timezone‑agnostic *now* for deterministic unit tests."""
    return datetime.now()


def _nl_date_to_dt(expr: str) -> datetime:
    """Very small natural‑language → datetime helper (no external lib)."""
    expr = expr.lower().strip()
    now = _today()
    if expr in {"today"}:
        return now
    if expr in {"tomorrow"}:
        return now + timedelta(days=1)
    if "next week" in expr:
        return now + timedelta(weeks=1)
    if "next month" in expr:
        return now + relativedelta(months=1)
    # weekday names
    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for w, idx in weekdays.items():
        if w in expr:
            delta = (idx - now.weekday()) % 7
            delta = 7 if delta == 0 else delta  # always *next*
            return now + timedelta(days=delta)
    # "in X days/weeks/months"
    m = re.search(r"(\d+)\s*(day|week|month)s?", expr)
    if m:
        num, unit = int(m.group(1)), m.group(2)
        if unit == "day":
            return now + timedelta(days=num)
        if unit == "week":
            return now + timedelta(weeks=num)
        if unit == "month":
            return now + relativedelta(months=num)
    # default fallback
    return now + timedelta(days=1)


# ---------------------------------------------------------------------------
# Core system class
# ---------------------------------------------------------------------------

class AgenticAISystem:
    """Orchestrates the four‑agent pipeline for a single user query."""

    def __init__(
        self,
        openai_key: Optional[str] = None,
        openweather_key: Optional[str] = None,
        flightradar24_key: Optional[str] = None,
    ) -> None:
        if openai_key is None:
            raise RuntimeError("OpenAI key required for AgenticAISystem")
        self.client = OpenAI(api_key=openai_key)
        self.openweather_key = openweather_key
        self.fr24_key = flightradar24_key

        self.conversation_history: List[AgentMessage] = []
        self.climate = EnhancedClimateAnalyzer(self.client)

    # ------------------------------------------------------------------
    # Small helpers shared by agents
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_temp_range(q: TravelQuery) -> Tuple[float, float]:
        rng = q.temperature_range or (20.0, 27.0)
        if len(rng) < 2:
            return (20.0, 27.0)
        lo, hi = sorted(map(float, rng[:2]))
        return (lo, hi)

    # ------------------------------------------------------------------
    # 1)   Query parsing (LLM) – returns TravelQuery
    # ------------------------------------------------------------------

    async def parse_travel_query(self, query: str) -> TravelQuery:
        """Ask GPT‑4o‑mini to turn free text into a `TravelQuery`."""
        system_prompt = (
            "You are a travel query parser. Return ONLY valid JSON with "
            "fields: regions, forecast_date, temperature_range (2 floats), "
            "origin_city, additional_criteria."
        )
        try:
            rsp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
            )
            data = json.loads(rsp.choices[0].message.content)
        except Exception:
            # minimal fallback
            data = {
                "regions": ["Europe"],
                "forecast_date": "tomorrow",
                "temperature_range": [20, 27],
                "origin_city": "Toronto",
                "additional_criteria": [],
            }

        # robust type handling
        temp_rng = data.get("temperature_range", [20, 27])
        if isinstance(temp_rng, (int, float)):
            temp_rng = [temp_rng - 3, temp_rng + 3]
        if isinstance(temp_rng, list) and len(temp_rng) == 1:
            temp_rng = [temp_rng[0] - 3, temp_rng[0] + 3]
        if len(temp_rng) < 2:
            temp_rng = [20, 27]

        parsed = TravelQuery(
            regions=data.get("regions", ["Europe"]),
            forecast_date=data.get("forecast_date", "tomorrow"),
            forecast_date_parsed=_nl_date_to_dt(data.get("forecast_date", "tomorrow")),
            temperature_range=tuple(map(float, temp_rng[:2])),
            origin_city=data.get("origin_city", "Toronto"),
            additional_criteria=data.get("additional_criteria", []),
            raw_query=query,
        )
        return parsed

    # ------------------------------------------------------------------
    # 2)   Research helpers – cities, coordinates, weather
    # ------------------------------------------------------------------

    async def _cities_for_regions(
        self, regions: List[str], n: int = 8
    ) -> List[Dict[str, str]]:
        """Return *n* diverse cities for each region (LLM)."""
        cities: List[Dict[str, str]] = []
        for region in regions:
            temperature = random.uniform(0.7, 0.9)
            prompt = (
                "Return EXACTLY {n} JSON items {{city, country}} – diverse, mix "
                "of famous & hidden gems in {region}."
            ).format(n=n, region=region)
            try:
                rsp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=temperature,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                batch = json.loads(rsp.choices[0].message.content)
                for item in batch:
                    item["region"] = region
                cities.extend(batch)
            except Exception:
                # crude fallback list
                if region.lower().startswith("europe"):
                    cities.extend(
                        [{"city": "Barcelona", "country": "Spain", "region": region},
                         {"city": "Krakow", "country": "Poland", "region": region}]
                    )
        random.shuffle(cities)
        return cities

    async def _coords(self, city: str, country: str) -> Optional[Dict[str, float]]:
        url = (
            f"https://nominatim.openstreetmap.org/search?q={city}%2C+{country}&format=json&limit=1"
        )
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, headers={"User-Agent": "AgenticAI/1.0"}) as r:
                if r.status == 200:
                    data = await r.json()
                    if data:
                        return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
        return None

    async def _openmeteo_day(
        self, lat: float, lon: float, day: datetime
    ) -> TaskResult:
        days_ahead = (day - _today()).days
        if days_ahead > 16:
            return TaskResult(False, None, "beyond 16‑day window")
        ds = day.strftime("%Y-%m-%d")
        url = (
            "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"  # noqa: E501
            "&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m,"  # noqa: E501
            "weather_code&start_date={d}&end_date={d}&timezone=auto"
        ).format(lat=lat, lon=lon, d=ds)
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url) as r:
                if r.status != 200:
                    return TaskResult(False, None, f"HTTP {r.status}")
                data = await r.json()
        daily = data.get("daily", {})
        if not daily:
            return TaskResult(False, None, "no daily block")
        hi = daily["temperature_2m_max"][0]
        lo = daily["temperature_2m_min"][0]
        hum = daily["relative_humidity_2m"][0]
        avg = (hi + lo) / 2
        return TaskResult(True, {
            "temp_max": hi,
            "temp_min": lo,
            "temp_avg": avg,
            "humidity": hum,
        })

    # ------------------------------------------------------------------
    # 3)   Flight helpers (minimal – geo estimate only by default)
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371  # km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
            math.radians(lat2)
        ) * math.sin(dlon / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    def _flight_estimate(self, origin: Dict[str, float], dest: Dict[str, float]) -> Tuple[float, str]:
        km = self._haversine(origin["lat"], origin["lon"], dest["lat"], dest["lon"])
        if km < 1500:
            hrs = km / 700 + 0.5
            via = "Direct likely"
        elif km < 4000:
            hrs = km / 800 + 0.75
            via = "Direct or 1 stop"
        else:
            hrs = km / 850 + 1.5
            via = "1–2 stops typical"
        return hrs, via

    # ------------------------------------------------------------------
    # 4)   The four agents – implemented as a single *switch* for brevity
    # ------------------------------------------------------------------

    async def _agent(self, role: AgentRole, q: TravelQuery) -> AgentMessage:
        """Dispatch logic based on *role* – returns AgentMessage."""
        tools: List[str] = []
        meta: Dict[str, Any] = {}
        content = ""

        if role == AgentRole.PLANNER:
            lo, hi = self._safe_temp_range(q)
            content = (
                "Plan: collect {{cities→coords}}, then weather (Open‑Meteo or "
                "seasonal), then climate deviations, finally flight estimates. "
                f"Target °C {lo}-{hi}."
            )
            tools = ["plan"]

        elif role == AgentRole.RESEARCHER:
            # 1. candidate cities
            cities = await self._cities_for_regions(q.regions, 8)
            weather_results: List[Dict[str, Any]] = []

            for item in cities:
                coords = await self._coords(item["city"], item["country"])
                if not coords:
                    continue
                wx = await self._openmeteo_day(
                    coords["lat"], coords["lon"], q.forecast_date_parsed
                )
                if not wx.success:
                    continue

                baseline = await self.climate.get_historical_climate_baseline(
                    item["city"], item["country"], q.forecast_date_parsed.month
                )
                deviation = await self.climate.analyse_deviation(wx.data, baseline)
                wx_blob = {
                    **item,
                    **wx.data,
                    "historical_baseline": baseline,
                    "climate_deviation": deviation,
                }
                weather_results.append(wx_blob)

            meta = {"weather_results": weather_results}
            content = f"Gathered weather for {len(weather_results)}/{len(cities)} cities."
            tools = ["openmeteo", "climate_analyzer"]

        elif role == AgentRole.ANALYZER:
            # needs previous researcher meta
            r_messages = [m for m in self.conversation_history if m.agent_role == AgentRole.RESEARCHER]
            if not r_messages or not r_messages[-1].metadata:
                content = "No research data to analyse"
            else:
                wx = r_messages[-1].metadata["weather_results"]
                lo, hi = self._safe_temp_range(q)
                scored: List[Dict[str, Any]] = []
                for w in wx:
                    meets = lo <= w["temp_avg"] <= hi
                    dev: WeatherDeviation = w["climate_deviation"]
                    temp_score = max(0, 10 - abs(w["temp_avg"] - ((lo + hi) / 2)))
                    climate_penalty = {"normal": 0, "slight": 0.5, "significant": 2, "extreme": 3}[dev.deviation_severity]
                    score = temp_score - climate_penalty
                    w["overall_score"] = score
                    w["meets"] = meets
                    scored.append(w)
                scored.sort(key=lambda x: x["overall_score"], reverse=True)
                meta = {"scored": scored}
                top = scored[0] if scored else {}
                content = (
                    f"Top candidate: {top.get('city', 'N/A')} – score {top.get('overall_score', 0):.1f}"
                )
            tools = ["analysis"]

        elif role == AgentRole.SYNTHESIZER:
            a_messages = [m for m in self.conversation_history if m.agent_role == AgentRole.ANALYZER]
            if not a_messages or not a_messages[-1].metadata:
                content = "No analysis to summarise"
            else:
                scored = a_messages[-1].metadata["scored"]
                best = scored[:3]
                bullets = []
                for d in best:
                    bullets.append(
                        f"• {d['city']} {d['temp_avg']:.1f}°C ({d['climate_deviation'].deviation_severity})"
                    )
                content = "\n".join(["Recommended:\n", *bullets])
            tools = ["synthesis"]

        msg = AgentMessage(
            agent_role=role,
            content=content,
            timestamp=_today(),
            tools_used=tools,
            metadata=meta,
        )
        self.conversation_history.append(msg)
        return msg

    # ------------------------------------------------------------------
    # Public workflow driver
    # ------------------------------------------------------------------

    async def run_agentic_workflow(self, user_query: str) -> List[AgentMessage]:
        """High‑level convenience wrapper for consumers (e.g., Shiny UI)."""
        self.conversation_history.clear()
        q = await self.parse_travel_query(user_query)

        for role in (
            AgentRole.PLANNER,
            AgentRole.RESEARCHER,
            AgentRole.ANALYZER,
            AgentRole.SYNTHESIZER,
        ):
            await self._agent(role, q)

        return list(self.conversation_history)


# ---------------------------------------------------------------------------
# CLI test – allows `python -m src_code.agentic_core "query"`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query_txt = " ".join(sys.argv[1:]) or "Find warm European cities next Friday between 18‑24°C from Toronto"
    core = AgenticAISystem(OPENAI_KEY or "DUMMY_KEY")  # raises if missing

    async def _main():
        msgs = await core.run_agentic_workflow(query_txt)
        for m in msgs:
            banner = f"[{m.agent_role.value.upper()}]"
            print(banner, m.content, "\n")

    asyncio.run(_main())
