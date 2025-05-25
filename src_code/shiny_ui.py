# src_code/shiny_ui.py
"""Step 4 – lightweight Shiny‑Py front‑end (fixed variable name clash)

This UI module provides the presentation layer and delegates all heavy
lifting to :pyclass:`src_code.agentic_core.AgenticAISystem`.

*Bug fix*: previously the `conversation` reactive state variable clashed
with the `@output.conversation` render function, causing
`AttributeError: 'ui' object has no attribute 'get'`. The state container
is now called **`conv_state`** and the output is **`messages_view`**.
"""
from __future__ import annotations

import asyncio

from shiny import App, reactive, render, ui  # type: ignore – shiny runtime

from config.config import (
    FLIGHTRADAR24_KEY,
    OPENAI_KEY,
    OPENWEATHER_KEY,
)
from src_code.agentic_core import AgenticAISystem

# ---------------------------------------------------------------------------
# Instantiate (may be None until user connects)
# ---------------------------------------------------------------------------

ai_system: AgenticAISystem | None = (
    AgenticAISystem(OPENAI_KEY, OPENWEATHER_KEY, FLIGHTRADAR24_KEY)
    if OPENAI_KEY
    else None
)

# ---------------------------------------------------------------------------
# UI layout – minimal, clean
# ---------------------------------------------------------------------------

app_ui = ui.page_fillable(
    ui.h1("🌍 Agentic Travel Planner", class_="text-center mb-4"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_password("api_key", "OpenAI API Key", placeholder="sk-..."),
            ui.input_action_button("connect", "Connect", class_="btn-primary mb-3"),
            ui.hr(),
            ui.input_text_area(
                "user_query",
                "Travel query",
                placeholder="e.g. Warm European cities next Friday 18-24 °C from Toronto",
                rows=3,
            ),
            ui.input_action_button("run", "🚀 Run agents", class_="btn-success"),
            width=300,
        ),
        ui.div(
            ui.output_ui("status"),
            ui.hr(),
            ui.output_ui("messages_view"),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Server logic – thin wrapper around AgenticAISystem
# ---------------------------------------------------------------------------

def server(input, output, session):  # noqa: ANN001 – shiny signature
    workflow_status = reactive.Value("🔗 Not connected")
    conv_state = reactive.Value([])  # type: ignore[var-annotated]

    # ---------------------------------------------------- connection button
    @reactive.Effect
    @reactive.event(input.connect)
    def _connect():
        global ai_system  # pylint: disable=global-statement
        key = input.api_key() or OPENAI_KEY
        if not key:
            workflow_status.set("❌ Enter an OpenAI key first")
            return
        try:
            ai_system = AgenticAISystem(key, OPENWEATHER_KEY, FLIGHTRADAR24_KEY)
            workflow_status.set("✅ Connected – ready")
        except Exception as exc:  # pragma: no cover – displayable error
            workflow_status.set(f"❌ Connection failed: {exc}")

    # ---------------------------------------------------- run workflow
    @reactive.Effect
    @reactive.event(input.run)
    async def _run():
        if ai_system is None:
            workflow_status.set("❌ Connect first")
            return
        query = input.user_query().strip()
        if not query:
            workflow_status.set("⚠️ Enter a query")
            return
        workflow_status.set("⏳ Working …")
        msgs = await ai_system.run_agentic_workflow(query)
        conv_state.set(msgs)
        workflow_status.set("✅ Done")

    # ---------------------------------------------------- outputs
    @output
    @render.ui
    def status():  # noqa: ANN001 – shiny signature
        txt = workflow_status.get()
        cls = (
            "alert alert-success"
            if txt.startswith("✅")
            else "alert alert-danger"
            if txt.startswith("❌")
            else "alert alert-info"
        )
        return ui.div(ui.p(txt, class_="m-0"), class_=cls)

    @output
    @render.ui
    def messages_view():  # noqa: ANN001 – shiny signature
        msgs = conv_state.get()
        if not msgs:
            return ui.p("No messages yet.")
        blocks = []
        for m in msgs:
            blocks.append(
                ui.div(
                    ui.strong(m.agent_role.value.title()),
                    ui.p(m.content, style="white-space: pre-wrap;"),
                    class_="border rounded p-2 mb-3",
                )
            )
        return ui.div(*blocks)


# ---------------------------------------------------------------------------
# Shiny app instance
# ---------------------------------------------------------------------------

app = App(app_ui, server)

# ---------------------------------------------------------------------------
# Manual run (`python -m src_code.shiny_ui`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
