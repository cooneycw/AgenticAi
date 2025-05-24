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


# Install required packages:
# pip install shiny openai pandas requests

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
        self.client = OpenAI(api_key=api_key)
        self.conversation_history: List[AgentMessage] = []
        self.available_tools = {
            "web_search": self.mock_web_search,
            "data_analysis": self.mock_data_analysis,
            "calculation": self.perform_calculation
        }

    async def mock_web_search(self, query: str) -> TaskResult:
        """Simulated web search tool"""
        await asyncio.sleep(1)  # Simulate API delay
        mock_results = {
            "artificial intelligence": "AI is rapidly advancing with LLMs and autonomous agents...",
            "climate change": "Global temperatures rising, renewable energy adoption increasing...",
            "space exploration": "Recent Mars missions successful, commercial spaceflight expanding...",
            "default": f"Search results for '{query}': Multiple relevant articles found discussing various aspects..."
        }

        result = mock_results.get(query.lower(), mock_results["default"])
        return TaskResult(success=True, data=result)

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


# Initialize the agentic system (you'll need to set your OpenAI API key)
ai_system = None

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
            ui.input_action_button("example1", "AI Research", class_="btn-outline-secondary btn-sm mb-2"),
            ui.input_action_button("example2", "Market Analysis", class_="btn-outline-secondary btn-sm mb-2"),
            ui.input_action_button("example3", "Technical Planning", class_="btn-outline-secondary btn-sm mb-2"),

            width=300
        ),

        ui.main_panel(
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
    workflow_status = reactive.Value("")
    conversation_messages = reactive.Value([])

    @reactive.Effect
    @reactive.event(input.connect)
    def connect_ai_system():
        global ai_system
        if input.api_key():
            try:
                ai_system = AgenticAISystem(input.api_key())
                workflow_status.set("✅ Connected to OpenAI API")
            except Exception as e:
                workflow_status.set(f"❌ Connection failed: {str(e)}")
        else:
            workflow_status.set("⚠️ Please enter your OpenAI API key")

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