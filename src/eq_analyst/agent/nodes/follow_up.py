"""Follow-up questions are handled by a tool-using ReAct agent.

The analysis pipeline is a fixed *workflow*: predictable and auditable. Follow-ups are
open-ended ("compare it with Infosys", "what's its P/E?"), so here an *agent* decides
which MCP tools to call, in a loop, until it can answer. Workflow where we need
control, agent where we need flexibility.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.runtime import Runtime

from eq_analyst.agent.context import AgentContext
from eq_analyst.agent.nodes.common import recent_messages
from eq_analyst.agent.state import AnalystState

HISTORY_WINDOW = 10
# Each tool call is ~2 steps; this caps the agent at a handful of calls (cost/latency guard).
MAX_AGENT_STEPS = 12


async def follow_up(state: AnalystState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    report = state["report"]
    if report is None:
        raise RuntimeError("follow_up ran without a previous report")

    system = ctx.prompts.render("follow-up-agent", report_json=report.model_dump_json(indent=2))
    history = recent_messages(state["messages"], HISTORY_WINDOW)

    result = await ctx.follow_up_agent.ainvoke(
        {"messages": [SystemMessage(system), *history]},
        config={"recursion_limit": MAX_AGENT_STEPS},
    )
    answer = result["messages"][-1].text or "Sorry, I couldn't work that out. Try rephrasing?"

    # Only the final answer is stored in memory, not the intermediate tool calls.
    return {"messages": [AIMessage(answer)], "response_kind": "message"}
