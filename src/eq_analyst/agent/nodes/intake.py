"""First node of every turn: understand what the user wants, then route."""

from __future__ import annotations

from typing import Any, Literal, cast

from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

from eq_analyst.agent.context import AgentContext
from eq_analyst.agent.nodes.common import recent_messages
from eq_analyst.agent.schemas import Intent
from eq_analyst.agent.state import AnalystState
from eq_analyst.core.log import get_logger
from eq_analyst.domain.models import Horizon

log = get_logger(__name__)

HISTORY_WINDOW = 6

# Per-turn keys that must not leak from the previous turn into this one.
FRESH_TURN: dict[str, Any] = {
    "ticker": None,
    "technical": None,
    "fundamentals": None,
    "sentiment": None,
    "risk": None,
    "draft": None,
}


async def intake(state: AnalystState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    report = state.get("report")
    current = f"{report.ticker.name} ({report.ticker.symbol})" if report else "none"

    system = ctx.prompts.render("intent-router", current_company=current)
    classifier = ctx.llm.with_structured_output(Intent)
    intent = cast(
        Intent,
        await classifier.ainvoke(
            [SystemMessage(system), *recent_messages(state["messages"], HISTORY_WINDOW)]
        ),
    )

    # Horizon: what the user said > the horizon of the analysis being discussed > medium.
    if intent.horizon is not None:
        horizon = intent.horizon
    elif intent.kind == "follow_up" and report is not None:
        horizon = report.horizon
    else:
        horizon = Horizon.MEDIUM

    log.info("intent_classified", kind=intent.kind, company=intent.company, horizon=horizon)
    return {"intent": intent, "horizon": horizon, **FRESH_TURN}


def route_intent(state: AnalystState) -> Literal["resolve_ticker", "follow_up", "reply_help"]:
    """Conditional edge: plain Python decides where the graph goes next."""
    intent = state["intent"]
    if intent.kind == "analyze" and intent.company:
        return "resolve_ticker"
    if intent.kind == "follow_up" and state.get("report") is not None:
        return "follow_up"
    return "reply_help"
