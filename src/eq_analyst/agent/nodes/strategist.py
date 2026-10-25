"""Synthesis: the LLM writes the view, then deterministic guardrails check it."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from eq_analyst.agent.context import AgentContext
from eq_analyst.agent.nodes.common import require_ticker
from eq_analyst.agent.state import AnalystState
from eq_analyst.core.log import get_logger
from eq_analyst.domain.guardrails import apply_guardrails
from eq_analyst.domain.models import AnalysisReport, Horizon, StrategyDraft
from eq_analyst.domain.signals import describe_technicals

log = get_logger(__name__)

HORIZON_GUIDANCE = {
    Horizon.SHORT: "SHORT (days to 2 weeks): weight price action, momentum and news most.",
    Horizon.MEDIUM: "MEDIUM (2 weeks to 3 months): balance technicals, fundamentals and news.",
    Horizon.LONG: "LONG (3+ months): weight fundamentals and valuation most; treat short-term "
    "technical noise lightly.",
}


def build_facts(state: AnalystState) -> dict[str, Any]:
    """Everything the strategist may use, as JSON-ready data. Nothing else goes in."""
    technical = state["technical"]
    risk = state["risk"]
    fundamentals = state.get("fundamentals")
    sentiment = state.get("sentiment")
    if technical is None or risk is None:
        raise RuntimeError("Strategist ran without technical data or risk assessment")

    return {
        "company": require_ticker(state).model_dump(),
        "technical": technical.model_dump(mode="json"),
        "technical_observations": describe_technicals(technical),
        "fundamentals": fundamentals.model_dump(mode="json") if fundamentals else "unavailable",
        "news_sentiment": sentiment.model_dump(mode="json") if sentiment else "unavailable",
        "risk_assessment": risk.model_dump(mode="json"),
    }


async def strategist(state: AnalystState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    system = ctx.prompts.render("strategist", horizon_guidance=HORIZON_GUIDANCE[state["horizon"]])
    facts = json.dumps(build_facts(state), indent=2, default=str)
    writer = ctx.llm.with_structured_output(StrategyDraft)
    draft = cast(
        StrategyDraft,
        await writer.ainvoke([SystemMessage(system), HumanMessage(f"FACTS:\n{facts}")]),
    )
    return {"draft": draft}


def summarize_for_memory(report: AnalysisReport) -> str:
    """A compact text version of the report, kept in the chat history for follow-ups."""
    return (
        f"Analysis of {report.ticker.name} ({report.ticker.symbol}), {report.horizon} horizon: "
        f"{report.recommendation} with {report.conviction} conviction; risk {report.risk.level}. "
        f"{report.thesis}"
    )


def guardrail(state: AnalystState) -> dict[str, Any]:
    """Nothing reaches the user without passing through here."""
    draft, risk, technical = state["draft"], state["risk"], state["technical"]
    if draft is None or risk is None or technical is None:
        raise RuntimeError("Guardrail ran without a draft, risk assessment or technical data")

    checked, notes = apply_guardrails(
        draft,
        risk,
        horizon=state["horizon"],
        has_fundamentals=state.get("fundamentals") is not None,
    )
    if notes:
        log.info("guardrail_triggered", notes=notes)

    report = AnalysisReport(
        ticker=require_ticker(state),
        horizon=state["horizon"],
        generated_at=datetime.now(UTC),
        technical=technical,
        fundamentals=state.get("fundamentals"),
        sentiment=state.get("sentiment"),
        risk=risk,
        recommendation=checked.recommendation,
        conviction=checked.conviction,
        thesis=checked.thesis,
        bull_points=checked.bull_points,
        bear_points=checked.bear_points,
        guardrail_notes=notes,
    )
    return {
        "report": report,
        "messages": [AIMessage(summarize_for_memory(report))],
        "response_kind": "report",
    }
