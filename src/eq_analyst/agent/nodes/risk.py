"""Join point after the parallel analysts: deterministic risk assessment."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.runtime import Runtime

from eq_analyst.agent.context import AgentContext
from eq_analyst.agent.state import AnalystState
from eq_analyst.domain.risk import assess_risk


def risk_manager(state: AnalystState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    technical = state.get("technical")
    if technical is None:  # without prices there is nothing to assess
        return {"risk": None}
    risk = assess_risk(
        technical, state.get("fundamentals"), state.get("sentiment"), runtime.context.risk_policy
    )
    return {"risk": risk}


def route_after_risk(state: AnalystState) -> Literal["strategist", "reply_data_unavailable"]:
    return "strategist" if state.get("risk") is not None else "reply_data_unavailable"
