"""Turn the company name the user typed into a tradable symbol."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.runtime import Runtime

from eq_analyst.agent.context import AgentContext
from eq_analyst.agent.market_data import MarketDataError
from eq_analyst.agent.state import AnalystState
from eq_analyst.core.log import get_logger
from eq_analyst.domain.ticker_selection import pick_best_match

log = get_logger(__name__)

ANALYSTS = ["technical_analyst", "fundamental_analyst", "sentiment_analyst"]


async def resolve_ticker(state: AnalystState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    query = state["intent"].company or ""
    try:
        result = await ctx.market_data.search_companies(query)
    except MarketDataError as exc:
        log.warning("ticker_search_failed", query=query, error=str(exc))
        return {"ticker": None}

    match = pick_best_match(query, result.matches, ctx.preferred_exchanges)
    log.info("ticker_resolved", query=query, symbol=match.symbol if match else None)
    return {"ticker": match}


def route_after_resolve(state: AnalystState) -> list[str] | Literal["reply_not_found"]:
    """Returning a *list* of node names fans out: the three analysts run in parallel."""
    if state.get("ticker") is None:
        return "reply_not_found"
    return ANALYSTS
