"""The three analysts. They run in parallel and each writes its own state key.

A failing analyst returns None instead of crashing the graph, so the report can still
be produced with partial data (the strategist is told what's missing). Only the
technical snapshot is mandatory, which the risk node checks.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from eq_analyst.agent.context import AgentContext
from eq_analyst.agent.market_data import MarketDataError
from eq_analyst.agent.nodes.common import require_ticker
from eq_analyst.agent.state import AnalystState
from eq_analyst.core.log import get_logger
from eq_analyst.domain.models import NewsItem, Sentiment

log = get_logger(__name__)

NEWS_LIMIT = 8


async def technical_analyst(state: AnalystState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """No LLM here: indicators are computed by the MCP server."""
    symbol = require_ticker(state).symbol
    try:
        return {"technical": await runtime.context.market_data.get_technical_snapshot(symbol)}
    except MarketDataError as exc:
        log.warning("technical_unavailable", symbol=symbol, error=str(exc))
        return {"technical": None}


async def fundamental_analyst(
    state: AnalystState, runtime: Runtime[AgentContext]
) -> dict[str, Any]:
    """No LLM here either: the facts go straight to the strategist."""
    symbol = require_ticker(state).symbol
    try:
        return {"fundamentals": await runtime.context.market_data.get_fundamentals(symbol)}
    except MarketDataError as exc:
        log.warning("fundamentals_unavailable", symbol=symbol, error=str(exc))
        return {"fundamentals": None}


def format_news(items: list[NewsItem]) -> str:
    lines = []
    for item in items:
        lines.append(f"- [{item.publisher or 'unknown'}] {item.title}")
        if item.summary:
            lines.append(f"  {item.summary}")
    return "\n".join(lines)


async def sentiment_analyst(state: AnalystState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """The one analyst that needs an LLM: reading news is a language task."""
    ctx = runtime.context
    ticker = require_ticker(state)
    try:
        feed = await ctx.market_data.get_news(ticker.symbol, limit=NEWS_LIMIT)
    except MarketDataError as exc:
        log.warning("news_unavailable", symbol=ticker.symbol, error=str(exc))
        return {"sentiment": None}
    if not feed.items:
        return {"sentiment": None}

    system = ctx.prompts.render("sentiment-analyst", company=ticker.name, symbol=ticker.symbol)
    # News is untrusted text, so it is fenced in tags and the prompt says to treat it as data.
    user = HumanMessage(f"<news>\n{format_news(feed.items)}\n</news>")
    scorer = ctx.llm.with_structured_output(Sentiment)
    sentiment = cast(Sentiment, await scorer.ainvoke([SystemMessage(system), user]))
    return {"sentiment": sentiment.model_copy(update={"headline_count": len(feed.items)})}
