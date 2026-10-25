"""Small helpers shared by several nodes."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AnyMessage, HumanMessage

from eq_analyst.agent.state import AnalystState
from eq_analyst.domain.models import TickerMatch


def require_ticker(state: AnalystState) -> TickerMatch:
    """Analysts only run after a ticker is resolved; fail loudly if wiring breaks that."""
    ticker = state.get("ticker")
    if ticker is None:
        raise RuntimeError("Analyst node ran without a resolved ticker")
    return ticker


def recent_messages(messages: Sequence[AnyMessage], limit: int) -> list[AnyMessage]:
    """The last `limit` messages, trimmed so the window starts with a user message.

    Sending only a recent window keeps prompts small and cheap as a chat grows. Chat
    models expect the conversation to open with a user turn, hence the trimming.
    """
    window = list(messages[-limit:])
    while window and not isinstance(window[0], HumanMessage):
        window.pop(0)
    return window
