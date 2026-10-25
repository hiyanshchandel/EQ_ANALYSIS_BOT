"""Terminal nodes with fixed replies. No LLM: these answers never need to vary."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from eq_analyst.agent.state import AnalystState

HELP_TEXT = (
    "I analyse listed stocks. Just name a company, for example:\n"
    "• Analyze Tata Consultancy Services\n"
    "• How does HDFC Bank look for the long term?\n"
    "• Reliance short-term view\n\n"
    "After an analysis you can ask follow-ups like 'why?' or 'compare it with Infosys'.\n"
    "Use /reset to start a fresh conversation."
)


def _reply(text: str) -> dict[str, Any]:
    return {"messages": [AIMessage(text)], "response_kind": "message"}


def reply_help(state: AnalystState) -> dict[str, Any]:
    return _reply(HELP_TEXT)


def reply_not_found(state: AnalystState) -> dict[str, Any]:
    company = state["intent"].company
    return _reply(
        f"I couldn't find a listed company matching '{company}'. "
        "Try the full name (e.g. 'Tata Consultancy Services') or the symbol (e.g. 'TCS.NS')."
    )


def reply_data_unavailable(state: AnalystState) -> dict[str, Any]:
    ticker = state.get("ticker")
    name = f"{ticker.name} ({ticker.symbol})" if ticker else "that company"
    return _reply(
        f"I found {name} but couldn't load its price data right now. Please try again shortly."
    )
