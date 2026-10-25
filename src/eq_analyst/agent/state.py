"""The graph's shared state.

LangGraph passes this dict through every node. A node returns *only the keys it
changed*; LangGraph merges them in. By default a returned key overwrites the old
value. `messages` is different: its reducer, `add_messages`, *appends*, which is how the
conversation history grows across turns.

The whole state is saved by the checkpointer after every step, keyed by thread id
(= Telegram chat id). That's what lets "why?" refer back to the previous analysis.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from eq_analyst.agent.schemas import Intent
from eq_analyst.domain.models import (
    AnalysisReport,
    Fundamentals,
    Horizon,
    RiskAssessment,
    Sentiment,
    StrategyDraft,
    TechnicalSnapshot,
    TickerMatch,
)


class AnalystState(TypedDict, total=False):
    # Conversation memory (persists across turns).
    messages: Annotated[list[AnyMessage], add_messages]
    # The last completed analysis (persists, so follow-ups can use it).
    report: AnalysisReport | None

    # Per-turn working data (reset by `intake` at the start of every turn).
    intent: Intent
    horizon: Horizon
    ticker: TickerMatch | None
    technical: TechnicalSnapshot | None
    fundamentals: Fundamentals | None
    sentiment: Sentiment | None
    risk: RiskAssessment | None
    draft: StrategyDraft | None
    # Tells the bot how to render the reply.
    response_kind: Literal["report", "message"]
