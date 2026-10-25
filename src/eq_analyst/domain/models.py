"""Typed data contracts shared by the MCP server, the agent and the bot.

These Pydantic models are the "language" the components speak. The MCP server returns
them as structured tool output, the agent stores them in graph state, and the LLM is
forced to answer in the shape of `Sentiment` and `StrategyDraft`.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Horizon(StrEnum):
    SHORT = "short"  # days to 2 weeks
    MEDIUM = "medium"  # 2 weeks to 3 months
    LONG = "long"  # 3+ months


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Recommendation(StrEnum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class Conviction(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# --- Market data (produced by the MCP server) --------------------------------


class TickerMatch(BaseModel):
    symbol: str = Field(description="Yahoo Finance symbol, e.g. 'TCS.NS'.")
    name: str
    exchange: str = Field(description="Yahoo exchange code, e.g. 'NSI' for NSE India.")
    exchange_display: str | None = None
    quote_type: str = Field(description="'EQUITY', 'ETF', 'INDEX', ...")


class TickerSearchResult(BaseModel):
    query: str
    matches: list[TickerMatch]


class TechnicalSnapshot(BaseModel):
    symbol: str
    currency: str | None = None
    as_of: date
    price: float
    change_1d_pct: float
    sma_20: float
    sma_50: float
    sma_200: float | None = Field(default=None, description="None if under 200 days of data.")
    rsi_14: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_lower: float
    atr_14: float
    atr_pct: float = Field(description="ATR as a percentage of price (daily volatility).")
    volume_ratio: float = Field(description="Latest volume / average of the previous 20 days.")
    high_52w: float
    low_52w: float


class Fundamentals(BaseModel):
    symbol: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    currency: str | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    debt_to_equity: float | None = Field(
        default=None, description="As a ratio (1.5 means debt is 1.5x equity)."
    )
    roe_pct: float | None = None
    profit_margin_pct: float | None = None
    revenue_growth_pct: float | None = None


class NewsItem(BaseModel):
    title: str
    publisher: str | None = None
    published_at: datetime | None = None
    summary: str | None = None
    url: str | None = None


class NewsFeed(BaseModel):
    symbol: str
    items: list[NewsItem]


# --- Analysis (produced by the agent) -----------------------------------------


class Sentiment(BaseModel):
    """LLM output schema: news sentiment for one company."""

    score: float = Field(
        ge=-1.0, le=1.0, description="-1 very negative, 0 neutral, +1 very positive."
    )
    label: Literal["positive", "neutral", "negative"]
    themes: list[str] = Field(description="Up to 3 short themes driving sentiment.", max_length=3)
    summary: str = Field(description="One or two sentences explaining the score.")
    headline_count: int = Field(default=0, description="Filled in by code, not by the model.")


class RiskFlag(BaseModel):
    code: str
    description: str


class RiskAssessment(BaseModel):
    score: int
    max_score: int
    level: RiskLevel
    flags: list[RiskFlag]
    stop_loss: float
    stop_distance_pct: float
    position_size_pct: float


class StrategyDraft(BaseModel):
    """LLM output schema: the strategist's recommendation before guardrails run."""

    recommendation: Recommendation
    conviction: Conviction
    thesis: str = Field(description="3-5 sentences. Only cite numbers that appear in the facts.")
    bull_points: list[str] = Field(
        description="Up to 3 reasons the stock could do well.", max_length=3
    )
    bear_points: list[str] = Field(
        description="Up to 3 reasons the stock could do badly.", max_length=3
    )


class AnalysisReport(BaseModel):
    """The final, guardrail-checked answer that the bot renders."""

    ticker: TickerMatch
    horizon: Horizon
    generated_at: datetime
    technical: TechnicalSnapshot
    fundamentals: Fundamentals | None
    sentiment: Sentiment | None
    risk: RiskAssessment
    recommendation: Recommendation
    conviction: Conviction
    thesis: str
    bull_points: list[str]
    bear_points: list[str]
    guardrail_notes: list[str] = Field(default_factory=list)


# Models that may be stored in (and loaded back from) LangGraph checkpoints.
# LangGraph refuses to deserialize unknown classes, so we allow-list ours explicitly.
CHECKPOINT_TYPES: tuple[type, ...] = (
    Horizon,
    RiskLevel,
    Recommendation,
    Conviction,
    TickerMatch,
    TechnicalSnapshot,
    Fundamentals,
    Sentiment,
    RiskFlag,
    RiskAssessment,
    StrategyDraft,
    AnalysisReport,
)
