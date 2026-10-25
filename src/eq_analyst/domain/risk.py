"""Rule-based risk scoring, stop loss and position sizing.

This replaces v1's approach of asking the LLM to "calculate" a stop loss. The rules
are simple, explainable and testable:

* Each red flag adds 1 point. The total maps to a risk level.
* Stop loss = price - 2 x ATR, i.e. outside the stock's normal daily noise.
* Position size uses the fixed-fractional method: risk at most 1% of the portfolio on
  the trade, so size = 1% / stop distance. That is then capped by the risk level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eq_analyst.domain.models import (
    Fundamentals,
    RiskAssessment,
    RiskFlag,
    RiskLevel,
    Sentiment,
    TechnicalSnapshot,
)
from eq_analyst.domain.signals import RSI_OVERBOUGHT, trend_label


@dataclass(frozen=True)
class RiskPolicy:
    """Tunable thresholds. Frozen so a policy can't be mutated mid-request."""

    max_debt_to_equity: float = 2.0
    max_atr_pct: float = 3.5
    min_sentiment_score: float = -0.3
    stop_atr_multiple: float = 2.0
    portfolio_risk_per_trade_pct: float = 1.0
    max_position_pct: dict[RiskLevel, float] = field(
        default_factory=lambda: {
            RiskLevel.LOW: 10.0,
            RiskLevel.MEDIUM: 7.0,
            RiskLevel.HIGH: 4.0,
            RiskLevel.CRITICAL: 0.0,  # don't open a new position at all
        }
    )


MAX_SCORE = 6


def _level_for(score: int) -> RiskLevel:
    if score <= 1:
        return RiskLevel.LOW
    if score == 2:
        return RiskLevel.MEDIUM
    if score <= 4:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


def collect_flags(
    technical: TechnicalSnapshot,
    fundamentals: Fundamentals | None,
    sentiment: Sentiment | None,
    policy: RiskPolicy,
) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    if trend_label(technical) == "downtrend":
        flags.append(
            RiskFlag(code="downtrend", description="Price is below its 50- and 200-day averages.")
        )
    if technical.macd < technical.macd_signal:
        flags.append(
            RiskFlag(code="bearish_momentum", description="MACD is below its signal line.")
        )
    if technical.rsi_14 > RSI_OVERBOUGHT:
        flags.append(
            RiskFlag(code="overbought", description=f"RSI {technical.rsi_14} is above 70.")
        )
    if technical.atr_pct > policy.max_atr_pct:
        flags.append(
            RiskFlag(
                code="high_volatility",
                description=f"Average daily range is {technical.atr_pct}% of price.",
            )
        )
    if (
        fundamentals is not None
        and fundamentals.debt_to_equity is not None
        and fundamentals.debt_to_equity > policy.max_debt_to_equity
    ):
        flags.append(
            RiskFlag(
                code="high_leverage", description=f"Debt/equity is {fundamentals.debt_to_equity}x."
            )
        )
    if sentiment is not None and sentiment.score < policy.min_sentiment_score:
        flags.append(
            RiskFlag(code="negative_news", description=f"News sentiment is {sentiment.score}.")
        )
    return flags


def assess_risk(
    technical: TechnicalSnapshot,
    fundamentals: Fundamentals | None,
    sentiment: Sentiment | None,
    policy: RiskPolicy | None = None,
) -> RiskAssessment:
    policy = policy or RiskPolicy()
    flags = collect_flags(technical, fundamentals, sentiment, policy)
    level = _level_for(len(flags))

    stop_loss = technical.price - policy.stop_atr_multiple * technical.atr_14
    stop_distance_pct = (technical.price - stop_loss) / technical.price * 100

    # Fixed-fractional sizing: if the stop is hit we lose `risk_per_trade` % of the portfolio.
    # e.g. 1% risk budget / 5% stop distance -> a 20% position (before the level cap).
    cap = policy.max_position_pct[level]
    if stop_distance_pct > 0:
        sized_by_stop = policy.portfolio_risk_per_trade_pct / stop_distance_pct * 100
        position_size_pct = min(sized_by_stop, cap)
    else:  # degenerate data (ATR of zero): fall back to the cap
        position_size_pct = cap

    return RiskAssessment(
        score=len(flags),
        max_score=MAX_SCORE,
        level=level,
        flags=flags,
        stop_loss=round(stop_loss, 2),
        stop_distance_pct=round(stop_distance_pct, 2),
        position_size_pct=round(position_size_pct, 1),
    )
