"""Translate indicator numbers into plain-English observations.

The strategist LLM receives both the raw numbers and these sentences. Doing the
interpretation in code means "is RSI above 70?" is answered by Python, not guessed by
a model, and the wording is consistent across every report.
"""

from __future__ import annotations

from eq_analyst.domain.models import TechnicalSnapshot

RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_SPIKE_RATIO = 2.0


def trend_label(snapshot: TechnicalSnapshot) -> str:
    """Classify the trend from price vs the 50- and 200-day averages."""
    long_average = snapshot.sma_200 or snapshot.sma_50
    if snapshot.price > snapshot.sma_50 and snapshot.price > long_average:
        return "uptrend"
    if snapshot.price < snapshot.sma_50 and snapshot.price < long_average:
        return "downtrend"
    return "sideways"


def describe_technicals(snapshot: TechnicalSnapshot) -> list[str]:
    s = snapshot
    observations = [
        f"Trend: {trend_label(s)} (price {s.price} vs SMA50 {s.sma_50}, SMA200 {s.sma_200})."
    ]

    if s.rsi_14 > RSI_OVERBOUGHT:
        observations.append(f"RSI {s.rsi_14} is overbought; pullback risk is elevated.")
    elif s.rsi_14 < RSI_OVERSOLD:
        observations.append(f"RSI {s.rsi_14} is oversold; selling may be exhausted.")
    else:
        observations.append(f"RSI {s.rsi_14} is neutral.")

    direction = "bullish" if s.macd > s.macd_signal else "bearish"
    observations.append(f"MACD is {direction} (MACD {s.macd} vs signal {s.macd_signal}).")

    if s.price >= s.bb_upper:
        observations.append(
            "Price is at or above the upper Bollinger Band (stretched to the upside)."
        )
    elif s.price <= s.bb_lower:
        observations.append(
            "Price is at or below the lower Bollinger Band (stretched to the downside)."
        )

    if s.volume_ratio >= VOLUME_SPIKE_RATIO:
        observations.append(
            f"Volume is {s.volume_ratio}x the 20-day average: unusual activity, check the news."
        )

    distance_from_high = (s.price / s.high_52w - 1) * 100
    observations.append(
        f"Price is {distance_from_high:.1f}% from its 52-week high of {s.high_52w}."
    )
    return observations
