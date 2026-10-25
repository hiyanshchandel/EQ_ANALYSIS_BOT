"""Technical indicators, computed deterministically with pandas.

v1 asked the LLM to reason about numbers it could not compute reliably. Here every
indicator is plain arithmetic, unit-tested, and the LLM only *interprets* the results.
"""

from __future__ import annotations

import pandas as pd

from eq_analyst.domain.models import TechnicalSnapshot

MIN_HISTORY_DAYS = 60  # below this, 50-day averages and MACD are not meaningful


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average: the mean of the last `window` values."""
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average: recent values weigh more than old ones."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (the standard definition).

    RSI = 100 - 100 / (1 + average_gain / average_loss). Above 70 is conventionally
    "overbought", below 30 "oversold". Wilder's smoothing is an EMA with alpha = 1/period.
    """
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = avg_gain / avg_loss
    result = 100 - (100 / (1 + relative_strength))
    # If there were no losses at all, avg_loss is 0 and RSI is by definition 100.
    return result.where(avg_loss != 0, 100.0)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line and histogram.

    MACD = EMA(12) - EMA(26). Signal = EMA(9) of MACD. Histogram = MACD - signal.
    MACD above its signal line is read as bullish momentum.
    """
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Upper, middle and lower Bollinger Bands (20-day mean +/- 2 standard deviations)."""
    middle = sma(close, window)
    std = close.rolling(window=window).std()
    return middle + num_std * std, middle, middle - num_std * std


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range: the typical daily price movement, used for stop losses.

    True range is the largest of: today's high-low, |high - yesterday's close| and
    |low - yesterday's close| (the last two capture overnight gaps).
    """
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def volume_ratio(volume: pd.Series, window: int = 20) -> float:
    """Latest volume divided by the average of the *previous* `window` sessions."""
    baseline = volume.iloc[-(window + 1) : -1].mean()
    if not baseline or pd.isna(baseline):
        return 1.0
    return float(volume.iloc[-1] / baseline)


def build_snapshot(symbol: str, history: pd.DataFrame, currency: str | None) -> TechnicalSnapshot:
    """Turn ~1 year of daily OHLCV candles into a `TechnicalSnapshot`.

    `history` must have columns Open, High, Low, Close, Volume, indexed by date.
    """
    history = history.dropna(subset=["Close"])
    if len(history) < MIN_HISTORY_DAYS:
        raise ValueError(
            f"{symbol}: need at least {MIN_HISTORY_DAYS} days of history, got {len(history)}"
        )

    close, high, low = history["Close"], history["High"], history["Low"]
    macd_line, signal_line, histogram = macd(close)
    upper, _, lower = bollinger_bands(close)
    atr_series = atr(high, low, close)

    price = float(close.iloc[-1])
    sma_200 = sma(close, 200).iloc[-1]
    last_atr = float(atr_series.iloc[-1])

    return TechnicalSnapshot(
        symbol=symbol,
        currency=currency,
        as_of=pd.Timestamp(history.index[-1]).date(),
        price=round(price, 2),
        change_1d_pct=round((price / float(close.iloc[-2]) - 1) * 100, 2),
        sma_20=round(float(sma(close, 20).iloc[-1]), 2),
        sma_50=round(float(sma(close, 50).iloc[-1]), 2),
        sma_200=None if pd.isna(sma_200) else round(float(sma_200), 2),
        rsi_14=round(float(rsi(close).iloc[-1]), 2),
        macd=round(float(macd_line.iloc[-1]), 4),
        macd_signal=round(float(signal_line.iloc[-1]), 4),
        macd_hist=round(float(histogram.iloc[-1]), 4),
        bb_upper=round(float(upper.iloc[-1]), 2),
        bb_lower=round(float(lower.iloc[-1]), 2),
        atr_14=round(last_atr, 2),
        atr_pct=round(last_atr / price * 100, 2),
        volume_ratio=round(volume_ratio(history["Volume"]), 2),
        high_52w=round(float(high.tail(252).max()), 2),
        low_52w=round(float(low.tail(252).min()), 2),
    )
