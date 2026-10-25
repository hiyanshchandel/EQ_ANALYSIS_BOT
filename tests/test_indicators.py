import pandas as pd
import pytest

from eq_analyst.domain.indicators import atr, build_snapshot, macd, rsi, sma, volume_ratio
from tests.fakes import make_history


def test_sma_is_rolling_mean() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sma(series, 3).tolist()[2:] == [2.0, 3.0, 4.0]


def test_rsi_is_100_when_price_only_rises() -> None:
    rising = pd.Series(range(1, 40), dtype=float)
    assert rsi(rising).iloc[-1] == 100.0


def test_rsi_is_low_when_price_mostly_falls() -> None:
    falling = pd.Series([100 - i + (0.5 if i % 5 == 0 else 0) for i in range(40)], dtype=float)
    assert rsi(falling).iloc[-1] < 30


def test_macd_positive_in_uptrend() -> None:
    line, _, _ = macd(pd.Series(range(1, 80), dtype=float))
    assert line.iloc[-1] > 0


def test_atr_equals_constant_daily_range() -> None:
    n = 40
    close = pd.Series([100.0] * n)
    result = atr(close + 1, close - 1, close)  # every day ranges exactly 2.0
    assert result.iloc[-1] == pytest.approx(2.0)


def test_volume_ratio_excludes_today_from_baseline() -> None:
    volume = pd.Series([100.0] * 20 + [300.0])
    assert volume_ratio(volume) == pytest.approx(3.0)


def test_build_snapshot_from_uptrend() -> None:
    snapshot = build_snapshot("TEST", make_history(daily_drift=0.003), "INR")
    assert snapshot.price > snapshot.sma_50 > 0
    assert snapshot.sma_200 is not None
    assert snapshot.low_52w <= snapshot.price <= snapshot.high_52w
    assert 0 <= snapshot.rsi_14 <= 100


def test_build_snapshot_rejects_short_history() -> None:
    with pytest.raises(ValueError, match="at least"):
        build_snapshot("NEW", make_history(days=30), "INR")
