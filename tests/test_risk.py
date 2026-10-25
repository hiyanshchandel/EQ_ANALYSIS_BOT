from eq_analyst.domain.models import Fundamentals, RiskLevel, Sentiment
from eq_analyst.domain.risk import assess_risk
from tests.fakes import make_snapshot


def test_healthy_stock_is_low_risk_and_position_is_capped() -> None:
    risk = assess_risk(make_snapshot(), fundamentals=None, sentiment=None)
    assert risk.level == RiskLevel.LOW
    assert risk.flags == []
    # stop = 100 - 2 * 2.0 = 96 -> 4% away; 1% / 4% = 25% position, capped at 10% for LOW.
    assert risk.stop_loss == 96.0
    assert risk.stop_distance_pct == 4.0
    assert risk.position_size_pct == 10.0


def test_flags_stack_up_to_critical_and_block_new_positions() -> None:
    risky = make_snapshot(
        price=80.0,
        sma_50=95.0,
        sma_200=100.0,  # downtrend
        macd=-1.0,
        macd_signal=0.5,  # bearish momentum
        atr_14=4.0,
        atr_pct=5.0,  # high volatility
    )
    leveraged = Fundamentals(symbol="X", debt_to_equity=3.1)
    gloomy = Sentiment(score=-0.7, label="negative", themes=["fraud probe"], summary="Bad news.")

    risk = assess_risk(risky, leveraged, gloomy)
    codes = {flag.code for flag in risk.flags}
    assert codes == {
        "downtrend",
        "bearish_momentum",
        "high_volatility",
        "high_leverage",
        "negative_news",
    }
    assert risk.level == RiskLevel.CRITICAL
    assert risk.position_size_pct == 0.0


def test_debt_to_equity_below_threshold_is_not_flagged() -> None:
    # Regression test for the v1 bug: Yahoo's 45.0 (percent) used to be compared with 2.
    healthy = Fundamentals(symbol="X", debt_to_equity=0.45)
    risk = assess_risk(make_snapshot(), healthy, sentiment=None)
    assert "high_leverage" not in {flag.code for flag in risk.flags}


def test_overbought_is_a_flag() -> None:
    risk = assess_risk(make_snapshot(rsi_14=78.0), None, None)
    assert [flag.code for flag in risk.flags] == ["overbought"]
