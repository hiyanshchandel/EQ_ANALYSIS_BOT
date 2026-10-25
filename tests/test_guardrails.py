from eq_analyst.domain.guardrails import apply_guardrails
from eq_analyst.domain.models import (
    Conviction,
    Horizon,
    Recommendation,
    RiskAssessment,
    RiskLevel,
    StrategyDraft,
)


def _draft(recommendation: Recommendation, conviction: Conviction) -> StrategyDraft:
    return StrategyDraft(
        recommendation=recommendation,
        conviction=conviction,
        thesis="t",
        bull_points=[],
        bear_points=[],
    )


def _risk(level: RiskLevel) -> RiskAssessment:
    return RiskAssessment(
        score=0,
        max_score=6,
        level=level,
        flags=[],
        stop_loss=90.0,
        stop_distance_pct=10.0,
        position_size_pct=5.0,
    )


def test_critical_risk_blocks_buy() -> None:
    checked, notes = apply_guardrails(
        _draft(Recommendation.BUY, Conviction.MEDIUM),
        _risk(RiskLevel.CRITICAL),
        horizon=Horizon.MEDIUM,
        has_fundamentals=True,
    )
    assert checked.recommendation == Recommendation.HOLD
    assert len(notes) == 1


def test_high_risk_caps_conviction() -> None:
    checked, _ = apply_guardrails(
        _draft(Recommendation.SELL, Conviction.HIGH),
        _risk(RiskLevel.HIGH),
        horizon=Horizon.SHORT,
        has_fundamentals=True,
    )
    assert checked.recommendation == Recommendation.SELL
    assert checked.conviction == Conviction.MEDIUM


def test_long_horizon_without_fundamentals_is_low_conviction() -> None:
    checked, notes = apply_guardrails(
        _draft(Recommendation.BUY, Conviction.HIGH),
        _risk(RiskLevel.LOW),
        horizon=Horizon.LONG,
        has_fundamentals=False,
    )
    assert checked.conviction == Conviction.LOW
    assert notes


def test_compliant_draft_is_untouched() -> None:
    draft = _draft(Recommendation.BUY, Conviction.HIGH)
    checked, notes = apply_guardrails(
        draft, _risk(RiskLevel.LOW), horizon=Horizon.MEDIUM, has_fundamentals=True
    )
    assert checked == draft
    assert notes == []
