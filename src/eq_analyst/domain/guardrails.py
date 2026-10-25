"""Deterministic guardrails applied to the LLM's recommendation.

An LLM can be persuasive and wrong. Before anything reaches a user, these rules
enforce the firm's risk policy no matter what the model wrote. Every intervention is
recorded as a note, shown to the user and sent to Langfuse as a score.
"""

from __future__ import annotations

from eq_analyst.domain.models import (
    Conviction,
    Horizon,
    Recommendation,
    RiskAssessment,
    RiskLevel,
    StrategyDraft,
)


def apply_guardrails(
    draft: StrategyDraft,
    risk: RiskAssessment,
    *,
    horizon: Horizon,
    has_fundamentals: bool,
) -> tuple[StrategyDraft, list[str]]:
    """Return a policy-compliant copy of `draft` plus a note for every change made."""
    recommendation = draft.recommendation
    conviction = draft.conviction
    notes: list[str] = []

    # Rule 1: never recommend opening a position the risk policy sizes at 0%.
    if risk.level == RiskLevel.CRITICAL and recommendation == Recommendation.BUY:
        recommendation = Recommendation.HOLD
        notes.append("BUY downgraded to HOLD: risk level is CRITICAL.")

    # Rule 2: high risk and high conviction don't go together.
    if risk.level == RiskLevel.HIGH and conviction == Conviction.HIGH:
        conviction = Conviction.MEDIUM
        notes.append("Conviction lowered to MEDIUM: risk level is HIGH.")

    # Rule 3: a long-term call without fundamentals is mostly guesswork.
    if horizon == Horizon.LONG and not has_fundamentals and conviction != Conviction.LOW:
        conviction = Conviction.LOW
        notes.append("Conviction lowered to LOW: no fundamental data for a long-term view.")

    checked = draft.model_copy(update={"recommendation": recommendation, "conviction": conviction})
    return checked, notes
