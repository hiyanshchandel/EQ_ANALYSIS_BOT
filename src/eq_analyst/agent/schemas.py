"""Structured-output schemas that only the agent uses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from eq_analyst.domain.models import Horizon


class Intent(BaseModel):
    """LLM output schema: what the user wants from this message."""

    kind: Literal["analyze", "follow_up", "other"] = Field(
        description=(
            "'analyze' = analyse a company; 'follow_up' = question about the stock already "
            "being discussed; 'other' = greetings, help or anything unrelated."
        )
    )
    company: str | None = Field(
        default=None,
        description="For 'analyze': the company name or symbol exactly as the user wrote it.",
    )
    horizon: Horizon | None = Field(
        default=None,
        description="Only if the user states a timeframe: short (<2 weeks), medium, long (3+ mo).",
    )
