"""Evaluators: functions that score one output. Langfuse stores each score on the trace.

Two kinds:
* deterministic checks for things with one right answer (cheap, exact, run on every case);
* an LLM-as-judge for qualities code can't check, like "is the thesis faithful to the
  data?". The judge's own reasoning is stored as the score comment, so it's auditable.
"""

from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import Evaluation
from langfuse.experiment import EvaluatorFunction
from pydantic import BaseModel, Field

from eq_analyst.agent.llm import build_chat_model
from eq_analyst.core.config import get_settings


def _check(name: str, expected: dict[str, Any], output: dict[str, Any]) -> list[Evaluation]:
    """Exact-match score for one field, skipped if the case doesn't specify it."""
    if name not in expected:
        return []
    passed = output.get(name) == expected[name]
    return [
        Evaluation(
            name=f"{name}_correct",
            value=1.0 if passed else 0.0,
            data_type="BOOLEAN",
            comment=f"expected={expected[name]!r} got={output.get(name)!r}",
        )
    ]


def intent_correct(*, output: dict[str, Any], expected_output: dict[str, Any], **_: Any) -> Any:
    return _check("intent", expected_output, output)


def symbol_correct(*, output: dict[str, Any], expected_output: dict[str, Any], **_: Any) -> Any:
    return _check("symbol", expected_output, output)


def horizon_correct(*, output: dict[str, Any], expected_output: dict[str, Any], **_: Any) -> Any:
    return _check("horizon", expected_output, output)


def policy_compliant(*, output: dict[str, Any], **_: Any) -> Any:
    """Invariant: a CRITICAL-risk report must never say BUY or size a position above 0%."""
    report = output.get("report")
    if report is None:
        return []
    critical = report["risk"]["level"] == "CRITICAL"
    violation = critical and (
        report["recommendation"] == "BUY" or report["risk"]["position_size_pct"] > 0
    )
    return Evaluation(name="policy_compliant", value=0.0 if violation else 1.0, data_type="BOOLEAN")


class JudgeVerdict(BaseModel):
    reasoning: str = Field(description="Briefly list any unsupported or contradicted claims.")
    score: float = Field(ge=0, le=1, description="1 = fully supported by the facts, 0 = invented.")


JUDGE_PROMPT = """You are auditing a stock-research assistant. Given the FACTS it had and
the THESIS it wrote, check every claim in the thesis. Penalise any number, event or
claim that does not appear in, or contradicts, the facts. Do not judge whether the
recommendation is a good investment; judge only faithfulness to the facts."""


async def faithfulness(*, output: dict[str, Any], **_: Any) -> Any:
    """LLM-as-judge: is the written thesis grounded in the data we gave the model?"""
    report = output.get("report")
    if report is None:
        return []
    facts = {k: report[k] for k in ("technical", "fundamentals", "sentiment", "risk")}
    judge = build_chat_model(get_settings()).with_structured_output(JudgeVerdict)
    verdict = cast(
        JudgeVerdict,
        await judge.ainvoke(
            [
                SystemMessage(JUDGE_PROMPT),
                HumanMessage(
                    f"FACTS:\n{json.dumps(facts, default=str)}\n\n"
                    f"THESIS:\n{report['thesis']}\n"
                    f"BULL POINTS: {report['bull_points']}\nBEAR POINTS: {report['bear_points']}"
                ),
            ]
        ),
    )
    return Evaluation(name="faithfulness", value=verdict.score, comment=verdict.reasoning)


def average_scores(*, item_results: list[Any], **_: Any) -> Any:
    """Run-level evaluator: the mean of each score across the whole dataset."""
    totals: dict[str, list[float]] = {}
    for result in item_results:
        for evaluation in result.evaluations:
            if isinstance(evaluation.value, int | float):
                totals.setdefault(evaluation.name, []).append(float(evaluation.value))
    return [
        Evaluation(name=f"avg_{name}", value=sum(values) / len(values))
        for name, values in totals.items()
    ]


# cast: our evaluators accept the subset of keyword arguments they use (plus **_).
ITEM_EVALUATORS = cast(
    list[EvaluatorFunction],
    [intent_correct, symbol_correct, horizon_correct, policy_compliant, faithfulness],
)
