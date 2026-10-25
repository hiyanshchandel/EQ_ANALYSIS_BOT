"""Checkpoint persistence, prompt rendering and Telegram formatting."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from eq_analyst.agent.checkpoint import sqlite_checkpointer
from eq_analyst.agent.graph import build_graph
from eq_analyst.agent.prompts import LOCAL_PROMPTS, PromptRegistry, render_template
from eq_analyst.agent.schemas import Intent
from eq_analyst.bot.formatting import format_report
from eq_analyst.domain.models import (
    AnalysisReport,
    Conviction,
    Horizon,
    Recommendation,
    Sentiment,
    StrategyDraft,
)
from eq_analyst.domain.risk import assess_risk
from tests.fakes import TCS, FakeLLM, make_context, make_snapshot


async def test_report_survives_a_restart(tmp_path: Path) -> None:
    """Simulates a bot restart: a new graph + new DB connection still sees the old report."""
    db = tmp_path / "checkpoints.sqlite"
    llm = FakeLLM(
        {
            Intent: [Intent(kind="analyze", company="tcs"), Intent(kind="follow_up")],
            Sentiment: Sentiment(score=0, label="neutral", themes=[], summary="Quiet."),
            StrategyDraft: StrategyDraft(
                recommendation=Recommendation.HOLD,
                conviction=Conviction.LOW,
                thesis="Wait.",
                bull_points=[],
                bear_points=[],
            ),
        }
    )
    context = make_context(llm)
    config = {"configurable": {"thread_id": "42"}}

    async with sqlite_checkpointer(db) as saver:
        await build_graph(saver).ainvoke(
            {"messages": [HumanMessage("analyze tcs")]}, config, context=context
        )

    async with sqlite_checkpointer(db) as saver:  # "restart"
        state = await build_graph(saver).aget_state(config)
    assert state.values["report"].ticker.symbol == "TCS.NS"
    assert isinstance(state.values["report"], AnalysisReport)


def test_every_prompt_renders_with_its_variables() -> None:
    registry = PromptRegistry(langfuse=None)
    assert "TCS" in registry.render("intent-router", current_company="TCS")
    assert "TCS.NS" in registry.render("sentiment-analyst", company="TCS", symbol="TCS.NS")
    assert "LONG" in registry.render("strategist", horizon_guidance="LONG")
    assert "{}" in registry.render("follow-up-agent", report_json="{}")
    assert set(LOCAL_PROMPTS) == {
        "intent-router",
        "sentiment-analyst",
        "strategist",
        "follow-up-agent",
    }


def test_missing_prompt_variable_fails_loudly() -> None:
    with pytest.raises(KeyError):
        render_template("Hello {{name}}")


def test_report_html_escapes_llm_text() -> None:
    snapshot = make_snapshot()
    report = AnalysisReport(
        ticker=TCS,
        horizon=Horizon.MEDIUM,
        generated_at=datetime.now(UTC),
        technical=snapshot,
        fundamentals=None,
        sentiment=None,
        risk=assess_risk(snapshot, None, None),
        recommendation=Recommendation.BUY,
        conviction=Conviction.MEDIUM,
        thesis="P/E < peers & <b>cheap</b>",
        bull_points=["a<b"],
        bear_points=[],
    )
    html = format_report(report)
    assert "&lt;b&gt;cheap&lt;/b&gt;" in html
    assert "₹100.00" in html
    assert "not financial advice" in html
