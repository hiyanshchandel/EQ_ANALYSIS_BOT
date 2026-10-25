"""End-to-end tests of the real LangGraph graph, with fake LLM and fake market data."""

from typing import Any

from langchain_core.messages import HumanMessage

from eq_analyst.agent.checkpoint import memory_checkpointer
from eq_analyst.agent.graph import build_graph
from eq_analyst.agent.schemas import Intent
from eq_analyst.domain.models import (
    Conviction,
    Horizon,
    Recommendation,
    RiskLevel,
    Sentiment,
    StrategyDraft,
)
from tests.fakes import FakeLLM, FakeMarketData, make_context, make_snapshot

NEUTRAL_NEWS = Sentiment(
    score=0.1, label="neutral", themes=["deal win"], summary="Mildly positive."
)
BUY_DRAFT = StrategyDraft(
    recommendation=Recommendation.BUY,
    conviction=Conviction.HIGH,
    thesis="Strong uptrend.",
    bull_points=["momentum"],
    bear_points=["valuation"],
)


async def _say(graph: Any, context: Any, text: str, thread: str = "chat-1") -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread}}
    return await graph.ainvoke({"messages": [HumanMessage(text)]}, config, context=context)  # type: ignore[no-any-return]


async def test_analysis_then_follow_up_uses_memory() -> None:
    llm = FakeLLM(
        {
            Intent: [
                Intent(kind="analyze", company="tcs", horizon=Horizon.LONG),
                Intent(kind="follow_up"),
            ],
            Sentiment: NEUTRAL_NEWS,
            StrategyDraft: BUY_DRAFT,
        }
    )
    context = make_context(llm)
    graph = build_graph(memory_checkpointer())

    first = await _say(graph, context, "how is tcs for the long term?")
    assert first["response_kind"] == "report"
    report = first["report"]
    assert report.ticker.symbol == "TCS.NS"
    assert report.horizon == Horizon.LONG
    assert report.recommendation == Recommendation.BUY
    assert report.sentiment.headline_count == 1

    second = await _say(graph, context, "why?")
    assert second["response_kind"] == "message"
    assert second["messages"][-1].text == "Because momentum is strong."
    # The report from turn 1 was loaded from the checkpointer and kept the horizon.
    assert second["report"].ticker.symbol == "TCS.NS"
    assert second["horizon"] == Horizon.LONG
    # The follow-up agent saw the conversation, starting with the user's first message.
    agent_input = context.follow_up_agent.inputs[0]["messages"]  # type: ignore[attr-defined]
    assert agent_input[1].text == "how is tcs for the long term?"


async def test_unknown_company_gets_a_helpful_reply() -> None:
    llm = FakeLLM({Intent: Intent(kind="analyze", company="xyzzy corp")})
    context = make_context(llm, FakeMarketData(matches=[]))
    result = await _say(build_graph(memory_checkpointer()), context, "analyze xyzzy corp")
    assert result["response_kind"] == "message"
    assert "couldn't find" in result["messages"][-1].text
    assert [schema for schema, _ in llm.calls] == [Intent]  # no wasted LLM calls


async def test_missing_price_data_stops_before_the_strategist() -> None:
    llm = FakeLLM({Intent: Intent(kind="analyze", company="tcs"), Sentiment: NEUTRAL_NEWS})
    context = make_context(llm, FakeMarketData(technical_error=True))
    result = await _say(build_graph(memory_checkpointer()), context, "analyze tcs")
    assert "couldn't load its price data" in result["messages"][-1].text
    assert StrategyDraft not in [schema for schema, _ in llm.calls]


async def test_guardrail_overrides_a_buy_on_critical_risk() -> None:
    falling_knife = make_snapshot(
        price=80.0,
        sma_50=95.0,
        sma_200=100.0,
        macd=-1.0,
        macd_signal=0.5,
        atr_14=4.0,
        atr_pct=5.0,
        rsi_14=25.0,
    )
    gloomy = Sentiment(score=-0.8, label="negative", themes=["probe"], summary="Bad.")
    llm = FakeLLM(
        {Intent: Intent(kind="analyze", company="tcs"), Sentiment: gloomy, StrategyDraft: BUY_DRAFT}
    )
    # 5 flags: downtrend, bearish momentum, high volatility, high leverage, negative news.
    context = make_context(llm, FakeMarketData(technical=falling_knife, debt_to_equity=3.0))

    result = await _say(build_graph(memory_checkpointer()), context, "analyze tcs")
    report = result["report"]
    assert report.risk.level == RiskLevel.CRITICAL
    assert report.risk.position_size_pct == 0.0
    # The LLM said BUY; policy turned it into HOLD and explained why.
    assert report.recommendation == Recommendation.HOLD
    assert report.guardrail_notes == ["BUY downgraded to HOLD: risk level is CRITICAL."]


async def test_greeting_and_orphan_follow_up_get_help() -> None:
    llm = FakeLLM({Intent: [Intent(kind="other"), Intent(kind="follow_up")]})
    graph, context = build_graph(memory_checkpointer()), make_context(llm)
    assert "I analyse listed stocks" in (await _say(graph, context, "hi"))["messages"][-1].text
    # A follow-up with no previous analysis has nothing to follow, so we show help.
    assert "I analyse listed stocks" in (await _say(graph, context, "why?"))["messages"][-1].text


async def test_conversations_are_isolated_by_thread() -> None:
    llm = FakeLLM(
        {
            Intent: [Intent(kind="analyze", company="tcs"), Intent(kind="follow_up")],
            Sentiment: NEUTRAL_NEWS,
            StrategyDraft: BUY_DRAFT,
        }
    )
    graph, context = build_graph(memory_checkpointer()), make_context(llm)
    await _say(graph, context, "analyze tcs", thread="alice")
    other = await _say(graph, context, "why?", thread="bob")
    assert other.get("report") is None  # Bob never saw Alice's analysis
