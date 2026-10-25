"""Wire the nodes into a LangGraph state machine.

START -> intake --(route_intent)--> resolve_ticker | follow_up | reply_help
resolve_ticker --(route_after_resolve)--> [technical, fundamental, sentiment] (parallel)
                                      \\-> reply_not_found
[all three analysts] -> risk_manager --(route_after_risk)--> strategist | reply_data_unavailable
strategist -> guardrail -> END
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from eq_analyst.agent.context import AgentContext
from eq_analyst.agent.nodes.analysts import (
    fundamental_analyst,
    sentiment_analyst,
    technical_analyst,
)
from eq_analyst.agent.nodes.follow_up import follow_up
from eq_analyst.agent.nodes.intake import intake, route_intent
from eq_analyst.agent.nodes.replies import reply_data_unavailable, reply_help, reply_not_found
from eq_analyst.agent.nodes.resolve import ANALYSTS, resolve_ticker, route_after_resolve
from eq_analyst.agent.nodes.risk import risk_manager, route_after_risk
from eq_analyst.agent.nodes.strategist import guardrail, strategist
from eq_analyst.agent.state import AnalystState

# The compiled graph's full generic type: (state, context, input, output).
AnalystGraph = CompiledStateGraph[AnalystState, AgentContext, AnalystState, AnalystState]


def build_graph(checkpointer: BaseCheckpointSaver[str] | None = None) -> AnalystGraph:
    builder = StateGraph(AnalystState, context_schema=AgentContext)

    builder.add_node("intake", intake)
    builder.add_node("resolve_ticker", resolve_ticker)
    builder.add_node("technical_analyst", technical_analyst)
    builder.add_node("fundamental_analyst", fundamental_analyst)
    builder.add_node("sentiment_analyst", sentiment_analyst)
    builder.add_node("risk_manager", risk_manager)
    builder.add_node("strategist", strategist)
    builder.add_node("guardrail", guardrail)
    builder.add_node("follow_up", follow_up)
    builder.add_node("reply_help", reply_help)
    builder.add_node("reply_not_found", reply_not_found)
    builder.add_node("reply_data_unavailable", reply_data_unavailable)

    builder.add_edge(START, "intake")
    builder.add_conditional_edges(
        "intake", route_intent, ["resolve_ticker", "follow_up", "reply_help"]
    )
    builder.add_conditional_edges(
        "resolve_ticker", route_after_resolve, [*ANALYSTS, "reply_not_found"]
    )
    # Fan-in: risk_manager waits until all three parallel analysts have finished.
    builder.add_edge(ANALYSTS, "risk_manager")
    builder.add_conditional_edges(
        "risk_manager", route_after_risk, ["strategist", "reply_data_unavailable"]
    )
    builder.add_edge("strategist", "guardrail")

    for terminal in (
        "guardrail",
        "follow_up",
        "reply_help",
        "reply_not_found",
        "reply_data_unavailable",
    ):
        builder.add_edge(terminal, END)

    return builder.compile(checkpointer=checkpointer, name="equity-analyst")
