"""Composition root: the one place where real dependencies are created and connected.

The bot, the eval runner and the tests all build the agent through here (tests pass
fakes instead), so the wiring is defined exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain.agents import create_agent
from langfuse import Langfuse
from langgraph.checkpoint.base import BaseCheckpointSaver

from eq_analyst.agent.context import AgentContext
from eq_analyst.agent.graph import AnalystGraph, build_graph
from eq_analyst.agent.llm import build_chat_model
from eq_analyst.agent.market_data import McpMarketData, build_mcp_client
from eq_analyst.agent.prompts import PromptRegistry
from eq_analyst.core.config import Settings


@dataclass(frozen=True)
class AgentRuntime:
    graph: AnalystGraph
    context: AgentContext


async def build_agent_runtime(
    settings: Settings,
    checkpointer: BaseCheckpointSaver[str] | None,
    langfuse: Langfuse | None,
) -> AgentRuntime:
    llm = build_chat_model(settings)
    mcp_client = build_mcp_client(settings.mcp_server_url)

    # Load the MCP tools as LangChain tools so the follow-up agent's LLM can call them.
    # This also fails fast at startup if the MCP server is unreachable.
    tools = await mcp_client.get_tools()
    follow_up_agent = create_agent(model=llm, tools=tools, name="follow_up_agent")

    context = AgentContext(
        llm=llm,
        market_data=McpMarketData(mcp_client),
        prompts=PromptRegistry(langfuse, label=settings.prompt_label),
        follow_up_agent=follow_up_agent,
        preferred_exchanges=tuple(settings.preferred_exchanges),
    )
    return AgentRuntime(graph=build_graph(checkpointer), context=context)
