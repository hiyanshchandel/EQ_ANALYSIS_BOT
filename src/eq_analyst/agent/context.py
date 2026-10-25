"""Dependencies the graph's nodes need, injected at run time.

LangGraph's *runtime context* is dependency injection for graphs: we pass an
`AgentContext` to `graph.ainvoke(..., context=ctx)` and every node receives it as
`runtime.context`. Nodes never build their own LLM or network clients, so tests can
inject fakes and production can inject real ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from eq_analyst.agent.market_data import MarketDataPort
from eq_analyst.agent.prompts import PromptRegistry
from eq_analyst.domain.risk import RiskPolicy


@dataclass(frozen=True)
class AgentContext:
    llm: BaseChatModel
    market_data: MarketDataPort
    prompts: PromptRegistry
    follow_up_agent: Runnable[Any, Any]
    preferred_exchanges: tuple[str, ...]
    risk_policy: RiskPolicy = field(default_factory=RiskPolicy)
