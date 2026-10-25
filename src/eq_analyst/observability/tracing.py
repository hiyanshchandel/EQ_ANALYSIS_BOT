"""Langfuse setup.

Langfuse records every LLM call, tool call and graph step as a *trace* (one per user
message), so we can see latency, token cost and exactly what the model saw. It is
optional: without keys the app runs normally and simply isn't traced.
"""

from __future__ import annotations

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from eq_analyst.core.config import Settings


def init_langfuse(settings: Settings) -> Langfuse | None:
    """Create the process-wide Langfuse client, or return None if not configured."""
    if not settings.langfuse_enabled or settings.langfuse_secret_key is None:
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_environment,
    )


def langchain_handler(settings: Settings) -> CallbackHandler:
    """A LangChain callback that turns graph/LLM/tool events into Langfuse spans.

    It attaches to whatever Langfuse span is currently active, so the graph's steps
    nest under the per-message trace we open in the service layer.
    """
    return CallbackHandler(public_key=settings.langfuse_public_key)
