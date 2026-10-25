"""Conversation persistence (LangGraph checkpointer backed by SQLite).

After every graph step LangGraph saves the state under the thread id. Restart the bot
and conversations survive. For multiple bot replicas you would swap in the Postgres
checkpointer; nothing else changes.

Security: checkpoints are deserialized on load, so we allow-list exactly which of our
classes may be reconstructed instead of trusting whatever is in the database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from eq_analyst.agent.schemas import Intent
from eq_analyst.domain.models import CHECKPOINT_TYPES


def checkpoint_serializer() -> JsonPlusSerializer:
    allowed = [(cls.__module__, cls.__name__) for cls in (*CHECKPOINT_TYPES, Intent)]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def memory_checkpointer() -> InMemorySaver:
    """Non-persistent checkpointer for tests and evals, with the same allow-list."""
    return InMemorySaver(serde=checkpoint_serializer())


@asynccontextmanager
async def sqlite_checkpointer(path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path) as connection:
        saver = AsyncSqliteSaver(connection, serde=checkpoint_serializer())
        await saver.setup()  # creates the tables on first run
        yield saver
