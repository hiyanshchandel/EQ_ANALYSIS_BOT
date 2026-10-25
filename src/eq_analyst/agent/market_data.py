"""How the agent gets market data: by calling the MCP server.

`MarketDataPort` is an interface (a Protocol). Nodes depend on the interface, not on
MCP, so tests can pass an in-memory fake and the graph runs without any network.
`McpMarketData` is the real implementation.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel

from eq_analyst.domain.models import Fundamentals, NewsFeed, TechnicalSnapshot, TickerSearchResult

M = TypeVar("M", bound=BaseModel)

MCP_SERVER_NAME = "market_data"


class MarketDataError(Exception):
    """The MCP tool ran but reported an error (unknown symbol, no data...)."""


class MarketDataPort(Protocol):
    async def search_companies(self, query: str) -> TickerSearchResult: ...
    async def get_technical_snapshot(self, symbol: str) -> TechnicalSnapshot: ...
    async def get_fundamentals(self, symbol: str) -> Fundamentals: ...
    async def get_news(self, symbol: str, limit: int = 8) -> NewsFeed: ...


def build_mcp_client(server_url: str) -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {MCP_SERVER_NAME: {"transport": "streamable_http", "url": server_url}}
    )


class McpMarketData:
    """Calls MCP tools directly (no LLM involved) and validates the typed results."""

    def __init__(self, client: MultiServerMCPClient) -> None:
        self._client = client

    async def _call(self, tool: str, arguments: dict[str, Any], model: type[M]) -> M:
        async with self._client.session(MCP_SERVER_NAME) as session:
            result = await session.call_tool(tool, arguments)
        if result.isError:
            message = " ".join(getattr(block, "text", "") for block in result.content)
            raise MarketDataError(message or f"MCP tool '{tool}' failed")
        if result.structuredContent is None:
            raise MarketDataError(f"MCP tool '{tool}' returned no structured content")
        # The server's output schema should match ours; validation catches drift.
        return model.model_validate(result.structuredContent)

    async def search_companies(self, query: str) -> TickerSearchResult:
        return await self._call("search_companies", {"query": query}, TickerSearchResult)

    async def get_technical_snapshot(self, symbol: str) -> TechnicalSnapshot:
        return await self._call("get_technical_snapshot", {"symbol": symbol}, TechnicalSnapshot)

    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        return await self._call("get_fundamentals", {"symbol": symbol}, Fundamentals)

    async def get_news(self, symbol: str, limit: int = 8) -> NewsFeed:
        return await self._call("get_news", {"symbol": symbol, "limit": limit}, NewsFeed)
