"""The market-data MCP server.

MCP (Model Context Protocol) is a standard way to expose tools to AI applications.
Putting market data behind MCP means:

* the agent doesn't import yfinance; it calls tools over HTTP, so the data layer can
  be deployed, scaled and swapped (say, for a paid data vendor) independently;
* any MCP host (Claude Desktop, an IDE, another team's agent) can reuse these tools;
* each tool has a typed schema, so inputs are validated before our code runs.

Tool docstrings matter: they are what an LLM reads when deciding which tool to call.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from eq_analyst.core.config import Settings
from eq_analyst.domain.models import Fundamentals, NewsFeed, TechnicalSnapshot, TickerSearchResult
from eq_analyst.mcp_server.yahoo import YahooFinanceClient

# Tool arguments may be written by an LLM, so validate them strictly.
Symbol = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z0-9.\-^=&]{1,20}$",
        description="Yahoo Finance symbol, e.g. 'TCS.NS' (NSE), 'RELIANCE.BO' (BSE), 'AAPL'.",
    ),
]

INSTRUCTIONS = """Market data for listed equities (prices, technical indicators,
fundamentals, news). Symbols are Yahoo Finance symbols. If you only know a company
name, call search_companies first to find its symbol."""


def create_server(settings: Settings, client: YahooFinanceClient | None = None) -> FastMCP:
    data = client or YahooFinanceClient()
    mcp = FastMCP(
        name="market-data",
        instructions=INSTRUCTIONS,
        host=settings.mcp_host,
        port=settings.mcp_port,
        # Stateless + plain JSON responses: any replica can serve any request, so the
        # server can sit behind an ordinary load balancer.
        stateless_http=True,
        json_response=True,
    )

    # yfinance is blocking. `asyncio.to_thread` runs it in a worker thread so one slow
    # Yahoo call doesn't freeze the server for every other client.

    @mcp.tool()
    async def search_companies(
        query: Annotated[
            str, Field(min_length=1, max_length=80, description="Company name or symbol.")
        ],
        limit: Annotated[int, Field(ge=1, le=15)] = 8,
    ) -> TickerSearchResult:
        """Find listed securities matching a company name, e.g. 'Tata Motors' or 'Infosys'.
        Returns candidate symbols with their exchange and type (EQUITY, ETF, ...)."""
        matches = await asyncio.to_thread(data.search, query, limit)
        return TickerSearchResult(query=query, matches=matches)

    @mcp.tool()
    async def get_technical_snapshot(symbol: Symbol) -> TechnicalSnapshot:
        """Latest price and technical indicators for a symbol: moving averages (20/50/200
        day), RSI(14), MACD, Bollinger Bands, ATR volatility, volume ratio, 52-week range."""
        return await asyncio.to_thread(data.technical_snapshot, symbol)

    @mcp.tool()
    async def get_fundamentals(symbol: Symbol) -> Fundamentals:
        """Valuation and quality metrics for a symbol: P/E, P/B, debt/equity, ROE,
        profit margin, revenue growth, market cap, sector."""
        return await asyncio.to_thread(data.fundamentals, symbol)

    @mcp.tool()
    async def get_news(symbol: Symbol, limit: Annotated[int, Field(ge=1, le=15)] = 8) -> NewsFeed:
        """Recent news headlines and short summaries for a symbol, newest first."""
        items = await asyncio.to_thread(data.news, symbol, limit)
        return NewsFeed(symbol=symbol, items=items)

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> Response:
        """Liveness probe for Docker / Kubernetes."""
        return JSONResponse({"status": "ok"})

    return mcp
