"""Thin, cached, retrying wrapper around yfinance.

yfinance scrapes Yahoo Finance, so calls are slow (~1s) and occasionally flaky. Two
standard defences:

* a TTL cache, so ten users asking about TCS in five minutes cost one Yahoo call;
* retries with exponential backoff for transient network errors.

All methods are synchronous (yfinance is); the MCP server runs them in a worker thread.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable
from datetime import UTC, datetime
from typing import Any, TypeVar

import pandas as pd
import yfinance as yf
from cachetools import TTLCache
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from eq_analyst.domain.indicators import build_snapshot
from eq_analyst.domain.models import Fundamentals, NewsItem, TechnicalSnapshot, TickerMatch

T = TypeVar("T")

MAX_SUMMARY_CHARS = 400


class MarketDataError(Exception):
    """A permanent failure (unknown symbol, no data). Not worth retrying."""


_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    retry=retry_if_not_exception_type(MarketDataError),
    reraise=True,
)


def _pct(value: Any) -> float | None:
    """Convert a fraction (0.183) to a rounded percentage (18.3)."""
    return None if value is None else round(float(value) * 100, 2)


def _parse_news_item(raw: dict[str, Any]) -> NewsItem | None:
    # Newer yfinance nests fields under "content"; older versions put them at the top level.
    content = raw.get("content") or raw
    title = content.get("title")
    if not title:
        return None

    provider = content.get("provider")
    publisher = provider.get("displayName") if isinstance(provider, dict) else raw.get("publisher")

    published_at: datetime | None = None
    if pub_date := content.get("pubDate"):
        published_at = datetime.fromisoformat(str(pub_date).replace("Z", "+00:00"))
    elif epoch := raw.get("providerPublishTime"):
        published_at = datetime.fromtimestamp(int(epoch), tz=UTC)

    canonical = content.get("canonicalUrl")
    url = canonical.get("url") if isinstance(canonical, dict) else raw.get("link")
    summary = content.get("summary") or content.get("description")

    return NewsItem(
        title=title,
        publisher=publisher,
        published_at=published_at,
        summary=summary[:MAX_SUMMARY_CHARS] if summary else None,
        url=url,
    )


class YahooFinanceClient:
    def __init__(self, cache_ttl_seconds: int = 300) -> None:
        self._cache: TTLCache[Hashable, Any] = TTLCache(maxsize=512, ttl=cache_ttl_seconds)
        # TTLCache is not thread-safe and our methods run in worker threads.
        self._lock = threading.Lock()

    def _cached(self, key: Hashable, loader: Callable[[], T]) -> T:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        value = loader()  # load outside the lock so slow calls don't block each other
        with self._lock:
            self._cache[key] = value
        return value

    # --- public API -----------------------------------------------------------

    def search(self, query: str, limit: int = 8) -> list[TickerMatch]:
        return self._cached(("search", query.lower(), limit), lambda: self._search(query, limit))

    def technical_snapshot(self, symbol: str) -> TechnicalSnapshot:
        return self._cached(("technical", symbol), lambda: self._technical_snapshot(symbol))

    def fundamentals(self, symbol: str) -> Fundamentals:
        return self._cached(("fundamentals", symbol), lambda: self._fundamentals(symbol))

    def news(self, symbol: str, limit: int = 8) -> list[NewsItem]:
        return self._cached(("news", symbol, limit), lambda: self._news(symbol, limit))

    # --- uncached loaders ------------------------------------------------------

    @_retry
    def _search(self, query: str, limit: int) -> list[TickerMatch]:
        quotes = yf.Search(
            query, max_results=limit, news_count=0, lists_count=0, recommended=0, include_cb=False
        ).quotes
        return [
            TickerMatch(
                symbol=q["symbol"],
                name=q.get("longname") or q.get("shortname") or q["symbol"],
                exchange=q.get("exchange", ""),
                exchange_display=q.get("exchDisp"),
                quote_type=q.get("quoteType", ""),
            )
            for q in quotes
            if q.get("symbol")
        ]

    @_retry
    def _technical_snapshot(self, symbol: str) -> TechnicalSnapshot:
        ticker = yf.Ticker(symbol)
        history: pd.DataFrame = ticker.history(period="1y", interval="1d", auto_adjust=True)
        if history.empty:
            raise MarketDataError(f"No price history found for symbol '{symbol}'.")
        currency = (ticker.history_metadata or {}).get("currency")
        try:
            return build_snapshot(symbol, history, currency)
        except ValueError as exc:  # e.g. a recent IPO with too little history
            raise MarketDataError(str(exc)) from exc

    @_retry
    def _fundamentals(self, symbol: str) -> Fundamentals:
        info: dict[str, Any] = yf.Ticker(symbol).info or {}
        if not info.get("quoteType"):
            raise MarketDataError(f"No fundamental data found for symbol '{symbol}'.")
        debt_to_equity = info.get("debtToEquity")
        return Fundamentals(
            symbol=symbol,
            name=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            currency=info.get("currency"),
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            pb_ratio=info.get("priceToBook"),
            # Yahoo reports debt/equity as a percentage (150 means 1.5x). v1 compared the
            # raw value with 2, which flagged almost every company as over-leveraged.
            debt_to_equity=None if debt_to_equity is None else round(debt_to_equity / 100, 2),
            roe_pct=_pct(info.get("returnOnEquity")),
            profit_margin_pct=_pct(info.get("profitMargins")),
            revenue_growth_pct=_pct(info.get("revenueGrowth")),
        )

    @_retry
    def _news(self, symbol: str, limit: int) -> list[NewsItem]:
        raw_items = yf.Ticker(symbol).get_news(count=limit) or []
        items = [_parse_news_item(raw) for raw in raw_items]
        return [item for item in items if item is not None][:limit]
