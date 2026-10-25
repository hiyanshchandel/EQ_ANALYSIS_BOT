"""Render results as Telegram HTML.

Telegram's HTML mode supports a few tags (<b>, <i>, <code>...). Any text that came from
a user, a website or an LLM is escaped first, otherwise a stray "<" breaks the message
(or injects markup).
"""

from __future__ import annotations

from html import escape

from eq_analyst.domain.models import AnalysisReport, Recommendation

DISCLAIMER = "Educational analysis, not financial advice."

BADGE = {
    Recommendation.BUY: "🟢",
    Recommendation.HOLD: "🟡",
    Recommendation.SELL: "🔴",
}

CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}


def money(value: float, currency: str | None) -> str:
    symbol = CURRENCY_SYMBOLS.get(currency or "", f"{currency} " if currency else "")
    return f"{symbol}{value:,.2f}"


def _bullets(items: list[str]) -> str:
    return "\n".join(f"• {escape(item)}" for item in items)


def format_report(report: AnalysisReport) -> str:
    t, r, cur = report.technical, report.risk, report.technical.currency
    name, symbol = escape(report.ticker.name), escape(report.ticker.symbol)
    exchange = escape(report.ticker.exchange_display or report.ticker.exchange)

    parts = [
        f"<b>{name}</b> ({symbol} · {exchange})",
        f"{BADGE[report.recommendation]} <b>{report.recommendation}</b> · "
        f"conviction {report.conviction} · {report.horizon} term",
        "",
        f"<b>Price</b> {money(t.price, cur)} ({t.change_1d_pct:+.2f}% today)",
        f"<b>Risk</b> {r.level} ({r.score}/{r.max_score}) · stop loss {money(r.stop_loss, cur)} "
        f"(−{r.stop_distance_pct}%) · max position {r.position_size_pct}%",
        "",
        f"<b>Thesis</b>\n{escape(report.thesis)}",
    ]
    if report.bull_points:
        parts += ["", f"<b>Bull case</b>\n{_bullets(report.bull_points)}"]
    if report.bear_points:
        parts += ["", f"<b>Bear case</b>\n{_bullets(report.bear_points)}"]
    if r.flags:
        parts += ["", f"<b>Risk flags</b>\n{_bullets([flag.description for flag in r.flags])}"]

    parts += [
        "",
        f"<b>Key levels</b> support {money(t.bb_lower, cur)} · "
        f"resistance {money(t.bb_upper, cur)} · "
        f"52w {money(t.low_52w, cur)}–{money(t.high_52w, cur)}",
    ]
    if report.sentiment:
        s = report.sentiment
        parts.append(
            f"<b>News</b> {s.label} ({s.score:+.2f}, {s.headline_count} headlines): "
            f"{escape(s.summary)}"
        )
    if report.fundamentals is None:
        parts.append("<i>Fundamental data was unavailable for this analysis.</i>")
    if report.guardrail_notes:
        parts += ["", f"⚠️ <b>Policy checks</b>\n{_bullets(report.guardrail_notes)}"]

    parts += ["", f"<i>Data as of {t.as_of:%d %b %Y}. {DISCLAIMER}</i>"]
    return "\n".join(parts)


def format_message(text: str) -> str:
    return escape(text)
