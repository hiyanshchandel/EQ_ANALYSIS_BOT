"""Pick the right listing when a user types a company name.

"Reliance" returns several Yahoo matches (NSE, BSE, foreign listings, funds). The choice
is a business rule, not a judgement call, so it lives in plain code: keep equities,
honour an exact symbol match, otherwise prefer exchanges in the configured order.
"""

from __future__ import annotations

from collections.abc import Sequence

from eq_analyst.domain.models import TickerMatch


def pick_best_match(
    query: str, matches: Sequence[TickerMatch], preferred_exchanges: Sequence[str]
) -> TickerMatch | None:
    equities = [m for m in matches if m.quote_type.upper() == "EQUITY"]
    if not equities:
        return None

    # 1. The user typed the symbol itself, e.g. "TCS.NS".
    normalized = query.strip().upper()
    for match in equities:
        if match.symbol.upper() == normalized:
            return match

    # 2. Otherwise prefer exchanges in priority order. Yahoo already ranks matches by
    #    relevance, so within an exchange we keep its order.
    rank = {code.upper(): position for position, code in enumerate(preferred_exchanges)}
    unranked = len(rank)
    best = min(
        enumerate(equities),
        key=lambda pair: (rank.get(pair[1].exchange.upper(), unranked), pair[0]),
    )
    return best[1]
