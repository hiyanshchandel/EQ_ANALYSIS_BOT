from eq_analyst.domain.models import TickerMatch
from eq_analyst.domain.ticker_selection import pick_best_match


def _match(symbol: str, exchange: str, quote_type: str = "EQUITY") -> TickerMatch:
    return TickerMatch(symbol=symbol, name=symbol, exchange=exchange, quote_type=quote_type)


PREFERRED = ["NSI", "BSE", "NMS"]


def test_prefers_nse_over_foreign_listing() -> None:
    matches = [_match("INFY", "NYQ"), _match("INFY.BO", "BSE"), _match("INFY.NS", "NSI")]
    assert pick_best_match("infosys", matches, PREFERRED).symbol == "INFY.NS"  # type: ignore[union-attr]


def test_exact_symbol_wins() -> None:
    matches = [_match("TCS.NS", "NSI"), _match("TCS.BO", "BSE")]
    assert pick_best_match("tcs.bo", matches, PREFERRED).symbol == "TCS.BO"  # type: ignore[union-attr]


def test_ignores_funds_and_indices() -> None:
    matches = [_match("^NSEI", "NSI", "INDEX"), _match("NIFTYBEES.NS", "NSI", "ETF")]
    assert pick_best_match("nifty", matches, PREFERRED) is None


def test_unknown_exchanges_keep_search_order() -> None:
    matches = [_match("A.X", "XXX"), _match("B.Y", "YYY")]
    assert pick_best_match("a", matches, PREFERRED).symbol == "A.X"  # type: ignore[union-attr]


def test_no_matches() -> None:
    assert pick_best_match("nothing", [], PREFERRED) is None
