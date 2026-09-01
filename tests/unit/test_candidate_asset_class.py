"""Candidate.asset_class must be the per-INSTRUMENT 3-way class.

Pins the fix for the taxonomy collision: viability_report.asset_class (which
feeds stats_row) is asset_class_for's 5-way GROUP-majority label, so it can
carry "commodity"/"fx"/"mixed" and can never match strategy_class_stats'
"fx_commodity". A candidate row is one ticker, so it is classified from the
ticker instead.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from allocation import fetch_signals
from signal_selection import parse_signal_selection, rule_matches


def _rows(*symbols):
    stats, advice = [], []
    for sym in symbols:
        stats.append(dict(strategy="s", symbol=sym, direction="LONG", entry=100.0,
                          stop=95.0, target=110.0, expected_value=2.0,
                          base_win_rate=0.6, backtest_period="6m",
                          base_sharpe=1.0, size=0.1, model="base",
                          # deliberately wrong group-majority label:
                          asset_class="equity"))
        advice.append(dict(base_signals=80))
    return stats, advice


def test_asset_class_comes_from_the_ticker_not_the_group_label():
    cands = fetch_signals(*_rows("BTC-USD", "CL=F", "AAPL", "GLD"))
    assert {c.ticker: c.asset_class for c in cands} == {
        "BTC-USD": "crypto", "CL=F": "fx_commodity",
        "AAPL": "equity", "GLD": "fx_commodity",
    }


def test_selection_rule_can_filter_on_the_stats_taxonomy():
    """'fx_commodity' matched nothing before this fix."""
    cands = fetch_signals(*_rows("BTC-USD", "CL=F", "AAPL", "GLD"))
    rule = parse_signal_selection("'Asset Class' == 'fx_commodity', TOP 5")
    assert {c.ticker for c in cands
            if rule_matches(rule, c, {}, {})[0]} == {"CL=F", "GLD"}
