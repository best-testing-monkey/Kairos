import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import kairos_daily_signals as daily  # noqa: E402


REPORT_SAMPLE = """
# Kairos Signals Report

## Portfolio Allocation

Selected 3 of 12 signals

| Ticker       | Dir   | Strategy        | Entry | Stop | Target | EV net | Score | Alloc | Model        | Leverage | Margin % |
|--------------|-------|-----------------|-------|------|--------|--------|-------|-------|--------------|----------|----------|
| BTC-USD      | Long  | Momentum        |   100 |   90 |    120 | 1.20%  |    85 | 10.0% | kronos-small | 10.0x    | 1.0%     |
| ETH-USD      | Short | MeanReversion   |   200 |  220 |    160 | 0.90%  |    70 | 5.0%  | kronos-small | 5.0x     | 1.0%     |

## Failures

- AAPL-USD: fetch timeout
- TSLA-USD: prediction error

## Summary

Done.
"""

# Same as above but generated with the default --max-leverage 1.0 (Leverage/
# Margin % columns blank), matching what write_md_section() actually emits.
REPORT_SAMPLE_UNLEVERAGED = """
## Portfolio Allocation

Selected 1 of 5 signals

| Ticker       | Dir   | Strategy        | Entry | Stop | Target | EV net | Score | Alloc | Model        | Leverage | Margin % |
|--------------|-------|-----------------|-------|------|--------|--------|-------|-------|--------------|----------|----------|
| BTC-USD      | Long  | Momentum        |   100 |   90 |    120 | 1.20%  |    85 | 10.0% | kronos-small |          |          |
"""


def test_parse_selected_signals() -> None:
    assert daily.parse_selected_signals(REPORT_SAMPLE) == (3, 12)


def test_parse_allocation_rows() -> None:
    rows = daily.parse_allocation_rows(REPORT_SAMPLE)
    assert len(rows) == 2
    assert rows[0] == {
        "ticker": "BTC-USD",
        "direction": "Long",
        "entry": "100",
        "stop": "90",
        "target": "120",
        "ev_net": "1.20%",
        "alloc": "10.0%",
        "leverage": "10.0x",
        "margin_pct": "1.0%",
    }
    assert rows[1]["ticker"] == "ETH-USD"


def test_parse_failure_count() -> None:
    assert daily.parse_failure_count(REPORT_SAMPLE) == 2


def test_format_allocation_row_leveraged_shows_margin_and_ev_on_margin() -> None:
    rows = daily.parse_allocation_rows(REPORT_SAMPLE)
    line = daily.format_allocation_row(rows[0])
    # ev_net 1.20% * leverage 10.0x = 12% expected return on margin posted.
    assert line == "1.0% margin 10.0x leverage BTC-USD Long @ 100 TP 120 SL 90 (ev 12%)"


def test_format_allocation_row_unleveraged_falls_back() -> None:
    rows = daily.parse_allocation_rows(REPORT_SAMPLE_UNLEVERAGED)
    line = daily.format_allocation_row(rows[0])
    assert line == "BTC-USD Long @ 100 TP 120 SL 90 (ev 1.20%) -- 10.0% alloc"


def test_build_success_message_includes_allocations_and_failures() -> None:
    report_path = Path("results/kairos_signals_20260723.md")
    message = daily.build_success_message(REPORT_SAMPLE, report_path, 3, 12, 2)
    assert "3 selected of 12 candidates" in message
    assert "BTC-USD Long @ 100 TP 120 SL 90 (ev 12%)" in message
    assert "2 fetch/prediction failures" in message


def test_build_success_message_no_signals() -> None:
    report_path = Path("results/kairos_signals_20260723.md")
    sample = """
## Portfolio Allocation

Selected 0 of 5 signals
"""
    message = daily.build_success_message(sample, report_path, 0, 5, 0)
    assert "0 selected of 5 candidates" in message
    assert "Top allocations" not in message
    assert "failures" not in message
