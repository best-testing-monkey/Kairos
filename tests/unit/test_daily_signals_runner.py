import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import kairos_daily_signals as daily  # noqa: E402


REPORT_SAMPLE = """
# Kairos Signals Report

## Portfolio Allocation

Selected 3 of 12 signals

| Ticker       | Dir   | Strategy        | Entry | Stop | Target | EV net | Score | Alloc | Model        |
|--------------|-------|-----------------|-------|------|--------|--------|-------|-------|--------------|
| BTC-USD      | LONG  | Momentum        |   100 |   90 |    120 | 1.20   |    85 | 10%   | kronos-small |
| ETH-USD      | SHORT | MeanReversion   |   200 |  220 |    160 | 0.90   |    70 | 5%    | kronos-small |

## Failures

- AAPL-USD: fetch timeout
- TSLA-USD: prediction error

## Summary

Done.
"""


def test_parse_selected_signals() -> None:
    assert daily.parse_selected_signals(REPORT_SAMPLE) == (3, 12)


def test_parse_allocation_rows() -> None:
    rows = daily.parse_allocation_rows(REPORT_SAMPLE)
    assert len(rows) == 2
    assert "BTC-USD LONG (Momentum) @ 10%" in rows
    assert "ETH-USD SHORT (MeanReversion) @ 5%" in rows


def test_parse_failure_count() -> None:
    assert daily.parse_failure_count(REPORT_SAMPLE) == 2


def test_build_success_message_includes_allocations_and_failures() -> None:
    report_path = Path("results/kairos_signals_20260723.md")
    message = daily.build_success_message(REPORT_SAMPLE, report_path, 3, 12, 2)
    assert "3 selected of 12 candidates" in message
    assert "BTC-USD LONG" in message
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
