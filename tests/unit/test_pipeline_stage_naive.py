"""`--stage naive` must be reachable from the pipeline CLI.

run_stage_naive() existed since the naive mode was added but was never wired
into argparse's --stage choices or the dispatch, so the only way to reach it
was a direct import (which is what scripts/run_oracle_dedup.py does). Anyone
looking for the mode where every other stage lives would not find it.

These pin the wiring, not run_stage_naive itself -- that is exercised by every
naive sweep.
"""
import sqlite3
import tempfile
import os

import pytest

import kairos_pipeline as kp


@pytest.fixture
def isolated_db(monkeypatch):
    """Point the module's DB at a throwaway file so a CLI test never touches
    the real pipeline_results.db."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setattr(kp, "DB_PATH", path)
    yield path
    os.unlink(path)


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(kp, "run_stage_naive",
                        lambda conn, assets, **kw: calls.append(("naive", assets, kw)) or 1)
    monkeypatch.setattr(kp, "run_stage_oracle",
                        lambda conn, assets, **kw: calls.append(("oracle", assets, kw)) or 1)
    return calls


def test_naive_is_an_accepted_stage(isolated_db, recorded):
    kp.main(["--stage", "naive", "--assets", "AAPL", "MSFT"])
    assert recorded[0][0] == "naive"
    assert recorded[0][1] == ["AAPL", "MSFT"]


def test_naive_threads_its_run_parameters(isolated_db, recorded):
    kp.main(["--stage", "naive", "--assets", "AAPL",
             "--interval", "1h", "--backtest_period", "3m", "--pred_samples", "50"])
    _, _, kw = recorded[0]
    assert kw["interval"] == "1h"
    assert kw["backtest_period"] == "3m"
    assert kw["pred_samples"] == 50


def test_naive_never_receives_the_disable_gate_threshold(isolated_db, recorded):
    """Naive results deliberately never reach the production disable gate, so
    there is no threshold to apply -- forwarding one would imply otherwise."""
    kp.main(["--stage", "naive", "--assets", "AAPL"])
    assert "disable_min_signals" not in recorded[0][2]


def test_disable_min_signals_is_still_rejected_for_naive(isolated_db, recorded):
    """Pre-existing guard. Pinned so nobody widens it to include naive on the
    assumption it was an oversight."""
    with pytest.raises(SystemExit):
        kp.main(["--stage", "naive", "--assets", "AAPL", "--disable_min_signals", "7"])


def test_oracle_dispatch_is_unchanged(isolated_db, recorded):
    """naive shares oracle's asset-resolution branch; oracle must be untouched."""
    kp.main(["--stage", "oracle", "--assets", "AAPL", "--disable_min_signals", "7"])
    name, assets, kw = recorded[0]
    assert name == "oracle"
    assert kw["disable_min_signals"] == 7


def test_naive_requires_assets_and_says_so_by_name(isolated_db, recorded):
    with pytest.raises(SystemExit, match="--stage naive requires"):
        kp.main(["--stage", "naive"])
    assert recorded == []
