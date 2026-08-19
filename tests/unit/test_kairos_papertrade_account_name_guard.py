"""Validation test for kairos_papertrade._refuse_duplicate_account_name.

Requires a live `phantom` install (the `phantom-ledger` package) but no GPU/model
download and no network access -- same pattern as
test_kairos_papertrade_remove_open_positions.py.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

ph = pytest.importorskip("phantom", reason="phantom_ledger not installed")

from kairos_papertrade import _refuse_duplicate_account_name  # noqa: E402


@pytest.fixture
def client(tmp_path):
    os.environ["PHANTOM_DATA"] = str(tmp_path)
    c = ph.Phantom(data_dir=str(tmp_path))
    import phantom.profiles as _profiles_pkg
    profile_path = os.path.join(os.path.dirname(_profiles_pkg.__file__), "ibkr.json")
    c.brokers.load(profile_path)
    try:
        yield c
    finally:
        c._conn.close()


def test_allows_fresh_name(client):
    _refuse_duplicate_account_name(client, "kairos_test_fresh")  # no raise


def test_refuses_existing_name(client):
    client.accounts.create(
        name="kairos_test_dupe", account_type="algorithm", broker="IBKR",
        capital=200.0, currency="EUR", algorithm_id="kairos_papertrade",
    )
    with pytest.raises(RuntimeError, match="already exists"):
        _refuse_duplicate_account_name(client, "kairos_test_dupe")
