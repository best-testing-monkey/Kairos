"""Tests for asset_class_for() and resolve_disabled_strategies() in
kairos_strategies.py: pure logic, no network/DB/model access.
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from kairos_strategies import (
    asset_class_for,
    resolve_disabled_strategies,
    _DISABLED_BY_PROFILE,
    _DISABLED_BY_CLASS,
)


class TestAssetClassFor:
    def test_crypto(self):
        assert asset_class_for(["BTC-USD", "ETH-USD", "SOL-USD"]) == "crypto"

    def test_fx(self):
        assert asset_class_for(["EURUSD=X", "GBPUSD=X", "USDCHF=X"]) == "fx"

    def test_commodity_futures(self):
        assert asset_class_for(["CL=F", "USO"]) == "commodity"

    def test_commodity_etfs(self):
        assert asset_class_for(["GLD", "SLV", "GDX"]) == "commodity"

    def test_mixed_commodity_group(self):
        # COPX, GC=F, GDX, GLD - all commodity-classified symbols
        assert asset_class_for(["COPX", "GC=F", "GDX", "GLD"]) == "commodity"

    def test_equity(self):
        assert asset_class_for(["AAPL", "MSFT", "SPY"]) == "equity"

    def test_mixed_no_majority(self):
        # 2 crypto, 2 equity: no strict majority -> mixed
        assert asset_class_for(["BTC-USD", "ETH-USD", "AAPL", "MSFT"]) == "mixed"

    def test_empty_list_is_mixed(self):
        assert asset_class_for([]) == "mixed"

    def test_majority_wins(self):
        # 3 equity, 1 crypto -> majority equity
        assert asset_class_for(["AAPL", "MSFT", "GOOG", "BTC-USD"]) == "equity"


class TestResolveDisabledStrategies:
    def test_exact_profile_wins_over_class(self):
        # BTC-USD,ETH-USD,SOL-USD has both an exact profile and would
        # otherwise fall into the crypto class bucket - exact must win.
        assets = ["BTC-USD", "ETH-USD", "SOL-USD"]
        result = resolve_disabled_strategies("1d", assets)
        assert result == _DISABLED_BY_PROFILE[("1d", "BTC-USD,ETH-USD,SOL-USD")]

    def test_exact_profile_independent_of_asset_order(self):
        assets = ["SOL-USD", "BTC-USD", "ETH-USD"]
        result = resolve_disabled_strategies("1d", assets)
        assert result == _DISABLED_BY_PROFILE[("1d", "BTC-USD,ETH-USD,SOL-USD")]

    def test_class_fallback_when_no_exact_profile(self):
        # No exact profile for this crypto group, should fall back to
        # the (1d, crypto) class bucket.
        assets = ["ADA-USD", "DOGE-USD", "DOT-USD", "LINK-USD"]
        result = resolve_disabled_strategies("1d", assets)
        assert result == _DISABLED_BY_CLASS[("1d", "crypto")]
        assert "volume_fade" in result

    def test_class_fallback_for_equity(self):
        assets = ["AAPL", "MSFT", "GOOG"]
        result = resolve_disabled_strategies("1d", assets)
        assert result == _DISABLED_BY_CLASS[("1d", "equity")]

    def test_empty_fallback_for_unknown_interval(self):
        assets = ["BTC-USD", "ETH-USD"]
        result = resolve_disabled_strategies("15m", assets)
        assert result == set()

    def test_empty_fallback_for_mixed_class_without_bucket(self):
        assets = ["BTC-USD", "AAPL"]
        assert asset_class_for(assets) == "mixed"
        result = resolve_disabled_strategies("1d", assets)
        assert result == set()
