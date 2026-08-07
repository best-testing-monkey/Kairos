"""tests/unit/test_kairos_margin.py — Unit tests for the margin classifier."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest  # noqa: E402

from kairos_margin import classify_symbol, load_margin_config, MarginClass, MarginConfig  # noqa: E402


@pytest.fixture
def cfg() -> MarginConfig:
    """Default IBKR-style margin config fixture."""
    return load_margin_config(
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "margin_ibkr.yaml")
    )


def test_load_margin_config_populates_top_level_fields(cfg: MarginConfig) -> None:
    assert cfg.base_currency == "EUR"
    assert cfg.benchmark_annual_pct == pytest.approx(3.15)
    assert cfg.negative_balance_protection is True
    assert cfg.closeout_fraction == pytest.approx(0.5)
    assert "fx_major" in cfg.classes
    assert "equity_cfd" in cfg.classes


def test_margin_class_exposes_required_fields(cfg: MarginConfig) -> None:
    fx = cfg.classes["fx_major"]
    assert fx.name == "fx_major"
    assert fx.initial_margin_pct == pytest.approx(3.33)
    assert fx.maintenance_margin_pct == pytest.approx(1.67)
    assert fx.financing_spread_pct == pytest.approx(1.5)


@pytest.mark.parametrize("symbol", ["EURUSD=X", "USDJPY=X"])
def test_fx_major_pairs_classified(symbol: str, cfg: MarginConfig) -> None:
    cls = classify_symbol(symbol, cfg)
    assert cls.name == "fx_major"
    assert cls.initial_margin_pct == pytest.approx(3.33)


@pytest.mark.parametrize("symbol", ["GC=F", "^GSPC", "SPY", "QQQ"])
def test_index_gold_major_classified(symbol: str, cfg: MarginConfig) -> None:
    cls = classify_symbol(symbol, cfg)
    assert cls.name == "index_gold_major"
    assert cls.initial_margin_pct == pytest.approx(5.0)


def test_commodity_other_futures_classified(cfg: MarginConfig) -> None:
    cls = classify_symbol("CL=F", cfg)
    assert cls.name == "commodity_other"
    assert cls.initial_margin_pct == pytest.approx(10.0)


def test_crypto_spot_when_cfd_disabled(cfg: MarginConfig) -> None:
    assert cfg.classes["crypto_cfd"].enabled is False
    cls = classify_symbol("BTC-USD", cfg)
    assert cls.name == "crypto_spot"
    assert cls.initial_margin_pct == pytest.approx(100.0)
    assert cls.maintenance_margin_pct == pytest.approx(0.0)


def test_equity_cfd_default_for_plain_ticker(cfg: MarginConfig) -> None:
    cls = classify_symbol("AAPL", cfg)
    assert cls.name == "equity_cfd"
    assert cls.initial_margin_pct == pytest.approx(20.0)


def test_per_symbol_override_wins(cfg: MarginConfig) -> None:
    cfg = MarginConfig(
        base_currency=cfg.base_currency,
        benchmark_annual_pct=cfg.benchmark_annual_pct,
        negative_balance_protection=cfg.negative_balance_protection,
        closeout_fraction=cfg.closeout_fraction,
        classes=cfg.classes,
        overrides={"AAPL": {"initial_margin_pct": 50.0}},
        short_borrow_annual_pct=cfg.short_borrow_annual_pct,
    )
    cls = classify_symbol("AAPL", cfg)
    assert cls.name == "equity_cfd"
    assert cls.initial_margin_pct == pytest.approx(50.0)
    assert cls.maintenance_margin_pct == pytest.approx(10.0)


def test_disabled_crypto_cfd_fallthrough_to_spot(cfg: MarginConfig) -> None:
    # When crypto_cfd is disabled, an explicit -USD ticker must not accidentally
    # match crypto_cfd even if it had a regex; it should fall through to crypto_spot.
    crypto_cfd = cfg.classes["crypto_cfd"]
    assert crypto_cfd.enabled is False

    enabled_spot = MarginClass(
        name="crypto_spot",
        initial_margin_pct=100.0,
        maintenance_margin_pct=0.0,
        financing_spread_pct=0.0,
        enabled=True,
        symbols=frozenset(),
        match=r"-USD$",
    )
    disabled_cfd = MarginClass(
        name="crypto_cfd",
        initial_margin_pct=50.0,
        maintenance_margin_pct=25.0,
        financing_spread_pct=2.5,
        enabled=False,
        symbols=frozenset(),
        match=r"-USD$",
    )
    cfg = MarginConfig(
        base_currency=cfg.base_currency,
        benchmark_annual_pct=cfg.benchmark_annual_pct,
        negative_balance_protection=cfg.negative_balance_protection,
        closeout_fraction=cfg.closeout_fraction,
        classes={"crypto_cfd": disabled_cfd, "crypto_spot": enabled_spot, "equity_cfd": cfg.classes["equity_cfd"]},
        overrides={},
        short_borrow_annual_pct=cfg.short_borrow_annual_pct,
    )

    cls = classify_symbol("BTC-USD", cfg)
    assert cls.name == "crypto_spot"


def test_explicit_symbols_take_precedence_over_regex(cfg: MarginConfig) -> None:
    # SPY is in index_gold_major symbols; it should not fall to equity_cfd default.
    cls = classify_symbol("SPY", cfg)
    assert cls.name == "index_gold_major"


def test_no_match_raises_when_no_default_exists() -> None:
    cfg = MarginConfig(
        base_currency="EUR",
        benchmark_annual_pct=3.15,
        negative_balance_protection=True,
        closeout_fraction=0.5,
        classes={
            "fx": MarginClass(
                name="fx",
                initial_margin_pct=3.33,
                maintenance_margin_pct=1.67,
                financing_spread_pct=1.5,
                enabled=True,
                symbols=frozenset(["EURUSD=X"]),
                match=None,
            )
        },
        overrides={},
        short_borrow_annual_pct={},
    )
    with pytest.raises(ValueError, match="No enabled margin class matches"):
        classify_symbol("AAPL", cfg)
