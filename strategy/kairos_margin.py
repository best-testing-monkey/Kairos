"""kairos_margin.py — Config-driven margin class loader and symbol classifier.

Pure module with no phantom, GPU, or network dependencies. Loads the IBKR-style
margin schedule from ``config/margin_ibkr.yaml`` and maps a ticker to its
margin class via explicit symbols, regex fallbacks, and a default class.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass(frozen=True)
class MarginClass:
    """Margin regime for a single asset class.

    Attributes:
        name: Config key for this class (e.g. ``fx_major``).
        initial_margin_pct: Initial margin requirement as a percentage of notional.
        maintenance_margin_pct: Maintenance margin requirement as a percentage.
        financing_spread_pct: Financing spread over the benchmark rate.
        enabled: Whether this class is active for classification.
        symbols: Optional explicit symbol set for this class.
        match: Optional regex pattern for this class; ``None`` marks the default
            fallback bucket.
    """

    name: str
    initial_margin_pct: float
    maintenance_margin_pct: float
    financing_spread_pct: float
    enabled: bool = True
    symbols: frozenset[str] = field(default_factory=frozenset)
    match: str | None = None


@dataclass(frozen=True)
class MarginConfig:
    """Loaded representation of ``config/margin_ibkr.yaml``.

    Attributes:
        base_currency: Account/reference currency.
        benchmark_annual_pct: Annual benchmark rate used for financing accrual.
        negative_balance_protection: Whether retail negative-balance protection applies.
        closeout_fraction: Equity / initial-margin liquidation trigger fraction.
        classes: Ordered mapping of class name -> ``MarginClass``.
        overrides: Per-symbol override dict mapping field name -> value.
        short_borrow_annual_pct: Short-borrow fee schedule with ``default`` and
            per-symbol ``overrides``.
    """

    base_currency: str
    benchmark_annual_pct: float
    negative_balance_protection: bool
    closeout_fraction: float
    classes: dict[str, MarginClass] = field(default_factory=dict)
    overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    short_borrow_annual_pct: dict[str, Any] = field(default_factory=dict)


def load_margin_config(path: str | Path) -> MarginConfig:
    """Load a YAML margin configuration into a ``MarginConfig``.

    Args:
        path: Filesystem path to the YAML config.

    Returns:
        ``MarginConfig`` with parsed classes and overrides.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the YAML is malformed or a required class field is missing.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Margin config {path} must contain a YAML mapping")

    classes: dict[str, MarginClass] = {}
    raw_classes = raw.get("classes", {})
    if not isinstance(raw_classes, dict):
        raise ValueError("'classes' must be a mapping")

    for name, spec in raw_classes.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Class spec for {name!r} must be a mapping")

        raw_symbols = spec.get("symbols", [])
        symbols = frozenset(raw_symbols) if isinstance(raw_symbols, list) else frozenset()

        classes[name] = MarginClass(
            name=name,
            initial_margin_pct=float(spec["initial_margin_pct"]),
            maintenance_margin_pct=float(spec["maintenance_margin_pct"]),
            financing_spread_pct=float(spec["financing_spread_pct"]),
            enabled=bool(spec.get("enabled", True)),
            symbols=symbols,
            match=spec.get("match"),
        )

    overrides = raw.get("overrides", {}) or {}
    if not isinstance(overrides, dict):
        raise ValueError("'overrides' must be a mapping")

    short_borrow = raw.get("short_borrow_annual_pct", {}) or {}
    if not isinstance(short_borrow, dict):
        raise ValueError("'short_borrow_annual_pct' must be a mapping")

    return MarginConfig(
        base_currency=str(raw.get("base_currency", "EUR")),
        benchmark_annual_pct=float(raw.get("benchmark_annual_pct", 0.0)),
        negative_balance_protection=bool(raw.get("negative_balance_protection", True)),
        closeout_fraction=float(raw.get("closeout_fraction", 0.5)),
        classes=classes,
        overrides=overrides,
        short_borrow_annual_pct=short_borrow,
    )


def classify_symbol(symbol: str, cfg: MarginConfig) -> MarginClass:
    """Return the ``MarginClass`` for ``symbol``.

    Matching order:
      1. Explicit ``symbols`` membership, in config class order.
      2. First matching ``match`` regex, in config class order.
      3. The class whose ``match`` is ``null`` (the default bucket).

    Disabled classes (``enabled: false``) are skipped, so a disabled
    ``crypto_cfd`` lets ``-USD`` tickers fall through to ``crypto_spot``.

    Per-symbol overrides apply after classification and win over the class
    defaults.

    Args:
        symbol: Ticker to classify.
        cfg: Loaded margin configuration.

    Returns:
        ``MarginClass`` for the symbol, with any per-symbol override applied.

    Raises:
        ValueError: If no enabled class matches and no default class exists.
    """
    default_class: MarginClass | None = None

    for cls in cfg.classes.values():
        if not cls.enabled:
            continue

        if cls.match == "" and not cls.symbols:
            # Treat empty string consistently with null: default bucket.
            default_class = cls
            continue

        # Explicit symbols list takes precedence for this class.
        if symbol in cls.symbols:
            return _apply_override(symbol, cls, cfg)

        # Regex match is checked only when no explicit symbols list matched.
        if cls.match is not None and re.search(cls.match, symbol):
            return _apply_override(symbol, cls, cfg)

        # Null match marks the default bucket.
        if cls.match is None and not cls.symbols:
            default_class = cls

    if default_class is not None:
        return _apply_override(symbol, default_class, cfg)

    raise ValueError(f"No enabled margin class matches {symbol!r}")


def _apply_override(symbol: str, cls: MarginClass, cfg: MarginConfig) -> MarginClass:
    """Return ``cls`` with any per-symbol override values applied."""
    override = cfg.overrides.get(symbol, {})
    if not override:
        return cls

    return MarginClass(
        name=cls.name,
        initial_margin_pct=float(override.get("initial_margin_pct", cls.initial_margin_pct)),
        maintenance_margin_pct=float(override.get("maintenance_margin_pct", cls.maintenance_margin_pct)),
        financing_spread_pct=float(override.get("financing_spread_pct", cls.financing_spread_pct)),
        enabled=cls.enabled,
        symbols=cls.symbols,
        match=cls.match,
    )
