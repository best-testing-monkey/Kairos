"""Verify the 27 awesome-quant strategies are wired into StrategyRegistry.build_all().

QA gap: these classes existed and passed their own unit tests, but were never
appended in kairos_orchestrator.StrategyRegistry.build_all(), so they never ran
in a backtest. This test asserts every one of the 27 class names is reachable
from build_all()'s output, either directly or via a base_strategy wrapper
chain, and that all registered strategy `.name` values are unique.
"""
from kairos_orchestrator import StrategyRegistry, OrchestratorConfig

NEW_CLASS_NAMES = {
    "VarianceRiskPremiumStrategy",
    "VARLeadLagStrategy",
    "GrangerPairsStrategy",
    "MatrixProfileAnomalyStrategy",
    "GBMDirectionStrategy",
    "SocialMomentumStrategy",
    "CVDDivergenceStrategy",
    "MultiFactorRankStrategy",
    "PCAResidualReversalStrategy",
    "StochasticFilterStrategy",
    "ADXGateStrategy",
    "OBVConfirmationStrategy",
    "MTFConsensusStrategy",
    "ATRBracketStrategy",
    "GARCHFilterStrategy",
    "VolTargetSizerStrategy",
    "ARIMADisagreementStrategy",
    "SeasonalityFilterStrategy",
    "ChangepointGuardStrategy",
    "MetaLabelStrategy",
    "LPPLSGuardStrategy",
    "NewsSentimentFilterStrategy",
    "Institutional13FFilterStrategy",
    "EconCalendarGuardStrategy",
    "VolumeProfileLevelsStrategy",
    "TWAPExecutionStrategy",
    "ImplementationShortfallStrategy",
}


def _unwrap_class_names(strategies):
    """Collect the class name of every strategy plus every base_strategy it
    wraps, walking the wrapper chain to arbitrary depth."""
    found = set()
    stack = list(strategies)
    seen_ids = set()
    while stack:
        s = stack.pop()
        if id(s) in seen_ids:
            continue
        seen_ids.add(id(s))
        found.add(type(s).__name__)

        base = getattr(s, "base_strategy", None)
        if base is not None:
            stack.append(base)

        base_list = getattr(s, "base_strategies", None)
        if base_list:
            stack.extend(base_list)
    return found


def test_all_27_new_strategies_registered():
    config = OrchestratorConfig(disabled_strategies=set())
    strategies = StrategyRegistry.build_all(config)

    found_class_names = _unwrap_class_names(strategies)

    missing = NEW_CLASS_NAMES - found_class_names
    assert not missing, f"New strategies missing from build_all(): {sorted(missing)}"


def test_registered_strategy_names_are_unique():
    # Disable the two generic post-processing wraps (kurtosis_filter,
    # liquidity_filter) that intentionally collapse every strategy's outer
    # .name to a shared wrapper name ("kurtosis_filter"/"liquidity_filter").
    # With those off, each registered entry's own .name must be unique -
    # this is what catches cases like the two ADXGateStrategy registrations
    # (trend vs. reversion) which require distinct instance-level names.
    config = OrchestratorConfig(
        disabled_strategies=set(),
        kurtosis_action="none",
        min_volume_percentile=0.0,
    )
    strategies = StrategyRegistry.build_all(config)

    names = [s.name for s in strategies]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"Duplicate strategy names in registry: {sorted(duplicates)}"
