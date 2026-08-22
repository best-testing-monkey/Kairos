"""test_signal_selection.py — Unit tests for the --signal-selection DSL parser.

Tests strategy/signal_selection.py's parse_signal_selection() grammar and
column registry in isolation, without allocation.py or GPU/DB involvement.
"""

import pytest

from signal_selection import (
    parse_signal_selection,
    evaluate_condition,
    resolve_column,
    column_kind,
    rule_matches,
    SignalSelectionRule,
    SelectionCondition,
    SignalSelectionError,
)


class TestValidConditions:
    """Single and multiple conditions, each supported operator, quote styles."""

    @pytest.mark.parametrize("op", [">", ">=", "<", "<=", "==", "!="])
    def test_numeric_condition_each_operator(self, op):
        rule = parse_signal_selection(f"'n' {op} 60")
        assert len(rule.conditions) == 1
        cond = rule.conditions[0]
        assert cond.column == "n"
        assert cond.op == op
        assert cond.value == 60.0

    def test_double_quotes_supported(self):
        rule = parse_signal_selection('"n" > 60')
        assert rule.conditions[0].column == "n"
        assert rule.conditions[0].value == 60.0

    def test_multiple_conditions(self):
        rule = parse_signal_selection("'n' > 60, 'Win raw' > 0.6")
        assert len(rule.conditions) == 2
        assert rule.conditions[0].column == "n"
        assert rule.conditions[0].op == ">"
        assert rule.conditions[0].value == 60.0
        assert rule.conditions[1].column == "Win raw"
        assert rule.conditions[1].op == ">"
        assert rule.conditions[1].value == 0.6

    def test_condition_column_case_insensitive(self):
        rule = parse_signal_selection("'WIN RAW' > 0.6")
        assert rule.conditions[0].column == "Win raw"

    def test_text_column_equality(self):
        rule = parse_signal_selection("'Dir' == 'long'")
        assert rule.conditions[0].column == "Dir"
        assert rule.conditions[0].op == "=="
        assert rule.conditions[0].value == "long"

    def test_text_column_not_equal_double_quoted_value(self):
        rule = parse_signal_selection('\'Strategy\' != "path_execution"')
        assert rule.conditions[0].value == "path_execution"

    def test_unquoted_string_value_falls_back_to_string(self):
        rule = parse_signal_selection("'Ticker' == AAPL")
        assert rule.conditions[0].value == "AAPL"

    def test_whitespace_around_clauses_stripped(self):
        rule = parse_signal_selection("  'n'   >   60  ,  'Win raw' > 0.6  ")
        assert len(rule.conditions) == 2


class TestOrderClause:
    def test_order_default_desc(self):
        rule = parse_signal_selection("ORDER 'EV raw %'")
        assert rule.order_by == ("EV raw %", True)

    def test_order_explicit_desc(self):
        rule = parse_signal_selection("ORDER 'EV raw %' DESC")
        assert rule.order_by == ("EV raw %", True)

    def test_order_explicit_asc(self):
        rule = parse_signal_selection("ORDER 'EV raw %' ASC")
        assert rule.order_by == ("EV raw %", False)

    def test_order_case_insensitive_keyword(self):
        rule = parse_signal_selection("order 'Score' asc")
        assert rule.order_by == ("Score", False)

    def test_no_order_clause_is_none(self):
        rule = parse_signal_selection("'n' > 60")
        assert rule.order_by is None


class TestTopClause:
    def test_top_parses_int(self):
        rule = parse_signal_selection("TOP 3")
        assert rule.top_n == 3

    def test_top_case_insensitive(self):
        rule = parse_signal_selection("top 5")
        assert rule.top_n == 5

    def test_no_top_clause_is_none(self):
        rule = parse_signal_selection("'n' > 60")
        assert rule.top_n is None


class TestUserExampleEndToEnd:
    def test_exact_user_example(self):
        spec = "'n' > 60, 'Win raw' > 0.6, ORDER 'EV raw %' DESC, TOP 3"
        rule = parse_signal_selection(spec)
        assert len(rule.conditions) == 2
        assert rule.conditions[0] == SelectionCondition(column="n", op=">", value=60.0)
        assert rule.conditions[1] == SelectionCondition(column="Win raw", op=">", value=0.6)
        assert rule.order_by == ("EV raw %", True)
        assert rule.top_n == 3
        assert rule.raw == spec


class TestErrors:
    """Every SignalSelectionError case listed in the design spec."""

    def test_unknown_column_in_condition(self):
        with pytest.raises(SignalSelectionError, match="Unknown column"):
            parse_signal_selection("'NotAColumn' > 1")

    def test_unknown_column_in_order(self):
        with pytest.raises(SignalSelectionError, match="Unknown column"):
            parse_signal_selection("ORDER 'NotAColumn'")

    def test_bad_operator(self):
        with pytest.raises(SignalSelectionError):
            parse_signal_selection("'n' = 60")

    def test_non_integer_top(self):
        with pytest.raises(SignalSelectionError, match="integer"):
            parse_signal_selection("TOP three")

    def test_non_integer_top_float(self):
        with pytest.raises(SignalSelectionError, match="integer"):
            parse_signal_selection("TOP 3.5")

    def test_numeric_operator_on_text_column(self):
        with pytest.raises(SignalSelectionError, match="text column"):
            parse_signal_selection("'Ticker' > 5")

    def test_numeric_operator_on_text_column_lt(self):
        with pytest.raises(SignalSelectionError, match="text column"):
            parse_signal_selection("'Dir' < 5")

    def test_order_on_text_column(self):
        with pytest.raises(SignalSelectionError, match="non-numeric"):
            parse_signal_selection("ORDER 'Ticker'")

    def test_duplicate_order(self):
        with pytest.raises(SignalSelectionError, match="Multiple ORDER"):
            parse_signal_selection("ORDER 'n' DESC, ORDER 'Score' ASC")

    def test_duplicate_top(self):
        with pytest.raises(SignalSelectionError, match="Multiple TOP"):
            parse_signal_selection("TOP 3, TOP 5")

    def test_unparseable_clause(self):
        with pytest.raises(SignalSelectionError):
            parse_signal_selection("this is not a valid clause")

    def test_malformed_condition_missing_quotes(self):
        with pytest.raises(SignalSelectionError):
            parse_signal_selection("n > 60")

    def test_is_value_error_subclass(self):
        assert issubclass(SignalSelectionError, ValueError)


class TestColumnRegistry:
    def test_text_columns_kind(self):
        for col in ("Ticker", "Cluster", "Strategy", "Dir"):
            assert column_kind(col) == "text"

    def test_numeric_columns_kind(self):
        for col in ("Entry", "Stop", "Target", "Risk %", "Reward %", "b", "n",
                    "Win raw", "Win shrunk", "EV raw %", "EV net %", "Kelly raw",
                    "Score", "Sharpe"):
            assert column_kind(col) == "numeric"

    def test_unknown_column_raises(self):
        with pytest.raises(SignalSelectionError):
            column_kind("Nope")


class _FakeCandidate:
    """Minimal stand-in matching the Candidate attributes resolvers read."""

    def __init__(self, ticker="AAA", strategy="s1", direction="long", n=100,
                 base_win_rate=0.6, ev_pct=3.0, entry=100.0, stop=95.0,
                 target=110.0, sharpe=1.0, model=None):
        self.ticker = ticker
        self.strategy = strategy
        self.direction = direction
        self.n = n
        self.base_win_rate = base_win_rate
        self.ev_pct = ev_pct
        self.entry = entry
        self.stop = stop
        self.target = target
        self.sharpe = sharpe
        self.model = model


class TestResolveAndEvaluate:
    def test_resolve_numeric_column_from_candidate(self):
        c = _FakeCandidate(n=77)
        assert resolve_column("n", c, {}, {}) == 77

    def test_resolve_derived_column(self):
        c = _FakeCandidate()
        derived = {"score": 1.23}
        assert resolve_column("Score", c, derived, {}) == 1.23

    def test_resolve_cluster_column(self):
        c = _FakeCandidate(ticker="AAPL")
        assert resolve_column("Cluster", c, {}, {"AAPL": "tech"}) == "tech"

    def test_resolve_cluster_column_default_empty(self):
        c = _FakeCandidate(ticker="ZZZ")
        assert resolve_column("Cluster", c, {}, {}) == ""

    def test_evaluate_condition_true(self):
        c = _FakeCandidate(n=100)
        cond = SelectionCondition(column="n", op=">", value=60.0)
        assert evaluate_condition(cond, c, {}, {}) is True

    def test_evaluate_condition_false(self):
        c = _FakeCandidate(n=10)
        cond = SelectionCondition(column="n", op=">", value=60.0)
        assert evaluate_condition(cond, c, {}, {}) is False

    def test_evaluate_condition_text_equality(self):
        c = _FakeCandidate(direction="long")
        cond = SelectionCondition(column="Dir", op="==", value="long")
        assert evaluate_condition(cond, c, {}, {}) is True


class TestModelColumn:
    """The Model column (added 2026-08-22): mirrors Candidate.model, "Base"
    or "Finetuned(<assets>)" -- not part of the original RFC schema, added
    specifically so a rule can distinguish a base vs finetuned signal for
    the same strategy+ticker, which nothing else in the registry could
    express."""

    def test_resolve_model_column(self):
        c = _FakeCandidate(model="Finetuned(ZEC-USD)")
        assert resolve_column("Model", c, {}, {}) == "Finetuned(ZEC-USD)"

    def test_resolve_model_column_base(self):
        c = _FakeCandidate(model="Base")
        assert resolve_column("Model", c, {}, {}) == "Base"

    def test_resolve_model_column_none_defaults_empty_string(self):
        c = _FakeCandidate(model=None)
        assert resolve_column("Model", c, {}, {}) == ""

    def test_model_condition_in_rule(self):
        rule = parse_signal_selection("'Model' == 'Finetuned(ZEC-USD)'")
        assert rule.conditions[0].column == "Model"
        c_match = _FakeCandidate(model="Finetuned(ZEC-USD)")
        c_no_match = _FakeCandidate(model="Base")
        assert evaluate_condition(rule.conditions[0], c_match, {}, {}) is True
        assert evaluate_condition(rule.conditions[0], c_no_match, {}, {}) is False

    def test_model_only_valid_with_equality_ops(self):
        with pytest.raises(SignalSelectionError):
            parse_signal_selection("'Model' > 'Base'")


class TestOrGroups:
    """Semicolon-separated OR-of-AND groups (added 2026-08-22) -- needed to
    express an exact whitelist of (Strategy, Ticker, Model) triples, which
    single-group AND-only conditions can't (no cross-column tuple matching,
    no OR)."""

    def test_single_group_unaffected_no_semicolon(self):
        """A spec with no semicolon must behave identically to before --
        one implicit group, condition_groups == [conditions]."""
        rule = parse_signal_selection("'n' > 60, 'Win raw' > 0.6")
        assert len(rule.condition_groups) == 1
        assert rule.condition_groups[0] == rule.conditions
        assert len(rule.conditions) == 2

    def test_two_groups_parsed_separately(self):
        rule = parse_signal_selection(
            "'Strategy' == 'a', 'Ticker' == 'X'; 'Strategy' == 'b', 'Ticker' == 'Y'"
        )
        assert len(rule.condition_groups) == 2
        assert [c.value for c in rule.condition_groups[0]] == ["a", "X"]
        assert [c.value for c in rule.condition_groups[1]] == ["b", "Y"]

    def test_order_and_top_apply_globally_regardless_of_which_group(self):
        """ORDER/TOP appearing in the SECOND group must still apply to the
        whole rule -- there is only ever one ORDER and one TOP total."""
        rule = parse_signal_selection(
            "'Strategy' == 'a'; 'Strategy' == 'b', ORDER 'Score' DESC, TOP 5"
        )
        assert rule.order_by == ("Score", True)
        assert rule.top_n == 5
        assert len(rule.condition_groups) == 2
        # Neither group's conditions include the ORDER/TOP clauses themselves.
        assert len(rule.condition_groups[0]) == 1
        assert len(rule.condition_groups[1]) == 1

    def test_duplicate_order_across_groups_rejected(self):
        with pytest.raises(SignalSelectionError):
            parse_signal_selection(
                "'Strategy' == 'a', ORDER 'Score' DESC; "
                "'Strategy' == 'b', ORDER 'Sharpe' ASC"
            )

    def test_empty_group_from_stray_semicolon_rejected(self):
        """A trailing (or doubled) semicolon must fail loudly, not silently
        produce a vacuously-true OR branch that matches everything."""
        with pytest.raises(SignalSelectionError):
            parse_signal_selection("'Strategy' == 'a';")

    def test_zero_condition_single_group_still_means_no_filtering(self):
        """Backward compat: a single-group spec with only ORDER/TOP (no
        semicolons, no real conditions) must keep meaning "no filtering" --
        NOT be rejected the way an empty group in a multi-group spec is."""
        rule = parse_signal_selection("ORDER 'Score' DESC, TOP 3")
        assert rule.condition_groups == [[]]
        assert rule.conditions == []


class TestRuleMatches:
    """rule_matches() -- the uniform AND/OR-of-AND evaluator allocation.py
    calls instead of looping rule.conditions directly."""

    def test_single_group_and_all_pass(self):
        rule = parse_signal_selection("'n' > 60, 'Win raw' > 0.6")
        c = _FakeCandidate(n=100, base_win_rate=0.8)
        matched, failure = rule_matches(rule, c, {}, {})
        assert matched is True
        assert failure is None

    def test_single_group_one_fails(self):
        rule = parse_signal_selection("'n' > 60, 'Win raw' > 0.6")
        c = _FakeCandidate(n=100, base_win_rate=0.1)
        matched, failure = rule_matches(rule, c, {}, {})
        assert matched is False
        assert failure.column == "Win raw"

    def test_or_group_matches_second_group(self):
        rule = parse_signal_selection(
            "'Strategy' == 'a', 'Ticker' == 'X'; 'Strategy' == 'b', 'Ticker' == 'Y'"
        )
        c = _FakeCandidate(strategy="b", ticker="Y")
        matched, failure = rule_matches(rule, c, {}, {})
        assert matched is True
        assert failure is None

    def test_or_group_matches_neither(self):
        rule = parse_signal_selection(
            "'Strategy' == 'a', 'Ticker' == 'X'; 'Strategy' == 'b', 'Ticker' == 'Y'"
        )
        c = _FakeCandidate(strategy="c", ticker="Z")
        matched, failure = rule_matches(rule, c, {}, {})
        assert matched is False
        assert failure is not None

    def test_exact_whitelist_use_case(self):
        """The real-world motivating case: match one of N explicit
        (Strategy, Ticker, Model) triples, distinguishing base vs finetuned
        for the same strategy+ticker where the whitelist wants only one."""
        rule = parse_signal_selection(
            "'Strategy' == 'path_execution', 'Ticker' == 'ZEC-USD', 'Model' == 'Base'; "
            "'Strategy' == 'support_confluence', 'Ticker' == 'AAPL', "
            "'Model' == 'Finetuned(AAPL)'"
        )
        # In the whitelist: path_execution/ZEC-USD/Base.
        c1 = _FakeCandidate(strategy="path_execution", ticker="ZEC-USD", model="Base")
        assert rule_matches(rule, c1, {}, {})[0] is True
        # Same strategy+ticker, but Finetuned -- NOT in the whitelist (only
        # Base is), and no other column could express this distinction.
        c2 = _FakeCandidate(strategy="path_execution", ticker="ZEC-USD", model="Finetuned(ZEC-USD)")
        assert rule_matches(rule, c2, {}, {})[0] is False
        # In the whitelist: support_confluence/AAPL/Finetuned(AAPL).
        c3 = _FakeCandidate(strategy="support_confluence", ticker="AAPL", model="Finetuned(AAPL)")
        assert rule_matches(rule, c3, {}, {})[0] is True
        # Same strategy+ticker, but Base -- NOT in the whitelist.
        c4 = _FakeCandidate(strategy="support_confluence", ticker="AAPL", model="Base")
        assert rule_matches(rule, c4, {}, {})[0] is False
        # Unrelated combo entirely.
        c5 = _FakeCandidate(strategy="high_low", ticker="SOL-USD", model="Base")
        assert rule_matches(rule, c5, {}, {})[0] is False
