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
                 target=110.0, sharpe=1.0):
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
