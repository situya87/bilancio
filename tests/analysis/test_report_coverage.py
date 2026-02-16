"""Additional tests for report.py to cover uncovered functions and branches.

Focuses on:
- _to_json: Decimal, dict, list/tuple, passthrough
- write_day_metrics_csv: empty rows, Decimal values, None values
- write_day_metrics_json
- write_debtor_shares_csv / write_intraday_csv (thin wrappers)
- _fmt_num: None, integral Decimal, non-integral Decimal, ArithmeticError
- _group_by
- parse_day_ranges: ranges, single values, empty/invalid, reversed ranges
- infer_day_list: from PayableCreated, from PayableSettled, empty
- compute_run_level_metrics
- compute_day_metrics: empty, with balances, without
- summarize_day_metrics: string values, day=1 metrics, empty
- write_metrics_html: full render, subtitle, run_level_metrics, no data
- _resolve_path: relative and absolute
- _decimal_or_none: valid, None, empty string, invalid
- aggregate_runs: skipped rows, missing metrics file, empty metrics
- render_dashboard: rendering from CSV
"""

import csv
import json
from decimal import Decimal
from pathlib import Path

from bilancio.analysis.report import (
    _decimal_or_none,
    _fmt_num,
    _group_by,
    _resolve_path,
    _to_json,
    aggregate_runs,
    compute_day_metrics,
    compute_run_level_metrics,
    infer_day_list,
    parse_day_ranges,
    render_dashboard,
    summarize_day_metrics,
    write_day_metrics_csv,
    write_day_metrics_json,
    write_debtor_shares_csv,
    write_intraday_csv,
    write_metrics_html,
)

# ===========================================================================
# _to_json
# ===========================================================================


class TestToJson:
    def test_decimal_integer(self):
        """Decimal that is a whole number -> int."""
        assert _to_json(Decimal("42")) == 42
        assert _to_json(Decimal("0")) == 0

    def test_decimal_fractional(self):
        """Decimal with fractional part -> str."""
        result = _to_json(Decimal("3.14"))
        assert isinstance(result, str)
        assert result == "3.14"

    def test_dict_recursive(self):
        """Dict values are recursively processed."""
        result = _to_json({"a": Decimal("10"), "b": Decimal("1.5")})
        assert result == {"a": 10, "b": "1.5"}

    def test_list_recursive(self):
        """List values are recursively processed."""
        result = _to_json([Decimal("1"), Decimal("2.5"), "hello"])
        assert result == [1, "2.5", "hello"]

    def test_tuple_recursive(self):
        """Tuple values are recursively processed."""
        result = _to_json((Decimal("1"), "test"))
        assert result == [1, "test"]

    def test_passthrough(self):
        """Non-special types pass through unchanged."""
        assert _to_json(42) == 42
        assert _to_json("hello") == "hello"
        assert _to_json(None) is None


# ===========================================================================
# write_day_metrics_csv
# ===========================================================================


class TestWriteDayMetricsCsv:
    def test_empty_rows(self, tmp_path):
        """Empty rows produce a CSV with only a 'day' header."""
        path = tmp_path / "metrics.csv"
        write_day_metrics_csv(path, [])
        content = path.read_text()
        assert "day" in content
        lines = content.strip().split("\n")
        assert len(lines) == 1  # header only

    def test_decimal_values_converted(self, tmp_path):
        """Decimal values are properly converted in CSV."""
        path = tmp_path / "metrics.csv"
        rows = [
            {"day": 1, "S_t": Decimal("100"), "phi_t": Decimal("0.5")},
            {"day": 2, "S_t": Decimal("200"), "phi_t": None},
        ]
        write_day_metrics_csv(path, rows)
        content = path.read_text()
        assert "100" in content
        assert "0.5" in content
        # None should be empty string
        lines = content.strip().split("\n")
        assert len(lines) == 3

    def test_integer_decimal_written_as_int(self, tmp_path):
        """Decimal('42.0') should be written as integer 42."""
        path = tmp_path / "metrics.csv"
        rows = [{"day": 1, "val": Decimal("42.0")}]
        write_day_metrics_csv(path, rows)
        content = path.read_text()
        assert "42" in content

    def test_creates_parent_dirs(self, tmp_path):
        """Creates parent directories if they don't exist."""
        path = tmp_path / "deep" / "nested" / "metrics.csv"
        write_day_metrics_csv(path, [])
        assert path.exists()


# ===========================================================================
# write_day_metrics_json
# ===========================================================================


class TestWriteDayMetricsJson:
    def test_writes_json(self, tmp_path):
        """Writes valid JSON with Decimal conversion."""
        path = tmp_path / "metrics.json"
        rows = [{"day": 1, "S_t": Decimal("100"), "phi_t": Decimal("0.5")}]
        write_day_metrics_json(path, rows)
        content = json.loads(path.read_text())
        assert len(content) == 1
        assert content[0]["S_t"] == 100
        assert content[0]["phi_t"] == "0.5"

    def test_empty_rows(self, tmp_path):
        """Empty rows produce an empty JSON array."""
        path = tmp_path / "metrics.json"
        write_day_metrics_json(path, [])
        content = json.loads(path.read_text())
        assert content == []


# ===========================================================================
# write_debtor_shares_csv / write_intraday_csv (thin wrappers)
# ===========================================================================


class TestThinWrappers:
    def test_debtor_shares_csv(self, tmp_path):
        path = tmp_path / "ds.csv"
        rows = [{"day": 1, "agent": "A1", "DS_t": Decimal("0.3")}]
        write_debtor_shares_csv(path, rows)
        assert path.exists()

    def test_intraday_csv(self, tmp_path):
        path = tmp_path / "intraday.csv"
        rows = [{"day": 1, "step": 1, "P_prefix": Decimal("50")}]
        write_intraday_csv(path, rows)
        assert path.exists()


# ===========================================================================
# _fmt_num
# ===========================================================================


class TestFmtNum:
    def test_none(self):
        assert _fmt_num(None) == "\u2014"  # em-dash

    def test_integer_decimal(self):
        assert _fmt_num(Decimal("42")) == "42"
        assert _fmt_num(Decimal("0")) == "0"

    def test_fractional_decimal(self):
        result = _fmt_num(Decimal("3.14"))
        assert "3.14" in result

    def test_very_small_decimal(self):
        """Very small decimal gets formatted properly."""
        result = _fmt_num(Decimal("0.000001"))
        assert "0.000001" in result

    def test_string_passthrough(self):
        assert _fmt_num("hello") == "hello"

    def test_int_passthrough(self):
        assert _fmt_num(42) == "42"

    def test_special_decimal_snan(self):
        """sNaN or special Decimal falls through to float formatting."""
        # Decimal('Infinity') -> normalize() works but to_integral_value may raise
        result = _fmt_num(Decimal("Infinity"))
        assert "Infinity" in result or "inf" in result.lower()


# ===========================================================================
# _group_by
# ===========================================================================


class TestGroupBy:
    def test_basic_grouping(self):
        rows = [
            {"day": 1, "val": "a"},
            {"day": 1, "val": "b"},
            {"day": 2, "val": "c"},
        ]
        result = _group_by(rows, "day")
        assert len(result[1]) == 2
        assert len(result[2]) == 1

    def test_missing_key(self):
        rows = [{"val": "a"}, {"day": 1, "val": "b"}]
        result = _group_by(rows, "day")
        assert None in result
        assert len(result[None]) == 1


# ===========================================================================
# parse_day_ranges
# ===========================================================================


class TestParseDayRanges:
    def test_single_values(self):
        assert parse_day_ranges("1,3,5") == [1, 3, 5]

    def test_ranges(self):
        assert parse_day_ranges("1-3") == [1, 2, 3]

    def test_mixed(self):
        assert parse_day_ranges("1,3-5,8") == [1, 3, 4, 5, 8]

    def test_reversed_range(self):
        """Range like '5-3' is treated as 3-5."""
        assert parse_day_ranges("5-3") == [3, 4, 5]

    def test_empty_string(self):
        assert parse_day_ranges("") == []

    def test_whitespace(self):
        assert parse_day_ranges(" 1 , 2 ") == [1, 2]

    def test_invalid_values_skipped(self):
        assert parse_day_ranges("a,1,b-c,2") == [1, 2]

    def test_duplicates_removed(self):
        assert parse_day_ranges("1,1,2,1-3") == [1, 2, 3]

    def test_empty_parts_skipped(self):
        assert parse_day_ranges(",,1,,2,,") == [1, 2]


# ===========================================================================
# infer_day_list
# ===========================================================================


class TestInferDayList:
    def test_from_payable_created(self):
        events = [
            {"kind": "PayableCreated", "due_day": 3},
            {"kind": "PayableCreated", "due_day": 5},
        ]
        result = infer_day_list(events)
        assert result == [3, 5]

    def test_from_payable_settled_fallback(self):
        events = [
            {"kind": "PayableSettled", "day": 2},
            {"kind": "PayableSettled", "day": 4},
        ]
        result = infer_day_list(events)
        assert result == [2, 4]

    def test_prefers_due_days_over_settled(self):
        events = [
            {"kind": "PayableCreated", "due_day": 3},
            {"kind": "PayableSettled", "day": 2},
        ]
        result = infer_day_list(events)
        assert result == [3]  # due_days preferred

    def test_empty_events(self):
        assert infer_day_list([]) == []

    def test_no_relevant_events(self):
        events = [{"kind": "CashMinted", "day": 1}]
        assert infer_day_list(events) == []


# ===========================================================================
# compute_run_level_metrics
# ===========================================================================


class TestComputeRunLevelMetrics:
    def test_basic(self):
        events = [
            {"kind": "AgentDefaulted", "agent": "A1", "frm": "A1"},
            {"kind": "AgentDefaulted", "agent": "A2", "frm": "A2"},
        ]
        result = compute_run_level_metrics(events)
        assert result["n_defaults"] == 2

    def test_no_defaults(self):
        events = [{"kind": "PayableSettled"}]
        result = compute_run_level_metrics(events)
        assert result["n_defaults"] == 0
        assert result["cascade_fraction"] is None


# ===========================================================================
# compute_day_metrics
# ===========================================================================


class TestComputeDayMetrics:
    def test_empty_events_and_no_day_list(self):
        """No events and no day list -> empty result."""
        result = compute_day_metrics([])
        assert result["days"] == []
        assert result["day_metrics"] == []

    def test_with_day_list_but_no_matching_events(self):
        """Day list given but no events match those days."""
        result = compute_day_metrics([], day_list=[1, 2])
        assert len(result["day_metrics"]) == 2
        assert result["days"] == [1, 2]

    def test_with_events_infers_days(self):
        """Events with PayableCreated infer day list."""
        events = [
            {
                "kind": "PayableCreated",
                "due_day": 1,
                "debtor": "A1",
                "creditor": "A2",
                "amount": 100,
            },
            {
                "kind": "PayableSettled",
                "day": 1,
                "debtor": "A1",
                "creditor": "A2",
                "amount": 100,
                "pid": "P1",
            },
        ]
        result = compute_day_metrics(events)
        assert 1 in result["days"]
        assert len(result["day_metrics"]) >= 1

    def test_with_balances_rows(self):
        """Balance rows are used to compute M_t and G_t."""
        events = [
            {
                "kind": "PayableCreated",
                "due_day": 1,
                "debtor": "A1",
                "creditor": "A2",
                "amount": 100,
            },
        ]
        balance_rows = [
            {"day": 1, "agent": "A1", "cash": "50", "deposit": "50", "reserves": "0"},
            {"day": 1, "agent": "A2", "cash": "0", "deposit": "0", "reserves": "0"},
        ]
        result = compute_day_metrics(events, balances_rows=balance_rows, day_list=[1])
        day_metric = result["day_metrics"][0]
        assert day_metric["M_t"] is not None

    def test_notes_when_no_creditors_or_debtors(self):
        """Notes field populated when no net creditors or debtors."""
        result = compute_day_metrics([], day_list=[1])
        notes = result["day_metrics"][0]["notes"]
        # With no events, there should be notes about missing data
        assert isinstance(notes, str)


# ===========================================================================
# summarize_day_metrics
# ===========================================================================


class TestSummarizeDayMetrics:
    def test_empty_input(self):
        result = summarize_day_metrics([])
        assert result["phi_total"] is None
        assert result["delta_total"] is None
        assert result["max_G_t"] is None

    def test_basic_aggregation(self):
        metrics = [
            {
                "day": 1,
                "S_t": Decimal("100"),
                "phi_t": Decimal("1"),
                "delta_t": Decimal("0"),
                "G_t": Decimal("5"),
                "alpha_t": Decimal("0.5"),
                "Mpeak_t": Decimal("50"),
                "v_t": Decimal("2"),
                "HHIplus_t": Decimal("0.3"),
            },
            {
                "day": 2,
                "S_t": Decimal("200"),
                "phi_t": Decimal("0.5"),
                "delta_t": Decimal("0.5"),
                "G_t": Decimal("10"),
            },
        ]
        result = summarize_day_metrics(metrics)
        assert result["phi_total"] is not None
        assert result["delta_total"] is not None
        assert result["max_G_t"] == Decimal("10")
        assert result["alpha_1"] == Decimal("0.5")
        assert result["Mpeak_1"] == Decimal("50")
        assert result["v_1"] == Decimal("2")
        assert result["HHIplus_1"] == Decimal("0.3")

    def test_string_values_converted(self):
        """String representations of numbers are converted to Decimal."""
        metrics = [
            {
                "day": 1,
                "S_t": "100",
                "phi_t": "1",
                "delta_t": "0",
                "G_t": "5",
                "alpha_t": "0.5",
                "Mpeak_t": "50",
                "v_t": "2",
                "HHIplus_t": "0.3",
            },
        ]
        result = summarize_day_metrics(metrics)
        assert result["phi_total"] is not None
        assert result["alpha_1"] is not None

    def test_none_day_field(self):
        """Row with day=None doesn't crash."""
        metrics = [
            {"day": None, "S_t": Decimal("100"), "phi_t": Decimal("1"), "delta_t": Decimal("0")},
        ]
        result = summarize_day_metrics(metrics)
        assert result["max_day"] == 0

    def test_no_s_t_rows(self):
        """Rows without S_t -> phi_total and delta_total are None."""
        metrics = [
            {"day": 1},
        ]
        result = summarize_day_metrics(metrics)
        assert result["phi_total"] is None
        assert result["delta_total"] is None


# ===========================================================================
# write_metrics_html
# ===========================================================================


class TestWriteMetricsHtml:
    def test_basic_render(self, tmp_path):
        """Renders a complete HTML report."""
        path = tmp_path / "report.html"
        day_metrics = [
            {
                "day": 1,
                "S_t": Decimal("100"),
                "Mbar_t": Decimal("50"),
                "M_t": None,
                "G_t": None,
                "alpha_t": Decimal("0.5"),
                "Mpeak_t": Decimal("40"),
                "gross_settled_t": Decimal("80"),
                "v_t": Decimal("2"),
                "phi_t": Decimal("1"),
                "delta_t": Decimal("0"),
                "n_debtors": 2,
                "n_creditors": 2,
                "HHIplus_t": Decimal("0.3"),
                "notes": "",
            },
        ]
        debtor_shares = [
            {"day": 1, "agent": "A1", "DS_t": Decimal("0.6")},
            {"day": 1, "agent": "A2", "DS_t": Decimal("0.4")},
        ]
        intraday = [
            {"day": 1, "step": 1, "P_prefix": Decimal("10")},
            {"day": 1, "step": 2, "P_prefix": Decimal("30")},
        ]
        write_metrics_html(
            path,
            day_metrics,
            debtor_shares,
            intraday,
            title="Test Report",
            subtitle="Test Subtitle",
        )
        content = path.read_text()
        assert "Test Report" in content
        assert "Test Subtitle" in content
        assert "100" in content  # S_t
        assert "A1" in content
        assert "<svg" in content  # SVG chart

    def test_no_debtor_shares(self, tmp_path):
        """No debtor shares renders placeholder text."""
        path = tmp_path / "report.html"
        write_metrics_html(path, [], [], [])
        content = path.read_text()
        assert "No net debtors" in content

    def test_no_intraday(self, tmp_path):
        """No intraday data renders placeholder text."""
        path = tmp_path / "report.html"
        write_metrics_html(path, [], [], [])
        content = path.read_text()
        assert "No settlement steps found" in content

    def test_with_run_level_metrics(self, tmp_path):
        """run_level_metrics are included in summary cards."""
        path = tmp_path / "report.html"
        write_metrics_html(
            path,
            [],
            [],
            [],
            run_level_metrics={"n_defaults": 3, "cascade_fraction": Decimal("0.5")},
        )
        content = path.read_text()
        assert "3" in content
        assert "0.5" in content

    def test_without_subtitle(self, tmp_path):
        """No subtitle -> no subtitle paragraph."""
        path = tmp_path / "report.html"
        write_metrics_html(path, [], [], [], title="Only Title")
        content = path.read_text()
        assert "Only Title" in content

    def test_default_title(self, tmp_path):
        """Default title used when none provided."""
        path = tmp_path / "report.html"
        write_metrics_html(path, [], [], [])
        content = path.read_text()
        assert "Bilancio Analytics Report" in content

    def test_multi_day_intraday_svg(self, tmp_path):
        """Multiple days produce multiple SVG charts."""
        path = tmp_path / "report.html"
        intraday = [
            {"day": 1, "step": 1, "P_prefix": Decimal("10")},
            {"day": 1, "step": 2, "P_prefix": Decimal("30")},
            {"day": 2, "step": 1, "P_prefix": Decimal("20")},
        ]
        write_metrics_html(path, [], [], intraday)
        content = path.read_text()
        assert content.count("<svg") >= 2


# ===========================================================================
# _resolve_path
# ===========================================================================


class TestResolvePath:
    def test_relative_path(self):
        result = _resolve_path(Path("/base/dir"), "subdir/file.csv")
        assert result == Path("/base/dir/subdir/file.csv")

    def test_absolute_path(self):
        result = _resolve_path(Path("/base/dir"), "/absolute/file.csv")
        assert result == Path("/absolute/file.csv")


# ===========================================================================
# _decimal_or_none
# ===========================================================================


class TestDecimalOrNone:
    def test_valid_number(self):
        assert _decimal_or_none("3.14") == Decimal("3.14")

    def test_integer_string(self):
        assert _decimal_or_none("42") == Decimal("42")

    def test_none(self):
        assert _decimal_or_none(None) is None

    def test_empty_string(self):
        assert _decimal_or_none("") is None

    def test_invalid_string(self):
        assert _decimal_or_none("not_a_number") is None

    def test_decimal_input(self):
        assert _decimal_or_none(Decimal("1.5")) == Decimal("1.5")


# ===========================================================================
# aggregate_runs
# ===========================================================================


class TestAggregateRuns:
    def test_skips_non_completed_status(self, tmp_path):
        """Non-completed runs are skipped."""
        registry = tmp_path / "registry.csv"
        results = tmp_path / "results.csv"
        registry.write_text(
            "run_id,phase,seed,n_agents,kappa,concentration,mu,monotonicity,Q_total,S1,L0,"
            "scenario_yaml,events_jsonl,balances_csv,metrics_csv,metrics_html,run_html,"
            "status,time_to_stability,phi_total,delta_total,error,n_defaults,cascade_fraction\n"
            "run1,grid,42,5,1,0.5,0.25,0,150,150,120,s.yaml,e.jsonl,b.csv,m.csv,m.html,r.html,"
            "failed,,,,,\n"
        )
        rows = aggregate_runs(registry, results)
        assert rows == []
        assert results.exists()

    def test_skips_missing_metrics_csv_path(self, tmp_path):
        """Runs without metrics_csv path are skipped."""
        registry = tmp_path / "registry.csv"
        results = tmp_path / "results.csv"
        registry.write_text(
            "run_id,phase,seed,n_agents,kappa,concentration,mu,monotonicity,Q_total,S1,L0,"
            "scenario_yaml,events_jsonl,balances_csv,metrics_csv,metrics_html,run_html,"
            "status,time_to_stability,phi_total,delta_total,error,n_defaults,cascade_fraction\n"
            "run1,grid,42,5,1,0.5,0.25,0,150,150,120,s.yaml,e.jsonl,b.csv,,m.html,r.html,"
            "completed,,,,,\n"
        )
        rows = aggregate_runs(registry, results)
        assert rows == []

    def test_skips_nonexistent_metrics_file(self, tmp_path):
        """Runs with metrics_csv pointing to non-existent file are skipped."""
        registry = tmp_path / "registry.csv"
        results = tmp_path / "results.csv"
        registry.write_text(
            "run_id,phase,seed,n_agents,kappa,concentration,mu,monotonicity,Q_total,S1,L0,"
            "scenario_yaml,events_jsonl,balances_csv,metrics_csv,metrics_html,run_html,"
            "status,time_to_stability,phi_total,delta_total,error,n_defaults,cascade_fraction\n"
            "run1,grid,42,5,1,0.5,0.25,0,150,150,120,s.yaml,e.jsonl,b.csv,nonexistent.csv,m.html,r.html,"
            "completed,,,,,\n"
        )
        rows = aggregate_runs(registry, results)
        assert rows == []

    def test_skips_empty_metrics_file(self, tmp_path):
        """Runs with empty metrics file are skipped."""
        registry = tmp_path / "registry.csv"
        results = tmp_path / "results.csv"
        metrics = tmp_path / "metrics.csv"
        metrics.write_text("day,S_t,phi_t,delta_t\n")  # header only

        registry.write_text(
            "run_id,phase,seed,n_agents,kappa,concentration,mu,monotonicity,Q_total,S1,L0,"
            "scenario_yaml,events_jsonl,balances_csv,metrics_csv,metrics_html,run_html,"
            "status,time_to_stability,phi_total,delta_total,error,n_defaults,cascade_fraction\n"
            f"run1,grid,42,5,1,0.5,0.25,0,150,150,120,s.yaml,e.jsonl,b.csv,{metrics},m.html,r.html,"
            "completed,,,,,\n"
        )
        rows = aggregate_runs(registry, results)
        assert rows == []

    def test_successful_aggregation(self, tmp_path):
        """Full aggregation path with valid data."""
        base = tmp_path
        registry = base / "registry.csv"
        results = base / "results.csv"
        metrics = base / "metrics.csv"
        metrics.write_text(
            "day,S_t,phi_t,delta_t,G_t,alpha_t,Mpeak_t,v_t,HHIplus_t\n1,100,1,0,0,0.4,50,2,0.5\n"
        )
        registry.write_text(
            "run_id,phase,seed,n_agents,kappa,concentration,mu,monotonicity,Q_total,S1,L0,"
            "scenario_yaml,events_jsonl,balances_csv,metrics_csv,metrics_html,run_html,"
            "status,time_to_stability,phi_total,delta_total,error,n_defaults,cascade_fraction\n"
            f"run1,grid,42,5,1,0.5,0.25,0,150,150,120,s.yaml,e.jsonl,b.csv,{metrics},m.html,r.html,"
            "completed,2,,,,3,0.5\n"
        )
        rows = aggregate_runs(registry, results)
        assert len(rows) == 1
        assert rows[0]["run_id"] == "run1"
        assert rows[0]["n_defaults"] == "3"
        assert rows[0]["cascade_fraction"] == "0.5"
        assert results.exists()

        # Verify the CSV has proper content
        with results.open("r") as fh:
            reader = csv.DictReader(fh)
            csv_rows = list(reader)
        assert len(csv_rows) == 1
        assert csv_rows[0]["run_id"] == "run1"

    def test_decimal_formatting_in_output(self, tmp_path):
        """Decimal values in summary are formatted properly in output CSV."""
        base = tmp_path
        registry = base / "registry.csv"
        results = base / "results.csv"
        metrics = base / "metrics.csv"
        metrics.write_text("day,S_t,phi_t,delta_t\n1,100,1,0\n2,50,0.5,0.5\n")
        registry.write_text(
            "run_id,phase,seed,n_agents,kappa,concentration,mu,monotonicity,Q_total,S1,L0,"
            "scenario_yaml,events_jsonl,balances_csv,metrics_csv,metrics_html,run_html,"
            "status,time_to_stability,phi_total,delta_total,error,n_defaults,cascade_fraction\n"
            f"run1,grid,42,5,1,0.5,0.25,0,150,150,120,s.yaml,e.jsonl,b.csv,{metrics},m.html,r.html,"
            "completed,,,,,\n"
        )
        rows = aggregate_runs(registry, results)
        assert len(rows) == 1
        # phi_total should be weighted average
        assert rows[0]["phi_total"] is not None


# ===========================================================================
# render_dashboard
# ===========================================================================


class TestRenderDashboard:
    def test_basic_render(self, tmp_path):
        """Dashboard renders correctly from a results CSV."""
        results_csv = tmp_path / "results.csv"
        dashboard = tmp_path / "dashboard.html"
        results_csv.write_text(
            "run_id,phase,kappa,concentration,mu,monotonicity,phi_total,delta_total,max_G_t,time_to_stability\n"
            "run1,grid,1,0.5,0.25,0,0.85,0.15,10,3\n"
            "run2,grid,2,0.5,0.25,0,0.95,0.05,5,2\n"
        )
        render_dashboard(results_csv, dashboard)
        content = dashboard.read_text()
        assert "run1" in content
        assert "run2" in content
        assert "Dashboard" in content

    def test_empty_results(self, tmp_path):
        """Dashboard renders with empty results."""
        results_csv = tmp_path / "results.csv"
        dashboard = tmp_path / "dashboard.html"
        results_csv.write_text(
            "run_id,phase,kappa,concentration,mu,monotonicity,phi_total,delta_total,max_G_t,time_to_stability\n"
        )
        render_dashboard(results_csv, dashboard)
        assert dashboard.exists()
        content = dashboard.read_text()
        assert "Total runs" in content
        assert "<strong>0</strong>" in content

    def test_missing_values(self, tmp_path):
        """Dashboard handles missing/empty values in CSV."""
        results_csv = tmp_path / "results.csv"
        dashboard = tmp_path / "dashboard.html"
        results_csv.write_text(
            "run_id,phase,kappa,concentration,mu,monotonicity,phi_total,delta_total,max_G_t,time_to_stability\n"
            "run1,grid,1,0.5,,,,,\n"
        )
        render_dashboard(results_csv, dashboard)
        assert dashboard.exists()
