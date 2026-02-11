"""Tests for strategy_outcomes analysis module."""

from __future__ import annotations

import pytest
from pathlib import Path

import pandas as pd

from bilancio.analysis.strategy_outcomes import (
    STRATEGIES,
    build_strategy_outcomes_by_run,
    build_strategy_outcomes_overall,
    run_strategy_analysis,
)


# ---------------------------------------------------------------------------
# Helpers to build a minimal experiment directory tree on disk
# ---------------------------------------------------------------------------

COMPARISON_CSV_HEADER = (
    "kappa,concentration,mu,outside_mid_ratio,seed,face_value,"
    "active_run_id,passive_run_id,delta_passive,delta_active,"
    "trading_effect,trading_relief_ratio\n"
)

REPAYMENT_EVENTS_HEADER = (
    "agent_id,instrument_id,face_value,due_day,outcome,strategy,"
    "buy_count,sell_count\n"
)


def _write_comparison_csv(root: Path, rows: list[str]) -> Path:
    """Write aggregate/comparison.csv with given data rows."""
    agg = root / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    path = agg / "comparison.csv"
    path.write_text(COMPARISON_CSV_HEADER + "\n".join(rows) + "\n")
    return path


def _write_repayment_events(root: Path, run_id: str, rows: list[str]) -> Path:
    """Write active/runs/<run_id>/out/repayment_events.csv."""
    out = root / "active" / "runs" / run_id / "out"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "repayment_events.csv"
    path.write_text(REPAYMENT_EVENTS_HEADER + "\n".join(rows) + "\n")
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def experiment_root(tmp_path: Path) -> Path:
    """Create a minimal experiment directory with one active run."""
    run_id = "run_001"
    # comparison.csv
    _write_comparison_csv(
        tmp_path,
        [
            f"0.5,1.0,0.5,1.0,42,1000,{run_id},run_p_001,0.3,0.2,0.1,0.33",
        ],
    )
    # repayment_events.csv
    _write_repayment_events(
        tmp_path,
        run_id,
        [
            "H1,I1,100,3,repaid,no_trade,0,0",
            "H2,I2,200,4,defaulted,hold_to_maturity,0,0",
            "H3,I3,150,5,repaid,sell_before,1,1",
            "H4,I4,250,3,repaid,round_trip,2,1",
            "H5,I5,300,4,defaulted,no_trade,0,0",
        ],
    )
    return tmp_path


@pytest.fixture
def experiment_two_runs(tmp_path: Path) -> Path:
    """Create an experiment with two active runs."""
    _write_comparison_csv(
        tmp_path,
        [
            "0.5,1.0,0.5,1.0,42,1000,run_a,run_pa,0.3,0.2,0.1,0.33",
            "0.5,1.0,0.5,1.0,99,1000,run_b,run_pb,0.4,0.15,0.25,0.625",
        ],
    )
    # Run A
    _write_repayment_events(
        tmp_path,
        "run_a",
        [
            "H1,I1,100,3,repaid,no_trade,0,0",
            "H2,I2,200,4,defaulted,hold_to_maturity,0,0",
        ],
    )
    # Run B
    _write_repayment_events(
        tmp_path,
        "run_b",
        [
            "H1,I1,500,3,repaid,sell_before,1,0",
            "H2,I2,500,4,defaulted,sell_before,1,1",
        ],
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: build_strategy_outcomes_by_run
# ---------------------------------------------------------------------------


class TestBuildStrategyOutcomesByRun:
    """Tests for build_strategy_outcomes_by_run."""

    def test_basic_single_run(self, experiment_root: Path) -> None:
        df = build_strategy_outcomes_by_run(experiment_root)
        assert len(df) == 1
        row = df.iloc[0]

        # Run parameters carried over
        assert row["run_id"] == "run_001"
        assert float(row["kappa"]) == 0.5
        assert float(row["delta_passive"]) == 0.3
        assert float(row["delta_active"]) == 0.2
        assert float(row["trading_effect"]) == 0.1

        # Totals
        assert row["total_liabilities"] == 5
        # Total face = 100 + 200 + 150 + 250 + 300 = 1000
        assert row["total_face_value"] == 1000.0
        # Defaults: I2 (200) + I5 (300) = 500
        assert row["default_count_total"] == 2
        assert row["default_face_total"] == 500.0
        assert row["default_rate_total"] == pytest.approx(0.5)

    def test_per_strategy_counts(self, experiment_root: Path) -> None:
        df = build_strategy_outcomes_by_run(experiment_root)
        row = df.iloc[0]

        # no_trade: H1 (repaid, 100) + H5 (defaulted, 300) = 2 items, face 400
        assert row["count_no_trade"] == 2
        assert row["face_no_trade"] == 400.0
        assert row["default_count_no_trade"] == 1
        assert row["default_face_no_trade"] == 300.0
        assert row["default_rate_no_trade"] == pytest.approx(300.0 / 400.0)

        # hold_to_maturity: H2 (defaulted, 200)
        assert row["count_hold_to_maturity"] == 1
        assert row["face_hold_to_maturity"] == 200.0
        assert row["default_count_hold_to_maturity"] == 1
        assert row["default_face_hold_to_maturity"] == 200.0
        assert row["default_rate_hold_to_maturity"] == pytest.approx(1.0)

        # sell_before: H3 (repaid, 150)
        assert row["count_sell_before"] == 1
        assert row["face_sell_before"] == 150.0
        assert row["default_count_sell_before"] == 0
        assert row["default_face_sell_before"] == 0.0
        assert row["default_rate_sell_before"] == 0.0

        # round_trip: H4 (repaid, 250)
        assert row["count_round_trip"] == 1
        assert row["face_round_trip"] == 250.0
        assert row["default_count_round_trip"] == 0
        assert row["default_face_round_trip"] == 0.0
        assert row["default_rate_round_trip"] == 0.0

    def test_missing_comparison_csv(self, tmp_path: Path) -> None:
        """No comparison.csv -> empty DataFrame."""
        df = build_strategy_outcomes_by_run(tmp_path)
        assert df.empty

    def test_missing_repayment_events(self, tmp_path: Path) -> None:
        """comparison.csv exists but no repayment_events.csv -> empty."""
        _write_comparison_csv(tmp_path, ["0.5,1,0.5,1,42,1000,run_x,run_px,0.3,0.2,0.1,0.33"])
        df = build_strategy_outcomes_by_run(tmp_path)
        assert df.empty

    def test_empty_repayment_events(self, tmp_path: Path) -> None:
        """repayment_events.csv with only headers -> skip run."""
        run_id = "run_empty"
        _write_comparison_csv(
            tmp_path,
            [f"0.5,1,0.5,1,42,1000,{run_id},run_p,0.3,0.2,0.1,0.33"],
        )
        out = tmp_path / "active" / "runs" / run_id / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "repayment_events.csv").write_text(REPAYMENT_EVENTS_HEADER)
        df = build_strategy_outcomes_by_run(tmp_path)
        assert df.empty

    def test_blank_active_run_id_raises(self, tmp_path: Path) -> None:
        """Rows with empty active_run_id cause a TypeError.

        pandas reads empty CSV fields as NaN (float), which is truthy,
        so ``if not run_id`` does not skip it.  The subsequent Path
        construction fails because Path / float is unsupported.
        """
        _write_comparison_csv(
            tmp_path,
            ["0.5,1,0.5,1,42,1000,,run_p,0.3,0.2,0.1,0.33"],
        )
        with pytest.raises(TypeError):
            build_strategy_outcomes_by_run(tmp_path)

    def test_corrupt_repayment_events_unreadable(self, tmp_path: Path) -> None:
        """Truly unreadable CSV is caught by the try/except around read_csv."""
        run_id = "run_bad"
        _write_comparison_csv(
            tmp_path,
            [f"0.5,1,0.5,1,42,1000,{run_id},run_p,0.3,0.2,0.1,0.33"],
        )
        out = tmp_path / "active" / "runs" / run_id / "out"
        out.mkdir(parents=True, exist_ok=True)
        # Write binary garbage that causes pd.read_csv to raise
        (out / "repayment_events.csv").write_bytes(b"\x00\x01\x02\x03")
        # The try/except around pd.read_csv catches this, run is skipped
        df = build_strategy_outcomes_by_run(tmp_path)
        assert df.empty

    def test_missing_columns_in_repayment_events(self, tmp_path: Path) -> None:
        """CSV with wrong columns raises KeyError (not caught internally).

        The code only has try/except around pd.read_csv, not around
        subsequent column access.  A file with unexpected columns will
        raise KeyError.
        """
        run_id = "run_bad_cols"
        _write_comparison_csv(
            tmp_path,
            [f"0.5,1,0.5,1,42,1000,{run_id},run_p,0.3,0.2,0.1,0.33"],
        )
        out = tmp_path / "active" / "runs" / run_id / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "repayment_events.csv").write_text("bad_col\nval\n")
        with pytest.raises(KeyError):
            build_strategy_outcomes_by_run(tmp_path)

    def test_non_numeric_face_value(self, tmp_path: Path) -> None:
        """Non-numeric face_value entries are coerced to 0."""
        run_id = "run_nan"
        _write_comparison_csv(
            tmp_path,
            [f"0.5,1,0.5,1,42,1000,{run_id},run_p,0.3,0.2,0.1,0.33"],
        )
        _write_repayment_events(
            tmp_path,
            run_id,
            [
                "H1,I1,not_a_number,3,repaid,no_trade,0,0",
                "H2,I2,200,4,defaulted,no_trade,0,0",
            ],
        )
        df = build_strategy_outcomes_by_run(tmp_path)
        assert len(df) == 1
        row = df.iloc[0]
        # "not_a_number" coerced to 0, so total = 0 + 200 = 200
        assert row["total_face_value"] == 200.0

    def test_two_runs(self, experiment_two_runs: Path) -> None:
        df = build_strategy_outcomes_by_run(experiment_two_runs)
        assert len(df) == 2
        ids = set(df["run_id"])
        assert ids == {"run_a", "run_b"}

    def test_zero_total_face_default_rate(self, tmp_path: Path) -> None:
        """When all face_values are 0, default_rate_total should be 0."""
        run_id = "run_zero"
        _write_comparison_csv(
            tmp_path,
            [f"0.5,1,0.5,1,42,0,{run_id},run_p,0,0,0,0"],
        )
        _write_repayment_events(
            tmp_path,
            run_id,
            [
                "H1,I1,0,3,repaid,no_trade,0,0",
                "H2,I2,0,4,defaulted,no_trade,0,0",
            ],
        )
        df = build_strategy_outcomes_by_run(tmp_path)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["default_rate_total"] == 0.0
        assert row["default_rate_no_trade"] == 0.0


# ---------------------------------------------------------------------------
# Tests: build_strategy_outcomes_overall
# ---------------------------------------------------------------------------


class TestBuildStrategyOutcomesOverall:
    """Tests for build_strategy_outcomes_overall."""

    def test_basic_aggregation(self, experiment_root: Path) -> None:
        by_run = build_strategy_outcomes_by_run(experiment_root)
        overall = build_strategy_outcomes_overall(by_run)

        assert not overall.empty
        # Should have one row per strategy per parameter combo
        # 1 combo x 4 strategies = 4 rows
        assert len(overall) == 4
        strategies_in_result = set(overall["strategy"])
        assert strategies_in_result == set(STRATEGIES)

    def test_aggregation_multiple_runs(self, experiment_two_runs: Path) -> None:
        by_run = build_strategy_outcomes_by_run(experiment_two_runs)
        overall = build_strategy_outcomes_overall(by_run)
        assert not overall.empty
        # Same kappa/conc/mu/ratio -> 1 group, 4 strategies
        assert len(overall) == 4

        # sell_before: run_a has 0, run_b has 500+500=1000 face
        sb_row = overall[overall["strategy"] == "sell_before"].iloc[0]
        assert sb_row["total_face"] == 1000.0
        # default_face for sell_before: run_b has 500 defaulted
        assert sb_row["default_face"] == 500.0
        assert sb_row["default_rate"] == pytest.approx(0.5)
        # runs_using = 1 (only run_b had sell_before)
        assert sb_row["runs_using_strategy"] == 1

    def test_empty_input(self) -> None:
        result = build_strategy_outcomes_overall(pd.DataFrame())
        assert result.empty

    def test_custom_group_cols(self, experiment_root: Path) -> None:
        by_run = build_strategy_outcomes_by_run(experiment_root)
        overall = build_strategy_outcomes_overall(by_run, group_cols=["kappa"])
        assert not overall.empty
        assert "kappa" in overall.columns

    def test_no_matching_group_cols(self, experiment_root: Path) -> None:
        by_run = build_strategy_outcomes_by_run(experiment_root)
        overall = build_strategy_outcomes_overall(
            by_run, group_cols=["nonexistent_column"]
        )
        assert overall.empty

    def test_single_group_col(self, experiment_root: Path) -> None:
        """Single group column triggers the tuple wrapping logic."""
        by_run = build_strategy_outcomes_by_run(experiment_root)
        overall = build_strategy_outcomes_overall(by_run, group_cols=["kappa"])
        assert not overall.empty
        assert len(overall) == 4  # 1 kappa value x 4 strategies

    def test_mean_trading_effect(self, experiment_two_runs: Path) -> None:
        by_run = build_strategy_outcomes_by_run(experiment_two_runs)
        overall = build_strategy_outcomes_overall(by_run)

        # no_trade: used by run_a (trading_effect=0.1), run_b doesn't use it
        nt_row = overall[overall["strategy"] == "no_trade"].iloc[0]
        # run_a used no_trade (count=1), run_b did not (count=0)
        # mean_trading_effect should be 0.1 (only run_a)
        assert nt_row["mean_trading_effect"] == pytest.approx(0.1)

    def test_default_rate_zero_face(self) -> None:
        """Strategy with 0 total face should have 0 default_rate."""
        data = {
            "run_id": ["r1"],
            "kappa": [0.5],
            "concentration": [1.0],
            "mu": [0.5],
            "outside_mid_ratio": [1.0],
            "trading_effect": [0.1],
        }
        for strat in STRATEGIES:
            data[f"count_{strat}"] = [0]
            data[f"face_{strat}"] = [0.0]
            data[f"default_count_{strat}"] = [0]
            data[f"default_face_{strat}"] = [0.0]
            data[f"default_rate_{strat}"] = [0.0]
        by_run = pd.DataFrame(data)
        overall = build_strategy_outcomes_overall(by_run)
        assert not overall.empty
        for _, row in overall.iterrows():
            assert row["default_rate"] == 0.0

    def test_different_parameter_combos(self, tmp_path: Path) -> None:
        """Two different kappa values should yield separate groups."""
        _write_comparison_csv(
            tmp_path,
            [
                "0.3,1.0,0.5,1.0,42,1000,run_lo,run_plo,0.5,0.4,0.1,0.2",
                "2.0,1.0,0.5,1.0,42,1000,run_hi,run_phi,0.1,0.05,0.05,0.5",
            ],
        )
        _write_repayment_events(
            tmp_path,
            "run_lo",
            ["H1,I1,100,3,defaulted,no_trade,0,0"],
        )
        _write_repayment_events(
            tmp_path,
            "run_hi",
            ["H1,I1,100,3,repaid,no_trade,0,0"],
        )
        by_run = build_strategy_outcomes_by_run(tmp_path)
        assert len(by_run) == 2
        overall = build_strategy_outcomes_overall(by_run)
        # 2 kappa groups x 4 strategies = 8 rows
        assert len(overall) == 8


# ---------------------------------------------------------------------------
# Tests: run_strategy_analysis (integration)
# ---------------------------------------------------------------------------


class TestRunStrategyAnalysis:
    """Tests for the top-level run_strategy_analysis function."""

    def test_writes_output_files(self, experiment_root: Path) -> None:
        by_run_path, overall_path = run_strategy_analysis(experiment_root)
        assert by_run_path.exists()
        assert overall_path.exists()

        # Verify content
        by_run_df = pd.read_csv(by_run_path)
        assert len(by_run_df) == 1
        overall_df = pd.read_csv(overall_path)
        assert len(overall_df) == 4

    def test_no_data_returns_empty_paths(self, tmp_path: Path) -> None:
        """When no comparison.csv exists, return empty Paths."""
        by_run_path, overall_path = run_strategy_analysis(tmp_path)
        assert by_run_path == Path()
        assert overall_path == Path()

    def test_creates_aggregate_dir(self, tmp_path: Path) -> None:
        """Aggregate directory is created if missing."""
        run_id = "run_mk"
        _write_comparison_csv(
            tmp_path,
            [f"0.5,1,0.5,1,42,1000,{run_id},run_p,0.3,0.2,0.1,0.33"],
        )
        _write_repayment_events(
            tmp_path,
            run_id,
            ["H1,I1,100,3,repaid,no_trade,0,0"],
        )
        by_run_path, overall_path = run_strategy_analysis(tmp_path)
        assert by_run_path.exists()
        assert overall_path.exists()


# ---------------------------------------------------------------------------
# Tests: STRATEGIES constant
# ---------------------------------------------------------------------------


def test_strategies_constant() -> None:
    """STRATEGIES list has expected members."""
    assert "no_trade" in STRATEGIES
    assert "hold_to_maturity" in STRATEGIES
    assert "sell_before" in STRATEGIES
    assert "round_trip" in STRATEGIES
    assert len(STRATEGIES) == 4
