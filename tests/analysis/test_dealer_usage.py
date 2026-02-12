"""Tests for dealer_usage_summary analysis module."""

from __future__ import annotations

import pytest
from pathlib import Path

import pandas as pd

from bilancio.analysis.dealer_usage_summary import (
    build_dealer_usage_by_run,
    run_dealer_usage_analysis,
    _compute_trade_metrics,
    _compute_inventory_metrics,
    _compute_system_state_metrics,
    _compute_repayment_metrics,
    _empty_trade_metrics,
    _empty_inventory_metrics,
    _empty_system_state_metrics,
    _empty_repayment_metrics,
)


# ---------------------------------------------------------------------------
# Helpers to build experiment directory trees
# ---------------------------------------------------------------------------

COMPARISON_CSV_HEADER = (
    "kappa,concentration,mu,outside_mid_ratio,seed,face_value,"
    "active_run_id,passive_run_id,delta_passive,delta_active,"
    "trading_effect,trading_relief_ratio\n"
)


def _write_comparison_csv(root: Path, rows: list[str]) -> Path:
    agg = root / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    path = agg / "comparison.csv"
    path.write_text(COMPARISON_CSV_HEADER + "\n".join(rows) + "\n")
    return path


def _write_csv(root: Path, run_id: str, filename: str, content: str) -> Path:
    """Write a CSV file under active/runs/<run_id>/out/."""
    out = root / "active" / "runs" / run_id / "out"
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def experiment_root(tmp_path: Path) -> Path:
    """Create a full experiment with all four CSV types for one run."""
    run_id = "run_001"

    # comparison.csv
    _write_comparison_csv(
        tmp_path,
        [f"0.5,1.0,0.5,1.0,42,1000,{run_id},run_p_001,0.3,0.2,0.1,0.33"],
    )

    # trades.csv
    _write_csv(
        tmp_path,
        run_id,
        "trades.csv",
        (
            "trader_id,instrument_id,direction,face_value,price,day\n"
            "H1,I1,sell,100,90,1\n"
            "H2,I2,buy,200,180,1\n"
            "H1,I3,sell,150,130,2\n"
            "D1,I4,buy,300,260,2\n"  # non-H trader (dealer-to-dealer)
        ),
    )

    # inventory_timeseries.csv
    _write_csv(
        tmp_path,
        run_id,
        "inventory_timeseries.csv",
        (
            "day,step,dealer_id,dealer_inventory,is_at_zero,hit_vbt_this_step\n"
            "1,0,D1,100,False,False\n"
            "1,1,D1,50,False,True\n"
            "2,0,D1,0,True,False\n"
            "2,1,D1,200,False,False\n"
        ),
    )

    # system_state_timeseries.csv
    _write_csv(
        tmp_path,
        run_id,
        "system_state_timeseries.csv",
        (
            "day,debt_to_money,total_face_value,total_money\n"
            "0,2.0,10000,5000\n"
            "1,1.5,8000,5000\n"
            "2,1.0,6000,5000\n"
        ),
    )

    # repayment_events.csv
    _write_csv(
        tmp_path,
        run_id,
        "repayment_events.csv",
        (
            "agent_id,instrument_id,face_value,due_day,outcome,strategy,"
            "buy_count,sell_count\n"
            "H1,I1,100,3,repaid,sell_before,1,1\n"
            "H2,I2,200,4,defaulted,no_trade,0,0\n"
            "H3,I3,150,5,repaid,no_trade,0,0\n"
            "H4,I4,250,3,defaulted,hold_to_maturity,1,0\n"
        ),
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: _compute_trade_metrics
# ---------------------------------------------------------------------------


class TestComputeTradeMetrics:
    """Tests for _compute_trade_metrics."""

    def test_basic(self) -> None:
        trades = pd.DataFrame(
            {
                "trader_id": ["H1", "H2", "D1"],
                "face_value": [100, 200, 300],
                "price": [90, 180, 260],
            }
        )
        m = _compute_trade_metrics(trades)
        assert m["dealer_trade_count"] == 3
        assert m["trader_dealer_trade_count"] == 2  # only H-prefix
        assert m["n_traders_using_dealer"] == 2  # H1, H2
        assert m["total_face_traded"] == 300.0  # H1+H2
        assert m["total_cash_volume"] == 270.0  # 90+180

    def test_empty_dataframe(self) -> None:
        m = _compute_trade_metrics(pd.DataFrame())
        assert m == _empty_trade_metrics()

    def test_no_h_traders(self) -> None:
        """All trades are dealer-to-dealer (no H-prefix)."""
        trades = pd.DataFrame(
            {
                "trader_id": ["D1", "D2"],
                "face_value": [100, 200],
                "price": [90, 180],
            }
        )
        m = _compute_trade_metrics(trades)
        assert m["dealer_trade_count"] == 2
        assert m["trader_dealer_trade_count"] == 0
        assert m["n_traders_using_dealer"] == 0
        assert m["total_face_traded"] == 0.0
        assert m["total_cash_volume"] == 0.0

    def test_missing_trader_id_column(self) -> None:
        """No trader_id column -> trader_trades is empty."""
        trades = pd.DataFrame(
            {"face_value": [100, 200], "price": [90, 180]}
        )
        m = _compute_trade_metrics(trades)
        assert m["dealer_trade_count"] == 2
        assert m["trader_dealer_trade_count"] == 0

    def test_missing_face_value_column(self) -> None:
        """No face_value column for H-traders -> total_face_traded is 0."""
        trades = pd.DataFrame(
            {"trader_id": ["H1", "H2"], "price": [90, 180]}
        )
        m = _compute_trade_metrics(trades)
        assert m["trader_dealer_trade_count"] == 2
        assert m["total_face_traded"] == 0.0
        assert m["total_cash_volume"] == 270.0

    def test_missing_price_column(self) -> None:
        """No price column for H-traders -> total_cash_volume is 0."""
        trades = pd.DataFrame(
            {"trader_id": ["H1"], "face_value": [100]}
        )
        m = _compute_trade_metrics(trades)
        assert m["total_cash_volume"] == 0.0
        assert m["total_face_traded"] == 100.0

    def test_nan_trader_ids(self) -> None:
        """NaN trader_id values should not crash str.startswith."""
        trades = pd.DataFrame(
            {
                "trader_id": [None, "H1", float("nan")],
                "face_value": [100, 200, 300],
                "price": [90, 180, 260],
            }
        )
        m = _compute_trade_metrics(trades)
        assert m["trader_dealer_trade_count"] == 1  # only H1
        assert m["n_traders_using_dealer"] == 1


# ---------------------------------------------------------------------------
# Tests: _compute_inventory_metrics
# ---------------------------------------------------------------------------


class TestComputeInventoryMetrics:
    """Tests for _compute_inventory_metrics."""

    def test_basic(self) -> None:
        inv = pd.DataFrame(
            {
                "day": [1, 1, 2, 2],
                "dealer_inventory": [100, 50, 0, 200],
                "is_at_zero": [False, False, True, False],
                "hit_vbt_this_step": [False, True, False, False],
            }
        )
        m = _compute_inventory_metrics(inv)
        # Day 1: any_positive=True (100>0), all_zero=False, any_vbt=True
        # Day 2: any_positive=True (200>0), all_zero=False, any_vbt=False
        assert m["dealer_active_fraction"] == pytest.approx(1.0)
        assert m["dealer_empty_fraction"] == pytest.approx(0.0)
        assert m["vbt_usage_fraction"] == pytest.approx(0.5)

    def test_empty_dataframe(self) -> None:
        m = _compute_inventory_metrics(pd.DataFrame())
        assert m == _empty_inventory_metrics()

    def test_no_day_column(self) -> None:
        inv = pd.DataFrame({"dealer_inventory": [100, 200]})
        m = _compute_inventory_metrics(inv)
        assert m == _empty_inventory_metrics()

    def test_string_boolean_values(self) -> None:
        """Boolean columns as string 'True'/'False' should work."""
        inv = pd.DataFrame(
            {
                "day": [1, 2],
                "dealer_inventory": [100, 0],
                "is_at_zero": ["False", "True"],
                "hit_vbt_this_step": ["True", "False"],
            }
        )
        m = _compute_inventory_metrics(inv)
        assert m["dealer_active_fraction"] == pytest.approx(0.5)
        assert m["dealer_empty_fraction"] == pytest.approx(0.5)
        assert m["vbt_usage_fraction"] == pytest.approx(0.5)

    def test_all_zero_inventory(self) -> None:
        inv = pd.DataFrame(
            {
                "day": [1, 2],
                "dealer_inventory": [0, 0],
                "is_at_zero": [True, True],
                "hit_vbt_this_step": [False, False],
            }
        )
        m = _compute_inventory_metrics(inv)
        assert m["dealer_active_fraction"] == pytest.approx(0.0)
        assert m["dealer_empty_fraction"] == pytest.approx(1.0)
        assert m["vbt_usage_fraction"] == pytest.approx(0.0)

    def test_missing_optional_columns(self) -> None:
        """Only 'day' column present -> falls back to False for missing cols."""
        inv = pd.DataFrame({"day": [1, 2]})
        m = _compute_inventory_metrics(inv)
        assert m["dealer_active_fraction"] == pytest.approx(0.0)
        assert m["dealer_empty_fraction"] == pytest.approx(0.0)
        assert m["vbt_usage_fraction"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: _compute_system_state_metrics
# ---------------------------------------------------------------------------


class TestComputeSystemStateMetrics:
    """Tests for _compute_system_state_metrics."""

    def test_basic(self) -> None:
        sys_ts = pd.DataFrame(
            {
                "debt_to_money": [2.0, 1.5, 1.0],
                "total_face_value": [10000, 8000, 6000],
            }
        )
        m = _compute_system_state_metrics(sys_ts)
        assert m["mean_debt_to_money"] == pytest.approx(1.5)
        assert m["final_debt_to_money"] == pytest.approx(1.0)
        # debt_shrink_rate = (10000 - 6000) / 10000 = 0.4
        assert m["debt_shrink_rate"] == pytest.approx(0.4)

    def test_empty_dataframe(self) -> None:
        m = _compute_system_state_metrics(pd.DataFrame())
        assert m == _empty_system_state_metrics()

    def test_no_debt_to_money_column(self) -> None:
        sys_ts = pd.DataFrame({"total_face_value": [10000, 6000]})
        m = _compute_system_state_metrics(sys_ts)
        assert m["mean_debt_to_money"] is None
        assert m["final_debt_to_money"] is None
        assert m["debt_shrink_rate"] == pytest.approx(0.4)

    def test_no_total_face_value_column(self) -> None:
        sys_ts = pd.DataFrame({"debt_to_money": [2.0, 1.0]})
        m = _compute_system_state_metrics(sys_ts)
        assert m["mean_debt_to_money"] == pytest.approx(1.5)
        assert m["final_debt_to_money"] == pytest.approx(1.0)
        assert m["debt_shrink_rate"] is None

    def test_zero_initial_face_value(self) -> None:
        """total_face_value starting at 0 -> debt_shrink_rate=0."""
        sys_ts = pd.DataFrame({"total_face_value": [0, 0]})
        m = _compute_system_state_metrics(sys_ts)
        assert m["debt_shrink_rate"] == pytest.approx(0.0)

    def test_single_row(self) -> None:
        sys_ts = pd.DataFrame(
            {"debt_to_money": [3.0], "total_face_value": [5000]}
        )
        m = _compute_system_state_metrics(sys_ts)
        assert m["mean_debt_to_money"] == pytest.approx(3.0)
        assert m["final_debt_to_money"] == pytest.approx(3.0)
        # shrink = (5000 - 5000) / 5000 = 0
        assert m["debt_shrink_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: _compute_repayment_metrics
# ---------------------------------------------------------------------------


class TestComputeRepaymentMetrics:
    """Tests for _compute_repayment_metrics."""

    def test_basic(self) -> None:
        rep = pd.DataFrame(
            {
                "agent_id": ["H1", "H2", "H3", "H4"],
                "face_value": [100, 200, 150, 250],
                "outcome": ["repaid", "defaulted", "repaid", "defaulted"],
                "buy_count": [1, 0, 0, 1],
                "sell_count": [1, 0, 0, 0],
            }
        )
        m = _compute_repayment_metrics(rep)

        # defaulted: H2 (face=200, not traded), H4 (face=250, traded)
        # frac_defaulted_that_traded = 250 / (200+250) = 250/450
        assert m["frac_defaulted_that_traded"] == pytest.approx(250.0 / 450.0)

        # repaid: H1 (face=100, traded), H3 (face=150, not traded)
        # frac_repaid_that_traded = 100 / (100+150) = 100/250
        assert m["frac_repaid_that_traded"] == pytest.approx(100.0 / 250.0)

    def test_empty_dataframe(self) -> None:
        m = _compute_repayment_metrics(pd.DataFrame())
        assert m == _empty_repayment_metrics()

    def test_missing_buy_sell_columns(self) -> None:
        rep = pd.DataFrame(
            {
                "face_value": [100],
                "outcome": ["repaid"],
            }
        )
        m = _compute_repayment_metrics(rep)
        assert m == _empty_repayment_metrics()

    def test_missing_outcome_column(self) -> None:
        rep = pd.DataFrame(
            {
                "face_value": [100],
                "buy_count": [1],
                "sell_count": [0],
            }
        )
        m = _compute_repayment_metrics(rep)
        assert m == _empty_repayment_metrics()

    def test_no_defaulted_outcomes(self) -> None:
        """All repaid -> frac_defaulted is None (no data)."""
        rep = pd.DataFrame(
            {
                "face_value": [100, 200],
                "outcome": ["repaid", "repaid"],
                "buy_count": [1, 0],
                "sell_count": [0, 0],
            }
        )
        m = _compute_repayment_metrics(rep)
        assert m["frac_defaulted_that_traded"] is None
        assert m["frac_repaid_that_traded"] == pytest.approx(100.0 / 300.0)

    def test_no_repaid_outcomes(self) -> None:
        """All defaulted -> frac_repaid is None."""
        rep = pd.DataFrame(
            {
                "face_value": [100, 200],
                "outcome": ["defaulted", "defaulted"],
                "buy_count": [0, 1],
                "sell_count": [0, 0],
            }
        )
        m = _compute_repayment_metrics(rep)
        assert m["frac_repaid_that_traded"] is None
        assert m["frac_defaulted_that_traded"] == pytest.approx(200.0 / 300.0)

    def test_zero_face_value_outcome(self) -> None:
        """When all face values for an outcome are 0, fraction is 0.0."""
        rep = pd.DataFrame(
            {
                "face_value": [0, 0],
                "outcome": ["repaid", "repaid"],
                "buy_count": [1, 0],
                "sell_count": [0, 0],
            }
        )
        m = _compute_repayment_metrics(rep)
        assert m["frac_repaid_that_traded"] == pytest.approx(0.0)

    def test_non_numeric_face_and_counts(self) -> None:
        """Non-numeric values are coerced to 0."""
        rep = pd.DataFrame(
            {
                "face_value": ["abc", "200"],
                "outcome": ["repaid", "defaulted"],
                "buy_count": ["x", "1"],
                "sell_count": ["0", "0"],
            }
        )
        m = _compute_repayment_metrics(rep)
        # face_value: abc -> 0, 200 -> 200
        # buy_count: x -> 0, 1 -> 1; sell_count: 0, 0
        # defaulted: H2 face=200, used_dealer=(1+0)>0=True
        assert m["frac_defaulted_that_traded"] == pytest.approx(1.0)
        # repaid: H1 face=0, used_dealer=(0+0)>0=False
        assert m["frac_repaid_that_traded"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: _empty_* helper functions
# ---------------------------------------------------------------------------


class TestEmptyMetrics:
    """Verify shapes/keys of all _empty_*_metrics helpers."""

    def test_empty_trade_metrics(self) -> None:
        m = _empty_trade_metrics()
        assert m["dealer_trade_count"] == 0
        assert m["trader_dealer_trade_count"] == 0
        assert m["n_traders_using_dealer"] == 0
        assert m["total_face_traded"] == 0.0
        assert m["total_cash_volume"] == 0.0

    def test_empty_inventory_metrics(self) -> None:
        m = _empty_inventory_metrics()
        assert m["dealer_active_fraction"] is None
        assert m["dealer_empty_fraction"] is None
        assert m["vbt_usage_fraction"] is None

    def test_empty_system_state_metrics(self) -> None:
        m = _empty_system_state_metrics()
        assert m["mean_debt_to_money"] is None
        assert m["final_debt_to_money"] is None
        assert m["debt_shrink_rate"] is None

    def test_empty_repayment_metrics(self) -> None:
        m = _empty_repayment_metrics()
        assert m["frac_defaulted_that_traded"] is None
        assert m["frac_repaid_that_traded"] is None


# ---------------------------------------------------------------------------
# Tests: build_dealer_usage_by_run (integration)
# ---------------------------------------------------------------------------


class TestBuildDealerUsageByRun:
    """Tests for build_dealer_usage_by_run."""

    def test_basic_full_run(self, experiment_root: Path) -> None:
        df = build_dealer_usage_by_run(experiment_root)
        assert len(df) == 1
        row = df.iloc[0]

        # Run params
        assert row["run_id"] == "run_001"
        assert float(row["kappa"]) == 0.5
        assert float(row["trading_effect"]) == 0.1

        # Trade metrics (4 total trades, 3 are H-prefix)
        assert row["dealer_trade_count"] == 4
        assert row["trader_dealer_trade_count"] == 3
        assert row["n_traders_using_dealer"] == 2  # H1, H2
        # H-trader face: 100+200+150 = 450
        assert row["total_face_traded"] == 450.0
        # H-trader price: 90+180+130 = 400
        assert row["total_cash_volume"] == 400.0

        # Inventory metrics
        assert row["dealer_active_fraction"] is not None
        assert row["vbt_usage_fraction"] is not None

        # System state metrics
        assert row["mean_debt_to_money"] == pytest.approx(1.5)
        assert row["final_debt_to_money"] == pytest.approx(1.0)
        assert row["debt_shrink_rate"] == pytest.approx(0.4)

        # Repayment metrics
        assert row["frac_defaulted_that_traded"] is not None
        assert row["frac_repaid_that_traded"] is not None

    def test_missing_comparison_csv(self, tmp_path: Path) -> None:
        df = build_dealer_usage_by_run(tmp_path)
        assert df.empty

    def test_missing_out_directory(self, tmp_path: Path) -> None:
        """Run exists in comparison but has no out/ directory -> skip."""
        _write_comparison_csv(
            tmp_path,
            ["0.5,1,0.5,1,42,1000,run_nodir,run_p,0.3,0.2,0.1,0.33"],
        )
        df = build_dealer_usage_by_run(tmp_path)
        assert df.empty

    def test_blank_active_run_id_raises(self, tmp_path: Path) -> None:
        """Blank active_run_id -> pandas NaN -> TypeError on Path construction."""
        _write_comparison_csv(
            tmp_path,
            ["0.5,1,0.5,1,42,1000,,run_p,0.3,0.2,0.1,0.33"],
        )
        with pytest.raises(TypeError):
            build_dealer_usage_by_run(tmp_path)

    def test_no_optional_csvs(self, tmp_path: Path) -> None:
        """Only comparison.csv and an empty out/ dir -> uses empty metrics."""
        run_id = "run_bare"
        _write_comparison_csv(
            tmp_path,
            [f"0.5,1,0.5,1,42,1000,{run_id},run_p,0.3,0.2,0.1,0.33"],
        )
        out = tmp_path / "active" / "runs" / run_id / "out"
        out.mkdir(parents=True, exist_ok=True)

        df = build_dealer_usage_by_run(tmp_path)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["dealer_trade_count"] == 0
        assert row["dealer_active_fraction"] is None
        assert row["mean_debt_to_money"] is None
        assert row["frac_defaulted_that_traded"] is None

    def test_multiple_runs(self, tmp_path: Path) -> None:
        _write_comparison_csv(
            tmp_path,
            [
                "0.5,1,0.5,1,42,1000,run_a,run_pa,0.3,0.2,0.1,0.33",
                "0.5,1,0.5,1,99,1000,run_b,run_pb,0.4,0.15,0.25,0.625",
            ],
        )
        for rid in ["run_a", "run_b"]:
            _write_csv(
                tmp_path,
                rid,
                "trades.csv",
                "trader_id,face_value,price\nH1,100,90\n",
            )
        df = build_dealer_usage_by_run(tmp_path)
        assert len(df) == 2
        assert set(df["run_id"]) == {"run_a", "run_b"}


# ---------------------------------------------------------------------------
# Tests: run_dealer_usage_analysis (integration)
# ---------------------------------------------------------------------------


class TestRunDealerUsageAnalysis:
    """Tests for run_dealer_usage_analysis."""

    def test_writes_output_file(self, experiment_root: Path) -> None:
        output_path = run_dealer_usage_analysis(experiment_root)
        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 1
        assert output_path.name == "dealer_usage_by_run.csv"

    def test_no_data_returns_empty_path(self, tmp_path: Path) -> None:
        result = run_dealer_usage_analysis(tmp_path)
        assert result == Path()

    def test_creates_aggregate_dir(self, tmp_path: Path) -> None:
        """Aggregate directory is created if it doesn't already exist."""
        run_id = "run_mk"
        _write_comparison_csv(
            tmp_path,
            [f"0.5,1,0.5,1,42,1000,{run_id},run_p,0.3,0.2,0.1,0.33"],
        )
        out = tmp_path / "active" / "runs" / run_id / "out"
        out.mkdir(parents=True, exist_ok=True)
        _write_csv(
            tmp_path,
            run_id,
            "trades.csv",
            "trader_id,face_value,price\nH1,100,90\n",
        )
        output_path = run_dealer_usage_analysis(tmp_path)
        assert output_path.exists()
