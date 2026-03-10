"""Tests for experiment runner coverage: execution, result building, and output writing.

Targets uncovered code in:
- balanced_comparison.py: _build_result_from_summaries, _write_summary_json,
  _write_stats_analysis, _write_activity_analysis, _run_all_sequential, run_all
- bank_comparison.py: _build_result_from_summaries, _write_summary_json,
  _write_stats_analysis, _write_activity_analysis, _run_all_sequential, run_all
- nbfi_comparison.py: _build_result_from_summaries, _write_summary_json,
  _write_stats_analysis, _write_activity_analysis, _run_all_sequential, run_all
- ring.py: _prepare_run path (via _execute_run), run_frontier, run_lhs,
  _decimal_list, _to_yaml_ready, config validators, _finalize_run,
  _rel_path, _artifact_loader_for_result, _liquidity_allocation_dict,
  load_ring_sweep_config
- comparison.py: _run_pair, _write_comparison_csv, _write_summary_json, run_all
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bilancio.experiments.ring import RingRunSummary


# =============================================================================
# Helpers
# =============================================================================


def _make_summary(
    *,
    run_id: str = "test_run_001",
    phase: str = "test",
    kappa: Decimal = Decimal("1"),
    concentration: Decimal = Decimal("1"),
    mu: Decimal = Decimal("0"),
    monotonicity: Decimal = Decimal("0"),
    delta_total: Decimal | None = Decimal("0.30"),
    phi_total: Decimal | None = Decimal("0.70"),
    time_to_stability: int = 5,
    n_defaults: int = 3,
    cascade_fraction: Decimal | None = Decimal("0.15"),
    cb_loans_created_count: int = 2,
    cb_interest_total_paid: int = 100,
    cb_loans_outstanding_pre_final: int = 1,
    bank_defaults_final: int = 0,
    cb_reserve_destruction_pct: float = 0.02,
    delta_bank: float | None = 0.05,
    deposit_loss_gross: int = 50,
    deposit_loss_pct: float | None = 0.01,
    payable_default_loss: int = 200,
    total_loss: int = 250,
    total_loss_pct: float | None = 0.025,
    intermediary_loss_total: float = 30.0,
    nbfi_loans_created: int = 0,
    initial_intermediary_capital: float = 500.0,
    S_total: float = 10000.0,
    dealer_metrics: dict | None = None,
    modal_call_id: str | None = None,
) -> RingRunSummary:
    """Create a RingRunSummary with rich defaults for testing."""
    return RingRunSummary(
        run_id=run_id,
        phase=phase,
        kappa=kappa,
        concentration=concentration,
        mu=mu,
        monotonicity=monotonicity,
        delta_total=delta_total,
        phi_total=phi_total,
        time_to_stability=time_to_stability,
        n_defaults=n_defaults,
        cascade_fraction=cascade_fraction,
        cb_loans_created_count=cb_loans_created_count,
        cb_interest_total_paid=cb_interest_total_paid,
        cb_loans_outstanding_pre_final=cb_loans_outstanding_pre_final,
        bank_defaults_final=bank_defaults_final,
        cb_reserve_destruction_pct=cb_reserve_destruction_pct,
        delta_bank=delta_bank,
        deposit_loss_gross=deposit_loss_gross,
        deposit_loss_pct=deposit_loss_pct,
        payable_default_loss=payable_default_loss,
        total_loss=total_loss,
        total_loss_pct=total_loss_pct,
        intermediary_loss_total=intermediary_loss_total,
        nbfi_loans_created=nbfi_loans_created,
        initial_intermediary_capital=initial_intermediary_capital,
        S_total=S_total,
        dealer_metrics=dealer_metrics,
        modal_call_id=modal_call_id,
    )


def _make_failed_summary(*, run_id: str = "failed_001", phase: str = "test") -> RingRunSummary:
    """Create a failed RingRunSummary (delta_total=None)."""
    return RingRunSummary(
        run_id=run_id,
        phase=phase,
        kappa=Decimal("1"),
        concentration=Decimal("1"),
        mu=Decimal("0"),
        monotonicity=Decimal("0"),
        delta_total=None,
        phi_total=None,
        time_to_stability=0,
    )


# =============================================================================
# 1. BankComparisonRunner._build_result_from_summaries
# =============================================================================


class TestBankBuildResultFromSummaries:
    """Tests for BankComparisonRunner._build_result_from_summaries."""

    def test_builds_result_from_completed_summaries(self, tmp_path: Path):
        """_build_result_from_summaries constructs result from two completed arm summaries."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig(kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")])
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        idle = _make_summary(run_id="idle_001", phase="bank_idle", delta_total=Decimal("0.50"))
        lend = _make_summary(run_id="lend_001", phase="bank_lend", delta_total=Decimal("0.30"))

        result = runner._build_result_from_summaries(
            arm_summaries={"idle": idle, "lend": lend},
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0"),
            monotonicity=Decimal("0"),
            outside_mid_ratio=Decimal("0.90"),
            seed=42,
        )

        assert result.kappa == Decimal("1")
        assert result.delta_idle == Decimal("0.50")
        assert result.delta_lend == Decimal("0.30")
        assert result.idle_status == "completed"
        assert result.lend_status == "completed"
        assert result.idle_run_id == "idle_001"
        assert result.lend_run_id == "lend_001"
        assert result.bank_lending_effect == Decimal("0.20")
        assert result.n_defaults_idle == 3
        assert result.n_defaults_lend == 3

    def test_builds_result_with_failed_arm(self, tmp_path: Path):
        """_build_result_from_summaries handles a failed arm (delta_total=None)."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        idle = _make_summary(run_id="idle_f", phase="bank_idle", delta_total=Decimal("0.40"))
        lend = _make_failed_summary(run_id="lend_f", phase="bank_lend")

        result = runner._build_result_from_summaries(
            arm_summaries={"idle": idle, "lend": lend},
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0"),
            monotonicity=Decimal("0"),
            outside_mid_ratio=Decimal("0.90"),
            seed=42,
        )

        assert result.idle_status == "completed"
        assert result.lend_status == "failed"
        assert result.bank_lending_effect is None

    def test_system_loss_computation(self, tmp_path: Path):
        """_build_result_from_summaries computes system_loss = total_loss + intermediary_loss."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        idle = _make_summary(run_id="idle_s", total_loss=200, intermediary_loss_total=50.0, S_total=10000.0)
        lend = _make_summary(run_id="lend_s", total_loss=100, intermediary_loss_total=20.0, S_total=10000.0)

        result = runner._build_result_from_summaries(
            arm_summaries={"idle": idle, "lend": lend},
            kappa=Decimal("1"), concentration=Decimal("1"),
            mu=Decimal("0"), monotonicity=Decimal("0"),
            outside_mid_ratio=Decimal("0.90"), seed=42,
        )

        assert result.system_loss_idle == pytest.approx(250.0)
        assert result.system_loss_lend == pytest.approx(120.0)
        assert result.system_loss_pct_idle == pytest.approx(0.025)
        assert result.system_loss_pct_lend == pytest.approx(0.012)


# =============================================================================
# 2. BankComparisonRunner._write_summary_json
# =============================================================================


class TestBankWriteSummaryJson:
    """Tests for BankComparisonRunner._write_summary_json."""

    def test_writes_summary_with_results(self, tmp_path: Path):
        """_write_summary_json writes valid JSON with summary statistics."""
        from bilancio.experiments.bank_comparison import (
            BankComparisonConfig,
            BankComparisonResult,
            BankComparisonRunner,
        )

        config = BankComparisonConfig(kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")])
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            BankComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=42, outside_mid_ratio=Decimal("0.90"),
                delta_idle=Decimal("0.50"), phi_idle=Decimal("0.70"),
                idle_run_id="idle_1", idle_status="completed",
                delta_lend=Decimal("0.30"), phi_lend=Decimal("0.80"),
                lend_run_id="lend_1", lend_status="completed",
                cb_loans_created_idle=5, cb_loans_created_lend=8,
                bank_defaults_final_idle=0, bank_defaults_final_lend=1,
            ),
        ]

        runner._write_summary_json()

        summary_path = tmp_path / "aggregate" / "summary.json"
        assert summary_path.exists()

        with summary_path.open() as fh:
            data = json.load(fh)

        assert data["total_combos"] == 1
        assert data["completed_combos"] == 1
        assert data["mean_delta_idle"] == pytest.approx(0.50)
        assert data["mean_delta_lend"] == pytest.approx(0.30)
        assert data["mean_bank_lending_effect"] == pytest.approx(0.20)
        assert data["combos_improved"] == 1
        assert data["combos_worsened"] == 0
        assert data["config"]["n_agents"] == 100

    def test_writes_summary_with_no_completed_results(self, tmp_path: Path):
        """_write_summary_json handles empty or all-failed results."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = []

        runner._write_summary_json()

        summary_path = tmp_path / "aggregate" / "summary.json"
        assert summary_path.exists()

        with summary_path.open() as fh:
            data = json.load(fh)

        assert data["total_combos"] == 0
        assert data["completed_combos"] == 0
        assert data["mean_delta_idle"] is None
        assert data["combos_improved"] == 0

    def test_skips_when_skip_local_processing(self, tmp_path: Path):
        """_write_summary_json is skipped in cloud-only mode."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.skip_local_processing = True

        runner._write_summary_json()

        summary_path = tmp_path / "aggregate" / "summary.json"
        assert not summary_path.exists()


# =============================================================================
# 3. BankComparisonRunner._write_stats_analysis and _write_activity_analysis
# =============================================================================


class TestBankWriteAnalyses:
    """Tests for _write_stats_analysis and _write_activity_analysis."""

    def test_stats_skips_when_no_results(self, tmp_path: Path):
        """_write_stats_analysis returns early when no results."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = []

        # Should not raise
        runner._write_stats_analysis()

    def test_stats_skips_when_skip_local_processing(self, tmp_path: Path):
        """_write_stats_analysis skips in cloud-only mode."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.skip_local_processing = True
        runner.comparison_results = [MagicMock()]

        runner._write_stats_analysis()
        # No exception = passes

    def test_activity_skips_when_no_results(self, tmp_path: Path):
        """_write_activity_analysis returns early when no results."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = []

        runner._write_activity_analysis()

    def test_activity_skips_when_skip_local_processing(self, tmp_path: Path):
        """_write_activity_analysis skips in cloud-only mode."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.skip_local_processing = True
        runner.comparison_results = [MagicMock()]

        runner._write_activity_analysis()


# =============================================================================
# 4. NBFIComparisonRunner._build_result_from_summaries
# =============================================================================


class TestNBFIBuildResultFromSummaries:
    """Tests for NBFIComparisonRunner._build_result_from_summaries."""

    def test_builds_result_from_completed_summaries(self, tmp_path: Path):
        """_build_result_from_summaries constructs result from two completed arm summaries."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig(kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")])
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        idle = _make_summary(run_id="nbfi_idle_001", phase="nbfi_idle", delta_total=Decimal("0.45"))
        lend = _make_summary(run_id="nbfi_lend_001", phase="nbfi_lend", delta_total=Decimal("0.25"))

        result = runner._build_result_from_summaries(
            arm_summaries={"idle": idle, "lend": lend},
            kappa=Decimal("1"), concentration=Decimal("1"),
            mu=Decimal("0"), monotonicity=Decimal("0"),
            outside_mid_ratio=Decimal("0.90"), seed=42,
        )

        assert result.delta_idle == Decimal("0.45")
        assert result.delta_lend == Decimal("0.25")
        assert result.idle_status == "completed"
        assert result.lend_status == "completed"
        assert result.lending_effect == Decimal("0.20")

    def test_builds_result_with_failed_arm(self, tmp_path: Path):
        """_build_result_from_summaries handles a failed arm."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        idle = _make_failed_summary(run_id="idle_f", phase="nbfi_idle")
        lend = _make_summary(run_id="lend_ok", phase="nbfi_lend", delta_total=Decimal("0.30"))

        result = runner._build_result_from_summaries(
            arm_summaries={"idle": idle, "lend": lend},
            kappa=Decimal("1"), concentration=Decimal("1"),
            mu=Decimal("0"), monotonicity=Decimal("0"),
            outside_mid_ratio=Decimal("0.90"), seed=42,
        )

        assert result.idle_status == "failed"
        assert result.lend_status == "completed"
        assert result.lending_effect is None

    def test_system_loss_and_intermediary_pct(self, tmp_path: Path):
        """_build_result_from_summaries computes system_loss and intermediary loss pct."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        idle = _make_summary(run_id="idle_s", total_loss=300, intermediary_loss_total=100.0, S_total=10000.0)
        lend = _make_summary(run_id="lend_s", total_loss=150, intermediary_loss_total=50.0, S_total=10000.0)

        result = runner._build_result_from_summaries(
            arm_summaries={"idle": idle, "lend": lend},
            kappa=Decimal("1"), concentration=Decimal("1"),
            mu=Decimal("0"), monotonicity=Decimal("0"),
            outside_mid_ratio=Decimal("0.90"), seed=42,
        )

        assert result.system_loss_idle == pytest.approx(400.0)
        assert result.system_loss_lend == pytest.approx(200.0)
        assert result.intermediary_loss_pct_idle == pytest.approx(0.01)
        assert result.intermediary_loss_pct_lend == pytest.approx(0.005)


# =============================================================================
# 5. NBFIComparisonRunner._write_summary_json
# =============================================================================


class TestNBFIWriteSummaryJson:
    """Tests for NBFIComparisonRunner._write_summary_json."""

    def test_writes_summary_with_results(self, tmp_path: Path):
        """_write_summary_json writes valid JSON."""
        from bilancio.experiments.nbfi_comparison import (
            NBFIComparisonConfig,
            NBFIComparisonResult,
            NBFIComparisonRunner,
        )

        config = NBFIComparisonConfig(kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")])
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            NBFIComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=42, outside_mid_ratio=Decimal("0.90"),
                delta_idle=Decimal("0.40"), phi_idle=Decimal("0.60"),
                idle_run_id="idle_1", idle_status="completed",
                delta_lend=Decimal("0.20"), phi_lend=Decimal("0.80"),
                lend_run_id="lend_1", lend_status="completed",
            ),
        ]

        runner._write_summary_json()

        summary_path = tmp_path / "aggregate" / "summary.json"
        assert summary_path.exists()

        with summary_path.open() as fh:
            data = json.load(fh)

        assert data["total_combos"] == 1
        assert data["completed_combos"] == 1
        assert data["mean_lending_effect"] == pytest.approx(0.20)
        assert data["combos_improved"] == 1

    def test_writes_summary_empty(self, tmp_path: Path):
        """_write_summary_json handles empty results."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = []

        runner._write_summary_json()

        summary_path = tmp_path / "aggregate" / "summary.json"
        with summary_path.open() as fh:
            data = json.load(fh)

        assert data["mean_lending_effect"] is None
        assert data["combos_improved"] == 0

    def test_skips_when_skip_local_processing(self, tmp_path: Path):
        """_write_summary_json skips in cloud-only mode."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.skip_local_processing = True

        runner._write_summary_json()

        assert not (tmp_path / "aggregate" / "summary.json").exists()


# =============================================================================
# 6. NBFIComparisonRunner._write_stats_analysis and _write_activity_analysis
# =============================================================================


class TestNBFIWriteAnalyses:
    """Tests for NBFI _write_stats_analysis and _write_activity_analysis."""

    def test_stats_skips_when_no_results(self, tmp_path: Path):
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = []
        runner._write_stats_analysis()

    def test_stats_skips_when_cloud_only(self, tmp_path: Path):
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.skip_local_processing = True
        runner.comparison_results = [MagicMock()]
        runner._write_stats_analysis()

    def test_activity_skips_when_no_results(self, tmp_path: Path):
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = []
        runner._write_activity_analysis()

    def test_activity_skips_when_cloud_only(self, tmp_path: Path):
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.skip_local_processing = True
        runner.comparison_results = [MagicMock()]
        runner._write_activity_analysis()


# =============================================================================
# 7. BalancedComparisonRunner._build_result_from_summaries
# =============================================================================


class TestBalancedBuildResultFromSummaries:
    """Tests for BalancedComparisonRunner._build_result_from_summaries."""

    def test_builds_result_passive_active(self, tmp_path: Path):
        """_build_result_from_summaries with passive and active arms."""
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        passive = _make_summary(
            run_id="passive_001", phase="balanced_passive",
            delta_total=Decimal("0.50"), dealer_metrics={"dealer_total_pnl": -10.0},
        )
        active = _make_summary(
            run_id="active_001", phase="balanced_active",
            delta_total=Decimal("0.30"),
            dealer_metrics={"dealer_total_pnl": 50.0, "dealer_total_return": 0.10, "total_trades": 15},
        )

        result = runner._build_result_from_summaries(
            arm_summaries={"passive": passive, "active": active},
            kappa=Decimal("1"), concentration=Decimal("1"),
            mu=Decimal("0"), monotonicity=Decimal("0"),
            outside_mid_ratio=Decimal("0.90"), seed=42,
        )

        assert result.delta_passive == Decimal("0.50")
        assert result.delta_active == Decimal("0.30")
        assert result.trading_effect == Decimal("0.20")
        assert result.passive_status == "completed"
        assert result.active_status == "completed"
        assert result.dealer_total_pnl == 50.0
        assert result.total_trades == 15

    def test_builds_result_with_lender_arm(self, tmp_path: Path):
        """_build_result_from_summaries includes lender arm data when present."""
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
            enable_lender=True,
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        passive = _make_summary(run_id="p", phase="passive", delta_total=Decimal("0.40"))
        active = _make_summary(run_id="a", phase="active", delta_total=Decimal("0.30"))
        lender = _make_summary(
            run_id="l",
            phase="lender",
            delta_total=Decimal("0.25"),
            nbfi_loans_created=7,
        )

        result = runner._build_result_from_summaries(
            arm_summaries={"passive": passive, "active": active, "lender": lender},
            kappa=Decimal("1"), concentration=Decimal("1"),
            mu=Decimal("0"), monotonicity=Decimal("0"),
            outside_mid_ratio=Decimal("0.90"), seed=42,
        )

        assert result.delta_lender == Decimal("0.25")
        assert result.lender_status == "completed"
        assert result.lending_effect == Decimal("0.15")
        assert result.total_loans == 7


# =============================================================================
# 8. BalancedComparisonRunner._write_summary_json
# =============================================================================


class TestBalancedWriteSummaryJson:
    """Tests for BalancedComparisonRunner._write_summary_json."""

    def test_writes_summary_with_results(self, tmp_path: Path):
        """_write_summary_json writes valid JSON with trading effect statistics."""
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonResult,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            BalancedComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=42,
                face_value=Decimal("20"), outside_mid_ratio=Decimal("0.90"),
                big_entity_share=Decimal("0.25"),
                delta_passive=Decimal("0.50"), phi_passive=Decimal("0.50"),
                passive_run_id="p1", passive_status="completed",
                delta_active=Decimal("0.30"), phi_active=Decimal("0.70"),
                active_run_id="a1", active_status="completed",
                n_defaults_passive=5, n_defaults_active=3,
                cascade_fraction_passive=Decimal("0.10"),
                cascade_fraction_active=Decimal("0.05"),
            ),
        ]

        runner._write_summary_json()

        summary_path = tmp_path / "aggregate" / "summary.json"
        assert summary_path.exists()

        with summary_path.open() as fh:
            data = json.load(fh)

        assert data["total_pairs"] == 1
        assert data["completed_pairs"] == 1
        assert data["mean_delta_passive"] == pytest.approx(0.50)
        assert data["mean_delta_active"] == pytest.approx(0.30)
        assert data["mean_trading_effect"] == pytest.approx(0.20)
        assert data["pairs_with_improvement"] == 1
        assert data["mean_n_defaults_passive"] == pytest.approx(5.0)
        assert data["mean_cascade_fraction_active"] == pytest.approx(0.05)

    def test_writes_summary_empty(self, tmp_path: Path):
        """_write_summary_json handles zero results."""
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = []

        runner._write_summary_json()

        summary_path = tmp_path / "aggregate" / "summary.json"
        with summary_path.open() as fh:
            data = json.load(fh)

        assert data["total_pairs"] == 0
        assert data["mean_trading_effect"] is None

    def test_skips_when_skip_local_processing(self, tmp_path: Path):
        """_write_summary_json skips in cloud-only mode."""
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.skip_local_processing = True

        runner._write_summary_json()

        assert not (tmp_path / "aggregate" / "summary.json").exists()


# =============================================================================
# 9. BalancedComparisonRunner._write_stats_analysis and _write_activity_analysis
# =============================================================================


class TestBalancedWriteAnalyses:
    """Tests for balanced _write_stats_analysis and _write_activity_analysis."""

    def test_stats_skips_when_no_results(self, tmp_path: Path):
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = []
        runner._write_stats_analysis()

    def test_stats_skips_when_cloud_only(self, tmp_path: Path):
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.skip_local_processing = True
        runner.comparison_results = [MagicMock()]
        runner._write_stats_analysis()

    def test_activity_skips_when_no_results(self, tmp_path: Path):
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = []
        runner._write_activity_analysis()

    def test_activity_skips_when_cloud_only(self, tmp_path: Path):
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.skip_local_processing = True
        runner.comparison_results = [MagicMock()]
        runner._write_activity_analysis()


# =============================================================================
# 10. ComparisonSweepRunner._write_summary_json and _write_comparison_csv
# =============================================================================


class TestComparisonSweepWriters:
    """Tests for comparison.py ComparisonSweepRunner output writing."""

    def test_write_comparison_csv(self, tmp_path: Path):
        """_write_comparison_csv writes valid CSV with all fields."""
        from bilancio.experiments.comparison import (
            ComparisonResult,
            ComparisonSweepConfig,
            ComparisonSweepRunner,
        )

        config = ComparisonSweepConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")],
        )
        runner = ComparisonSweepRunner(config, tmp_path)

        runner.comparison_results = [
            ComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"),
                mu=Decimal("0"), monotonicity=Decimal("0"), seed=42,
                delta_control=Decimal("0.50"), phi_control=Decimal("0.50"),
                control_run_id="c1", control_status="completed",
                delta_treatment=Decimal("0.30"), phi_treatment=Decimal("0.70"),
                treatment_run_id="t1", treatment_status="completed",
                dealer_total_pnl=10.5, total_trades=7,
            ),
        ]

        runner._write_comparison_csv()

        import csv
        csv_path = tmp_path / "aggregate" / "comparison.csv"
        assert csv_path.exists()

        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["kappa"] == "1"
        assert rows[0]["delta_control"] == "0.50"
        assert rows[0]["delta_treatment"] == "0.30"
        assert rows[0]["delta_reduction"] == "0.20"
        assert rows[0]["dealer_total_pnl"] == "10.5"

    def test_write_summary_json(self, tmp_path: Path):
        """_write_summary_json writes valid JSON with statistics."""
        from bilancio.experiments.comparison import (
            ComparisonResult,
            ComparisonSweepConfig,
            ComparisonSweepRunner,
        )

        config = ComparisonSweepConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")],
        )
        runner = ComparisonSweepRunner(config, tmp_path)

        runner.comparison_results = [
            ComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"),
                mu=Decimal("0"), monotonicity=Decimal("0"), seed=42,
                delta_control=Decimal("0.50"), phi_control=Decimal("0.50"),
                control_run_id="c1", control_status="completed",
                delta_treatment=Decimal("0.30"), phi_treatment=Decimal("0.70"),
                treatment_run_id="t1", treatment_status="completed",
            ),
        ]

        runner._write_summary_json()

        summary_path = tmp_path / "aggregate" / "summary.json"
        assert summary_path.exists()

        with summary_path.open() as fh:
            data = json.load(fh)

        assert data["total_pairs"] == 1
        assert data["completed_pairs"] == 1
        assert data["mean_delta_control"] == pytest.approx(0.50)
        assert data["mean_delta_treatment"] == pytest.approx(0.30)
        assert data["pairs_with_improvement"] == 1

    def test_write_summary_json_empty(self, tmp_path: Path):
        """_write_summary_json handles empty results."""
        from bilancio.experiments.comparison import ComparisonSweepConfig, ComparisonSweepRunner

        config = ComparisonSweepConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")],
        )
        runner = ComparisonSweepRunner(config, tmp_path)
        runner.comparison_results = []

        runner._write_summary_json()

        summary_path = tmp_path / "aggregate" / "summary.json"
        with summary_path.open() as fh:
            data = json.load(fh)

        assert data["mean_delta_control"] is None
        assert data["pairs_with_improvement"] == 0

    def test_write_comparison_csv_handles_none_dealer_metrics(self, tmp_path: Path):
        """_write_comparison_csv handles None values in dealer metrics."""
        from bilancio.experiments.comparison import (
            ComparisonResult,
            ComparisonSweepConfig,
            ComparisonSweepRunner,
        )

        config = ComparisonSweepConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")],
        )
        runner = ComparisonSweepRunner(config, tmp_path)

        runner.comparison_results = [
            ComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"),
                mu=Decimal("0"), monotonicity=Decimal("0"), seed=42,
                delta_control=None, phi_control=None,
                control_run_id="c1", control_status="failed",
                delta_treatment=None, phi_treatment=None,
                treatment_run_id="t1", treatment_status="failed",
            ),
        ]

        runner._write_comparison_csv()

        import csv
        csv_path = tmp_path / "aggregate" / "comparison.csv"
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert rows[0]["delta_control"] == ""
        assert rows[0]["delta_treatment"] == ""
        assert rows[0]["delta_reduction"] == ""
        assert rows[0]["dealer_total_pnl"] == ""


# =============================================================================
# 11. BalancedComparisonRunner._get_enabled_arm_defs
# =============================================================================


class TestBalancedGetEnabledArmDefs:
    """Tests for _get_enabled_arm_defs with various arm flags."""

    def test_default_has_passive_and_active(self, tmp_path: Path):
        """Default config enables only passive and active arms."""
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        arms = runner._get_enabled_arm_defs()
        names = [a[0] for a in arms]
        assert names == ["passive", "active"]

    def test_all_arms_enabled(self, tmp_path: Path):
        """When all arm flags are True, all 7 arms are enabled."""
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
            enable_lender=True,
            enable_dealer_lender=True,
            enable_bank_passive=True,
            enable_bank_dealer=True,
            enable_bank_dealer_nbfi=True,
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        arms = runner._get_enabled_arm_defs()
        names = [a[0] for a in arms]
        assert "passive" in names
        assert "active" in names
        assert "lender" in names
        assert "dealer_lender" in names
        assert "bank_passive" in names
        assert "bank_dealer" in names
        assert "bank_dealer_nbfi" in names
        assert len(names) == 7


# =============================================================================
# 12. run_all dispatches to sequential or batch
# =============================================================================


class TestRunAllDispatch:
    """Tests for run_all dispatching to _run_all_sequential vs _run_all_batch."""

    def test_bank_run_all_dispatches_to_sequential(self, tmp_path: Path):
        """BankComparisonRunner.run_all calls _run_all_sequential when executor lacks execute_batch."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")],
        )
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        with patch.object(runner, "_run_all_sequential", return_value=[]) as mock_seq:
            result = runner.run_all()

        mock_seq.assert_called_once()
        assert result == []

    def test_nbfi_run_all_dispatches_to_sequential(self, tmp_path: Path):
        """NBFIComparisonRunner.run_all calls _run_all_sequential when executor lacks execute_batch."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")],
        )
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        with patch.object(runner, "_run_all_sequential", return_value=[]) as mock_seq:
            result = runner.run_all()

        mock_seq.assert_called_once()
        assert result == []

    def test_balanced_run_all_dispatches_to_sequential(self, tmp_path: Path):
        """BalancedComparisonRunner.run_all calls _run_all_sequential when executor lacks execute_batch."""
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        with patch.object(runner, "_run_all_sequential", return_value=[]) as mock_seq:
            result = runner.run_all()

        mock_seq.assert_called_once()
        assert result == []

    def test_bank_run_all_dispatches_to_batch(self, tmp_path: Path):
        """BankComparisonRunner.run_all calls _run_all_batch when executor has execute_batch."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")],
        )
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        # Give the executor a fake execute_batch method
        runner.executor.execute_batch = MagicMock()  # type: ignore[attr-defined]

        with patch.object(runner, "_run_all_batch", return_value=[]) as mock_batch:
            result = runner.run_all()

        mock_batch.assert_called_once()
        assert result == []


# =============================================================================
# 13. RingSweepRunner._prepare_run via run_grid
# =============================================================================


class TestRingPrepareAndFinalize:
    """Tests for RingSweepRunner _prepare_run flow through run_grid."""

    def test_run_grid_with_mocked_execute_tracks_seeds(self, tmp_path: Path):
        """run_grid increments seed for each run."""
        from bilancio.experiments.ring import RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=10,
        )

        seeds_used = []

        def mock_execute(phase, kappa, concentration, mu, monotonicity, seed, **kwargs):
            seeds_used.append(seed)
            return _make_summary(run_id=f"test_{seed}", phase=phase, kappa=kappa,
                                 concentration=concentration, mu=mu, monotonicity=monotonicity)

        with patch.object(runner, "_execute_run", mock_execute):
            summaries = runner.run_grid(
                [Decimal("1"), Decimal("2")], [Decimal("1")],
                [Decimal("0")], [Decimal("0")],
            )

        assert len(summaries) == 2
        assert seeds_used == [10, 11]


# =============================================================================
# 14. ComparisonSweepRunner run_all with mocked execution
# =============================================================================


class TestComparisonSweepRunAll:
    """Tests for comparison.py ComparisonSweepRunner.run_all with mocked runners."""

    def test_run_all_invokes_run_pair_for_each_combo(self, tmp_path: Path):
        """run_all calls _run_pair for each kappa x concentration x mu combination."""
        from bilancio.experiments.comparison import (
            ComparisonResult,
            ComparisonSweepConfig,
            ComparisonSweepRunner,
        )

        config = ComparisonSweepConfig(
            kappas=[Decimal("1"), Decimal("2")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
        )
        runner = ComparisonSweepRunner(config, tmp_path)

        pair_calls = []

        def mock_run_pair(kappa, concentration, mu, monotonicity):
            pair_calls.append((kappa, concentration, mu))
            return ComparisonResult(
                kappa=kappa, concentration=concentration,
                mu=mu, monotonicity=monotonicity, seed=len(pair_calls),
                delta_control=Decimal("0.40"), phi_control=Decimal("0.60"),
                control_run_id=f"c{len(pair_calls)}", control_status="completed",
                delta_treatment=Decimal("0.20"), phi_treatment=Decimal("0.80"),
                treatment_run_id=f"t{len(pair_calls)}", treatment_status="completed",
            )

        with patch.object(runner, "_run_pair", mock_run_pair):
            results = runner.run_all()

        assert len(pair_calls) == 2
        assert len(results) == 2
        # Verify CSV was written
        csv_path = tmp_path / "aggregate" / "comparison.csv"
        assert csv_path.exists()
        # Verify summary JSON was written
        summary_path = tmp_path / "aggregate" / "summary.json"
        assert summary_path.exists()


# =============================================================================
# 15. _load_existing_results / resumption
# =============================================================================


class TestLoadExistingResults:
    """Tests for resumption from existing comparison CSV."""

    def test_bank_loads_existing_csv(self, tmp_path: Path):
        """BankComparisonRunner loads existing CSV and advances seed counter."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        # Pre-create a comparison CSV
        agg_dir = tmp_path / "aggregate"
        agg_dir.mkdir(parents=True)
        csv_path = agg_dir / "comparison.csv"

        import csv
        fields = BankComparisonRunner.COMPARISON_FIELDS
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            row = {f: "" for f in fields}
            row.update({
                "kappa": "1", "concentration": "1", "mu": "0",
                "monotonicity": "0", "outside_mid_ratio": "0.90",
                "seed": "50",
                "delta_idle": "0.30", "delta_lend": "0.20",
                "bank_lending_effect": "0.10",
                "idle_run_id": "idle_x", "idle_status": "completed",
                "lend_run_id": "lend_x", "lend_status": "completed",
            })
            writer.writerow(row)

        config = BankComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")],
        )
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        # Runner should have advanced seed past 50
        assert runner.seed_counter >= 51
        # And tracked completed counts
        assert len(runner._completed_counts) > 0


# =============================================================================
# 16. Summary JSON with worsened results
# =============================================================================


class TestSummaryJsonWithMixedResults:
    """Tests for summary JSON with improved/unchanged/worsened results."""

    def test_bank_summary_counts_worsened(self, tmp_path: Path):
        """_write_summary_json correctly counts worsened results."""
        from bilancio.experiments.bank_comparison import (
            BankComparisonConfig,
            BankComparisonResult,
            BankComparisonRunner,
        )

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            # Improved: lending reduced defaults
            BankComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=1, outside_mid_ratio=Decimal("0.90"),
                delta_idle=Decimal("0.50"), phi_idle=Decimal("0.50"),
                idle_run_id="i1", idle_status="completed",
                delta_lend=Decimal("0.30"), phi_lend=Decimal("0.70"),
                lend_run_id="l1", lend_status="completed",
            ),
            # Worsened: lending increased defaults
            BankComparisonResult(
                kappa=Decimal("2"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=2, outside_mid_ratio=Decimal("0.90"),
                delta_idle=Decimal("0.20"), phi_idle=Decimal("0.80"),
                idle_run_id="i2", idle_status="completed",
                delta_lend=Decimal("0.40"), phi_lend=Decimal("0.60"),
                lend_run_id="l2", lend_status="completed",
            ),
            # Unchanged: no difference
            BankComparisonResult(
                kappa=Decimal("4"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=3, outside_mid_ratio=Decimal("0.90"),
                delta_idle=Decimal("0.10"), phi_idle=Decimal("0.90"),
                idle_run_id="i3", idle_status="completed",
                delta_lend=Decimal("0.10"), phi_lend=Decimal("0.90"),
                lend_run_id="l3", lend_status="completed",
            ),
        ]

        runner._write_summary_json()

        summary_path = tmp_path / "aggregate" / "summary.json"
        with summary_path.open() as fh:
            data = json.load(fh)

        assert data["combos_improved"] == 1
        assert data["combos_worsened"] == 1
        assert data["combos_unchanged"] == 1
        assert data["total_combos"] == 3
        assert data["completed_combos"] == 3


# =============================================================================
# 17. Ring utility functions: _decimal_list, _to_yaml_ready
# =============================================================================


class TestRingUtilityFunctions:
    """Tests for ring.py utility functions."""

    def test_decimal_list_basic(self):
        """_decimal_list parses comma-separated decimals."""
        from bilancio.experiments.ring import _decimal_list

        result = _decimal_list("0.25, 0.5, 1, 2, 4")
        assert result == [Decimal("0.25"), Decimal("0.5"), Decimal("1"), Decimal("2"), Decimal("4")]

    def test_decimal_list_empty_parts(self):
        """_decimal_list skips empty parts from trailing commas."""
        from bilancio.experiments.ring import _decimal_list

        result = _decimal_list("1, 2, ,")
        assert result == [Decimal("1"), Decimal("2")]

    def test_decimal_list_single_value(self):
        """_decimal_list handles a single value."""
        from bilancio.experiments.ring import _decimal_list

        result = _decimal_list("0.5")
        assert result == [Decimal("0.5")]

    def test_to_yaml_ready_dict(self):
        """_to_yaml_ready converts nested dicts with Decimals."""
        from bilancio.experiments.ring import _to_yaml_ready

        obj = {"a": Decimal("5"), "b": Decimal("1.5"), "c": None}
        result = _to_yaml_ready(obj)
        assert result == {"a": 5, "b": 1.5}

    def test_to_yaml_ready_list(self):
        """_to_yaml_ready converts lists with Decimals."""
        from bilancio.experiments.ring import _to_yaml_ready

        obj = [Decimal("3"), Decimal("1.5"), "hello"]
        result = _to_yaml_ready(obj)
        assert result == [3, 1.5, "hello"]

    def test_to_yaml_ready_decimal_integer(self):
        """_to_yaml_ready converts integer Decimals to int."""
        from bilancio.experiments.ring import _to_yaml_ready

        assert _to_yaml_ready(Decimal("10")) == 10
        assert isinstance(_to_yaml_ready(Decimal("10")), int)

    def test_to_yaml_ready_decimal_float(self):
        """_to_yaml_ready converts non-integer Decimals to float."""
        from bilancio.experiments.ring import _to_yaml_ready

        result = _to_yaml_ready(Decimal("3.14"))
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_to_yaml_ready_passthrough(self):
        """_to_yaml_ready passes through non-Decimal/dict/list values."""
        from bilancio.experiments.ring import _to_yaml_ready

        assert _to_yaml_ready("hello") == "hello"
        assert _to_yaml_ready(42) == 42
        assert _to_yaml_ready(True) is True


# =============================================================================
# 18. RingSweepConfig validators
# =============================================================================


class TestRingSweepConfigValidation:
    """Tests for _RingSweepGridConfig, _RingSweepLHSConfig, _RingSweepFrontierConfig validators."""

    def test_grid_config_requires_kappas_when_enabled(self):
        """Grid config raises when enabled but kappas empty."""
        from pydantic import ValidationError as PydanticValidationError

        from bilancio.experiments.ring import _RingSweepGridConfig

        with pytest.raises(PydanticValidationError, match="kappas"):
            _RingSweepGridConfig(enabled=True, kappas=[], concentrations=[Decimal("1")], mus=[Decimal("0")])

    def test_grid_config_disabled_allows_empty(self):
        """Grid config allows empty lists when disabled."""
        from bilancio.experiments.ring import _RingSweepGridConfig

        config = _RingSweepGridConfig(enabled=False)
        assert config.kappas == []

    def test_grid_config_defaults_monotonicities(self):
        """Grid config defaults monotonicities to [0] when empty."""
        from bilancio.experiments.ring import _RingSweepGridConfig

        config = _RingSweepGridConfig(
            enabled=True,
            kappas=[Decimal("1")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
            monotonicities=[],
        )
        assert config.monotonicities == [Decimal("0")]

    def test_lhs_config_requires_ranges_when_count_positive(self):
        """LHS config raises when count > 0 but ranges missing."""
        from pydantic import ValidationError as PydanticValidationError

        from bilancio.experiments.ring import _RingSweepLHSConfig

        with pytest.raises(PydanticValidationError, match="kappa_range"):
            _RingSweepLHSConfig(count=10)

    def test_lhs_config_defaults_monotonicity_range(self):
        """LHS config defaults monotonicity_range to (0, 0) when count > 0."""
        from bilancio.experiments.ring import _RingSweepLHSConfig

        config = _RingSweepLHSConfig(
            count=5,
            kappa_range=(Decimal("0.1"), Decimal("2")),
            concentration_range=(Decimal("0.5"), Decimal("2")),
            mu_range=(Decimal("0"), Decimal("1")),
        )
        assert config.monotonicity_range == (Decimal("0"), Decimal("0"))

    def test_lhs_config_zero_count_passes(self):
        """LHS config with count=0 needs no ranges."""
        from bilancio.experiments.ring import _RingSweepLHSConfig

        config = _RingSweepLHSConfig(count=0)
        assert config.count == 0

    def test_frontier_config_requires_fields_when_enabled(self):
        """Frontier config raises when enabled but fields missing."""
        from pydantic import ValidationError as PydanticValidationError

        from bilancio.experiments.ring import _RingSweepFrontierConfig

        with pytest.raises(PydanticValidationError, match="missing"):
            _RingSweepFrontierConfig(enabled=True)

    def test_frontier_config_disabled_passes(self):
        """Frontier config with enabled=False needs no fields."""
        from bilancio.experiments.ring import _RingSweepFrontierConfig

        config = _RingSweepFrontierConfig(enabled=False)
        assert not config.enabled

    def test_ring_sweep_config_bad_version(self):
        """RingSweepConfig rejects version != 1."""
        from pydantic import ValidationError as PydanticValidationError

        from bilancio.experiments.ring import RingSweepConfig

        with pytest.raises(PydanticValidationError, match="version"):
            RingSweepConfig(version=2)


# =============================================================================
# 19. load_ring_sweep_config
# =============================================================================


class TestLoadRingSweepConfig:
    """Tests for load_ring_sweep_config file loading and validation."""

    def test_load_valid_config(self, tmp_path: Path):
        """load_ring_sweep_config loads a valid YAML config."""
        import yaml

        from bilancio.experiments.ring import load_ring_sweep_config

        config_data = {
            "version": 1,
            "grid": {
                "enabled": True,
                "kappas": [0.5, 1.0],
                "concentrations": [1],
                "mus": [0],
            },
        }
        config_path = tmp_path / "sweep.yaml"
        with config_path.open("w") as fh:
            yaml.safe_dump(config_data, fh)

        config = load_ring_sweep_config(config_path)
        assert config.version == 1
        assert config.grid is not None
        assert len(config.grid.kappas) == 2

    def test_load_missing_file(self, tmp_path: Path):
        """load_ring_sweep_config raises FileNotFoundError for missing file."""
        from bilancio.experiments.ring import load_ring_sweep_config

        with pytest.raises(FileNotFoundError):
            load_ring_sweep_config(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path):
        """load_ring_sweep_config raises ValueError for invalid config."""
        from bilancio.experiments.ring import load_ring_sweep_config

        config_path = tmp_path / "bad.yaml"
        config_path.write_text("version: 99\n")

        with pytest.raises(ValueError, match="Invalid sweep configuration"):
            load_ring_sweep_config(config_path)

    def test_load_non_mapping_yaml(self, tmp_path: Path):
        """load_ring_sweep_config raises ValueError for non-mapping YAML."""
        from bilancio.experiments.ring import load_ring_sweep_config

        config_path = tmp_path / "list.yaml"
        config_path.write_text("- item1\n- item2\n")

        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_ring_sweep_config(config_path)


# =============================================================================
# 20. RingSweepRunner utility methods
# =============================================================================


class TestRingSweepRunnerUtilities:
    """Tests for _rel_path, _artifact_loader_for_result, _liquidity_allocation_dict, _upsert_registry."""

    def test_rel_path_normal(self, tmp_path: Path):
        """_rel_path returns relative path from base_dir."""
        from bilancio.experiments.ring import RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=1,
        )

        abs_path = tmp_path / "runs" / "test_001" / "scenario.yaml"
        rel = runner._rel_path(abs_path)
        assert "runs" in rel
        assert "scenario.yaml" in rel

    def test_rel_path_outside_base(self, tmp_path: Path):
        """_rel_path returns absolute path when path is not under base_dir."""
        from bilancio.experiments.ring import RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=1,
        )

        abs_path = Path("/tmp/somewhere/else.txt")
        rel = runner._rel_path(abs_path)
        assert rel == str(abs_path)

    def test_liquidity_allocation_uniform(self, tmp_path: Path):
        """_liquidity_allocation_dict for uniform mode."""
        from bilancio.experiments.ring import RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=1,
        )
        result = runner._liquidity_allocation_dict()
        assert result == {"mode": "uniform"}

    def test_liquidity_allocation_single_at(self, tmp_path: Path):
        """_liquidity_allocation_dict for single_at mode includes agent."""
        from bilancio.experiments.ring import RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="single_at", liquidity_agent="firm_0", base_seed=1,
        )
        result = runner._liquidity_allocation_dict()
        assert result == {"mode": "single_at", "agent": "firm_0"}

    def test_artifact_loader_local(self, tmp_path: Path):
        """_artifact_loader_for_result returns LocalArtifactLoader for local storage."""
        from bilancio.experiments.ring import RingSweepRunner
        from bilancio.runners import ExecutionResult
        from bilancio.storage import LocalArtifactLoader
        from bilancio.storage.models import RunStatus

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=1,
        )

        result = ExecutionResult(
            run_id="test",
            status=RunStatus.COMPLETED,
            artifacts={},
            storage_type="local",
            storage_base=str(tmp_path),
        )
        loader = runner._artifact_loader_for_result(result)
        assert isinstance(loader, LocalArtifactLoader)

    def test_upsert_registry(self, tmp_path: Path):
        """_upsert_registry creates a RegistryEntry and calls store.upsert."""
        from bilancio.experiments.ring import RingSweepRunner
        from bilancio.storage.models import RunStatus

        mock_store = MagicMock()
        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=1,
            registry_store=mock_store,
        )

        runner._upsert_registry(
            run_id="test_run",
            phase="grid",
            status=RunStatus.RUNNING,
            parameters={"phase": "grid", "seed": 42},
        )

        mock_store.upsert.assert_called_once()
        entry = mock_store.upsert.call_args[0][0]
        assert entry.run_id == "test_run"
        assert entry.status == RunStatus.RUNNING


# =============================================================================
# 21. RingSweepRunner.run_lhs
# =============================================================================


class TestRingRunLHS:
    """Tests for RingSweepRunner.run_lhs."""

    def test_run_lhs_zero_count_returns_empty(self, tmp_path: Path):
        """run_lhs with count=0 returns empty list."""
        from bilancio.experiments.ring import RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=1,
        )

        result = runner.run_lhs(0, kappa_range=(Decimal("0.1"), Decimal("2")),
                                 concentration_range=(Decimal("0.5"), Decimal("2")),
                                 mu_range=(Decimal("0"), Decimal("1")),
                                 monotonicity_range=(Decimal("0"), Decimal("0")))
        assert result == []

    def test_run_lhs_invokes_execute_run(self, tmp_path: Path):
        """run_lhs generates LHS samples and calls _execute_run for each."""
        from bilancio.experiments.ring import RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=10,
        )

        call_count = [0]

        def mock_execute(phase, kappa, concentration, mu, monotonicity, seed, **kwargs):
            call_count[0] += 1
            return _make_summary(run_id=f"lhs_{seed}", phase=phase, kappa=kappa,
                                 concentration=concentration, mu=mu, monotonicity=monotonicity)

        with patch.object(runner, "_execute_run", mock_execute):
            summaries = runner.run_lhs(
                3,
                kappa_range=(Decimal("0.1"), Decimal("2")),
                concentration_range=(Decimal("0.5"), Decimal("2")),
                mu_range=(Decimal("0"), Decimal("1")),
                monotonicity_range=(Decimal("0"), Decimal("0")),
            )

        assert len(summaries) == 3
        assert call_count[0] == 3


# =============================================================================
# 22. RingSweepRunner.run_frontier
# =============================================================================


class TestRingRunFrontier:
    """Tests for RingSweepRunner.run_frontier."""

    def test_run_frontier_invokes_execute_run(self, tmp_path: Path):
        """run_frontier calls _execute_run through the frontier sampling function."""
        from bilancio.experiments.ring import RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=10,
        )

        call_count = [0]

        def mock_execute(phase, kappa, concentration, mu, monotonicity, seed, **kwargs):
            call_count[0] += 1
            return _make_summary(
                run_id=f"frontier_{seed}", phase=phase, kappa=kappa,
                concentration=concentration, mu=mu, monotonicity=monotonicity,
                delta_total=Decimal("0.1") if kappa > Decimal("1") else Decimal("0.5"),
            )

        with patch.object(runner, "_execute_run", mock_execute):
            summaries = runner.run_frontier(
                concentrations=[Decimal("1")],
                mus=[Decimal("0")],
                monotonicities=[Decimal("0")],
                kappa_low=Decimal("0.1"),
                kappa_high=Decimal("2"),
                tolerance=Decimal("0.5"),
                max_iterations=3,
            )

        # frontier should call execute at least once (the midpoint checks)
        assert len(summaries) >= 1
        assert call_count[0] >= 1


# =============================================================================
# 23. RingSweepRunner._finalize_run cloud-only path
# =============================================================================


class TestFinalizeRunCloudPath:
    """Tests for RingSweepRunner._finalize_run with cloud metrics."""

    def test_finalize_run_cloud_with_metrics(self, tmp_path: Path):
        """_finalize_run uses pre-computed metrics from cloud execution."""
        from bilancio.experiments.ring import PreparedRun, RingSweepRunner
        from bilancio.runners import ExecutionResult, RunOptions
        from bilancio.storage.models import RunStatus

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=1,
        )
        runner.skip_local_processing = True

        prepared = PreparedRun(
            run_id="cloud_001",
            phase="grid",
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0"),
            monotonicity=Decimal("0"),
            seed=42,
            scenario_config={"initial_actions": []},
            options=RunOptions(mode="until_stable", max_days=10),
            run_dir=tmp_path / "runs" / "cloud_001",
            out_dir=tmp_path / "runs" / "cloud_001" / "out",
            scenario_path=tmp_path / "runs" / "cloud_001" / "scenario.yaml",
            base_params={"phase": "grid"},
            S1=Decimal("1000"),
            L0=Decimal("500"),
        )

        result = ExecutionResult(
            run_id="cloud_001",
            status=RunStatus.COMPLETED,
            artifacts={},
            storage_type="local",
            storage_base=str(tmp_path),
            metrics={
                "delta_total": 0.3,
                "phi_total": 0.7,
                "max_day": 5,
                "n_defaults": 2,
                "cascade_fraction": 0.1,
                "cb_loans_created_count": 1,
                "S_total": 10000,
            },
            modal_call_id="fc-test",
        )

        summary = runner._finalize_run(prepared, result)
        assert summary.delta_total == Decimal("0.3")
        assert summary.phi_total == Decimal("0.7")
        assert summary.time_to_stability == 5
        assert summary.n_defaults == 2
        assert summary.modal_call_id == "fc-test"

    def test_finalize_run_failure(self, tmp_path: Path):
        """_finalize_run returns summary with None delta for failed runs."""
        from bilancio.experiments.ring import PreparedRun, RingSweepRunner
        from bilancio.runners import ExecutionResult, RunOptions
        from bilancio.storage.models import RunStatus

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=1,
        )
        runner.skip_local_processing = True

        prepared = PreparedRun(
            run_id="fail_001",
            phase="grid",
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0"),
            monotonicity=Decimal("0"),
            seed=42,
            scenario_config={},
            options=RunOptions(mode="until_stable", max_days=10),
            run_dir=tmp_path / "runs" / "fail_001",
            out_dir=tmp_path / "runs" / "fail_001" / "out",
            scenario_path=tmp_path / "runs" / "fail_001" / "scenario.yaml",
            base_params={"phase": "grid"},
            S1=Decimal("1000"),
            L0=Decimal("500"),
        )

        result = ExecutionResult(
            run_id="fail_001",
            status=RunStatus.FAILED,
            artifacts={},
            storage_type="local",
            storage_base=str(tmp_path),
            error="Boom!",
        )

        summary = runner._finalize_run(prepared, result)
        assert summary.delta_total is None
        assert summary.phi_total is None
        assert summary.time_to_stability == 0


# =============================================================================
# 24. Bank/NBFI _write_comparison_csv
# =============================================================================


class TestBankNBFIWriteComparisonCsv:
    """Tests for _write_comparison_csv for bank and NBFI runners."""

    def test_bank_write_comparison_csv(self, tmp_path: Path):
        """BankComparisonRunner._write_comparison_csv writes all fields."""
        from bilancio.experiments.bank_comparison import (
            BankComparisonConfig,
            BankComparisonResult,
            BankComparisonRunner,
        )

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            BankComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=42, outside_mid_ratio=Decimal("0.90"),
                delta_idle=Decimal("0.50"), phi_idle=Decimal("0.50"),
                idle_run_id="i1", idle_status="completed",
                delta_lend=Decimal("0.30"), phi_lend=Decimal("0.70"),
                lend_run_id="l1", lend_status="completed",
                cb_loans_created_idle=5, cb_loans_created_lend=8,
                bank_defaults_final_idle=0, bank_defaults_final_lend=1,
                total_loss_idle=200, total_loss_lend=100,
            ),
        ]

        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        assert csv_path.exists()

        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["kappa"] == "1"
        assert rows[0]["delta_idle"] == "0.50"
        assert rows[0]["delta_lend"] == "0.30"
        assert rows[0]["bank_lending_effect"] == "0.20"
        assert rows[0]["cb_loans_created_idle"] == "5"
        assert rows[0]["cb_loans_created_lend"] == "8"

    def test_nbfi_write_comparison_csv(self, tmp_path: Path):
        """NBFIComparisonRunner._write_comparison_csv writes all fields."""
        from bilancio.experiments.nbfi_comparison import (
            NBFIComparisonConfig,
            NBFIComparisonResult,
            NBFIComparisonRunner,
        )

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            NBFIComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=42, outside_mid_ratio=Decimal("0.90"),
                delta_idle=Decimal("0.45"), phi_idle=Decimal("0.55"),
                idle_run_id="i1", idle_status="completed",
                delta_lend=Decimal("0.20"), phi_lend=Decimal("0.80"),
                lend_run_id="l1", lend_status="completed",
                total_loss_idle=300, total_loss_lend=150,
                intermediary_loss_idle=50.0, intermediary_loss_lend=25.0,
            ),
        ]

        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        assert csv_path.exists()

        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["delta_idle"] == "0.45"
        assert rows[0]["delta_lend"] == "0.20"
        assert rows[0]["lending_effect"] == "0.25"

    def test_bank_write_comparison_csv_with_none_values(self, tmp_path: Path):
        """BankComparisonRunner._write_comparison_csv handles None values."""
        from bilancio.experiments.bank_comparison import (
            BankComparisonConfig,
            BankComparisonResult,
            BankComparisonRunner,
        )

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            BankComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=42, outside_mid_ratio=Decimal("0.90"),
                delta_idle=None, phi_idle=None,
                idle_run_id="i1", idle_status="failed",
                delta_lend=None, phi_lend=None,
                lend_run_id="l1", lend_status="failed",
            ),
        ]

        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert rows[0]["delta_idle"] == ""
        assert rows[0]["bank_lending_effect"] == ""


# =============================================================================
# 25. NBFI _load_existing_results
# =============================================================================


class TestNBFILoadExistingResults:
    """Tests for NBFIComparisonRunner._load_existing_results."""

    def test_nbfi_loads_existing_csv(self, tmp_path: Path):
        """NBFIComparisonRunner loads existing CSV and advances seed counter."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        agg_dir = tmp_path / "aggregate"
        agg_dir.mkdir(parents=True)
        csv_path = agg_dir / "comparison.csv"

        fields = NBFIComparisonRunner.COMPARISON_FIELDS
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            row = {f: "" for f in fields}
            row.update({
                "kappa": "1", "concentration": "1", "mu": "0",
                "monotonicity": "0", "outside_mid_ratio": "0.90",
                "seed": "100",
                "delta_idle": "0.30", "delta_lend": "0.20",
                "lending_effect": "0.10",
                "idle_run_id": "idle_x", "idle_status": "completed",
                "lend_run_id": "lend_x", "lend_status": "completed",
            })
            writer.writerow(row)

        config = NBFIComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")],
        )
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        assert runner.seed_counter >= 101
        assert len(runner._completed_counts) > 0

    def test_nbfi_skips_load_when_no_csv(self, tmp_path: Path):
        """NBFIComparisonRunner handles missing CSV gracefully."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        assert runner.seed_counter == config.base_seed
        assert len(runner._completed_counts) == 0


# =============================================================================
# 26. _format_time
# =============================================================================


class TestFormatTime:
    """Tests for _format_time utility in bank/NBFI runners."""

    def test_bank_format_time_seconds(self, tmp_path: Path):
        """_format_time returns seconds format for < 60s."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(45) == "45s"

    def test_bank_format_time_minutes(self, tmp_path: Path):
        """_format_time returns minutes format for 60-3600s."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(150) == "2.5m"

    def test_bank_format_time_hours(self, tmp_path: Path):
        """_format_time returns hours format for >= 3600s."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(3720) == "1h 2m"

    def test_nbfi_format_time_seconds(self, tmp_path: Path):
        """NBFI _format_time returns seconds format for < 60s."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(30) == "30s"

    def test_nbfi_format_time_minutes(self, tmp_path: Path):
        """NBFI _format_time returns minutes format for 60-3600s."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(300) == "5.0m"

    def test_nbfi_format_time_hours(self, tmp_path: Path):
        """NBFI _format_time returns hours format for >= 3600s."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(7260) == "2h 1m"


# =============================================================================
# 27. Bank/NBFI _make_key and _next_seed
# =============================================================================


class TestBankNBFIMakeKeyAndSeed:
    """Tests for _make_key and _next_seed helper methods."""

    def test_bank_make_key(self, tmp_path: Path):
        """BankComparisonRunner._make_key returns a tuple of string representations."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        key = runner._make_key(Decimal("1"), Decimal("2"), Decimal("0"), Decimal("0"), Decimal("0.9"))
        assert key == ("1", "2", "0", "0", "0.9")

    def test_bank_next_seed(self, tmp_path: Path):
        """BankComparisonRunner._next_seed increments counter."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig(base_seed=100)
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        s1 = runner._next_seed()
        s2 = runner._next_seed()
        assert s1 == 100
        assert s2 == 101

    def test_nbfi_make_key(self, tmp_path: Path):
        """NBFIComparisonRunner._make_key returns a tuple of string representations."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        key = runner._make_key(Decimal("0.5"), Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0.90"))
        assert key == ("0.5", "1", "0", "0", "0.90")

    def test_nbfi_next_seed(self, tmp_path: Path):
        """NBFIComparisonRunner._next_seed increments counter."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig(base_seed=50)
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        s1 = runner._next_seed()
        s2 = runner._next_seed()
        assert s1 == 50
        assert s2 == 51


# =============================================================================
# 28. NBFIComparisonResult properties
# =============================================================================


class TestNBFIComparisonResultProperties:
    """Tests for NBFIComparisonResult computed properties."""

    def test_lending_effect(self):
        """lending_effect is delta_idle - delta_lend."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonResult

        r = NBFIComparisonResult(
            kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
            monotonicity=Decimal("0"), seed=1, outside_mid_ratio=Decimal("0.90"),
            delta_idle=Decimal("0.50"), phi_idle=Decimal("0.50"),
            idle_run_id="i", idle_status="completed",
            delta_lend=Decimal("0.30"), phi_lend=Decimal("0.70"),
            lend_run_id="l", lend_status="completed",
        )
        assert r.lending_effect == Decimal("0.20")

    def test_lending_effect_none_when_idle_failed(self):
        """lending_effect is None when delta_idle is None."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonResult

        r = NBFIComparisonResult(
            kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
            monotonicity=Decimal("0"), seed=1, outside_mid_ratio=Decimal("0.90"),
            delta_idle=None, phi_idle=None,
            idle_run_id="i", idle_status="failed",
            delta_lend=Decimal("0.30"), phi_lend=Decimal("0.70"),
            lend_run_id="l", lend_status="completed",
        )
        assert r.lending_effect is None

    def test_lending_relief_ratio(self):
        """lending_relief_ratio = effect / delta_idle."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonResult

        r = NBFIComparisonResult(
            kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
            monotonicity=Decimal("0"), seed=1, outside_mid_ratio=Decimal("0.90"),
            delta_idle=Decimal("0.50"), phi_idle=Decimal("0.50"),
            idle_run_id="i", idle_status="completed",
            delta_lend=Decimal("0.30"), phi_lend=Decimal("0.70"),
            lend_run_id="l", lend_status="completed",
        )
        assert r.lending_relief_ratio == Decimal("0.40")

    def test_system_loss_lending_effect(self):
        """system_loss_lending_effect computes difference in system loss pct."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonResult

        r = NBFIComparisonResult(
            kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
            monotonicity=Decimal("0"), seed=1, outside_mid_ratio=Decimal("0.90"),
            delta_idle=Decimal("0.50"), phi_idle=Decimal("0.50"),
            idle_run_id="i", idle_status="completed",
            delta_lend=Decimal("0.30"), phi_lend=Decimal("0.70"),
            lend_run_id="l", lend_status="completed",
            system_loss_pct_idle=0.05, system_loss_pct_lend=0.02,
        )
        assert r.system_loss_lending_effect == pytest.approx(0.03)


# =============================================================================
# 29. NBFI summary JSON with mixed results
# =============================================================================


class TestNBFISummaryWithMixedResults:
    """Tests for NBFI _write_summary_json with improved/worsened/unchanged."""

    def test_nbfi_summary_counts_all_categories(self, tmp_path: Path):
        """_write_summary_json correctly counts improved, worsened, unchanged."""
        from bilancio.experiments.nbfi_comparison import (
            NBFIComparisonConfig,
            NBFIComparisonResult,
            NBFIComparisonRunner,
        )

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            # Improved
            NBFIComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=1, outside_mid_ratio=Decimal("0.90"),
                delta_idle=Decimal("0.40"), phi_idle=Decimal("0.60"),
                idle_run_id="i1", idle_status="completed",
                delta_lend=Decimal("0.20"), phi_lend=Decimal("0.80"),
                lend_run_id="l1", lend_status="completed",
            ),
            # Worsened
            NBFIComparisonResult(
                kappa=Decimal("2"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=2, outside_mid_ratio=Decimal("0.90"),
                delta_idle=Decimal("0.10"), phi_idle=Decimal("0.90"),
                idle_run_id="i2", idle_status="completed",
                delta_lend=Decimal("0.30"), phi_lend=Decimal("0.70"),
                lend_run_id="l2", lend_status="completed",
            ),
            # Unchanged
            NBFIComparisonResult(
                kappa=Decimal("4"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=3, outside_mid_ratio=Decimal("0.90"),
                delta_idle=Decimal("0.05"), phi_idle=Decimal("0.95"),
                idle_run_id="i3", idle_status="completed",
                delta_lend=Decimal("0.05"), phi_lend=Decimal("0.95"),
                lend_run_id="l3", lend_status="completed",
            ),
        ]

        runner._write_summary_json()

        summary_path = tmp_path / "aggregate" / "summary.json"
        with summary_path.open() as fh:
            data = json.load(fh)

        assert data["combos_improved"] == 1
        assert data["combos_worsened"] == 1
        assert data["combos_unchanged"] == 1
        assert data["total_combos"] == 3


# =============================================================================
# 30. Bank _get_enabled_arm_defs
# =============================================================================


class TestBankGetEnabledArmDefs:
    """Tests for BankComparisonRunner._get_enabled_arm_defs."""

    def test_bank_default_arms(self, tmp_path: Path):
        """Default bank config enables idle and lend arms."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        arms = runner._get_enabled_arm_defs()
        names = [a[0] for a in arms]
        assert "idle" in names
        assert "lend" in names
        assert len(names) == 2


# =============================================================================
# 31. NBFI _get_enabled_arm_defs
# =============================================================================


class TestNBFIGetEnabledArmDefs:
    """Tests for NBFIComparisonRunner._get_enabled_arm_defs."""

    def test_nbfi_default_arms(self, tmp_path: Path):
        """Default NBFI config enables idle and lend arms."""
        from bilancio.experiments.nbfi_comparison import NBFIComparisonConfig, NBFIComparisonRunner

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        arms = runner._get_enabled_arm_defs()
        names = [a[0] for a in arms]
        assert "idle" in names
        assert "lend" in names
        assert len(names) == 2


# =============================================================================
# 32. BalancedComparisonResult computed properties (lines 290-474)
# =============================================================================


class TestBalancedResultComputedProperties:
    """Tests for BalancedComparisonResult computed properties (trading_effect, lending_effect, etc.)."""

    def _make_balanced_result(self, **overrides):
        """Helper: create a BalancedComparisonResult with sensible defaults."""
        from bilancio.experiments.balanced_comparison import BalancedComparisonResult

        defaults = dict(
            kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
            monotonicity=Decimal("0"), seed=1,
            face_value=Decimal("20"), outside_mid_ratio=Decimal("0.90"),
            big_entity_share=Decimal("0.25"),
            delta_passive=Decimal("0.50"), phi_passive=Decimal("0.50"),
            passive_run_id="p1", passive_status="completed",
            delta_active=Decimal("0.30"), phi_active=Decimal("0.70"),
            active_run_id="a1", active_status="completed",
            n_defaults_passive=5, n_defaults_active=3,
            cascade_fraction_passive=Decimal("0.10"),
            cascade_fraction_active=Decimal("0.05"),
        )
        defaults.update(overrides)
        return BalancedComparisonResult(**defaults)

    def test_trading_effect(self):
        """trading_effect = delta_passive - delta_active."""
        r = self._make_balanced_result()
        assert r.trading_effect == Decimal("0.20")

    def test_trading_effect_none_when_missing(self):
        """trading_effect is None when delta_active is None."""
        r = self._make_balanced_result(delta_active=None)
        assert r.trading_effect is None

    def test_trading_relief_ratio(self):
        """trading_relief_ratio = effect / delta_passive."""
        r = self._make_balanced_result()
        assert r.trading_relief_ratio == Decimal("0.40")

    def test_trading_relief_ratio_zero_passive(self):
        """trading_relief_ratio = 0 when delta_passive is 0."""
        r = self._make_balanced_result(delta_passive=Decimal("0"), delta_active=Decimal("0"))
        assert r.trading_relief_ratio == Decimal("0")

    def test_cascade_effect(self):
        """cascade_effect = cascade_passive - cascade_active."""
        r = self._make_balanced_result()
        assert r.cascade_effect == Decimal("0.05")

    def test_cascade_effect_none_when_missing(self):
        """cascade_effect is None when cascade_fraction_active is None."""
        r = self._make_balanced_result(cascade_fraction_active=None)
        assert r.cascade_effect is None

    def test_lending_effect(self):
        """lending_effect = delta_passive - delta_lender."""
        r = self._make_balanced_result(delta_lender=Decimal("0.20"))
        assert r.lending_effect == Decimal("0.30")

    def test_lending_effect_none(self):
        """lending_effect is None when delta_lender is None."""
        r = self._make_balanced_result()
        assert r.lending_effect is None

    def test_combined_effect(self):
        """combined_effect = delta_passive - delta_dealer_lender."""
        r = self._make_balanced_result(delta_dealer_lender=Decimal("0.15"))
        assert r.combined_effect == Decimal("0.35")

    def test_combined_effect_none(self):
        """combined_effect is None when delta_dealer_lender is None."""
        r = self._make_balanced_result()
        assert r.combined_effect is None

    def test_bank_passive_effect(self):
        """bank_passive_effect = delta_passive - delta_bank_passive."""
        r = self._make_balanced_result(delta_bank_passive=Decimal("0.25"))
        assert r.bank_passive_effect == Decimal("0.25")

    def test_bank_passive_effect_none(self):
        """bank_passive_effect is None when delta_bank_passive is None."""
        r = self._make_balanced_result()
        assert r.bank_passive_effect is None

    def test_bank_dealer_effect(self):
        """bank_dealer_effect = delta_passive - delta_bank_dealer."""
        r = self._make_balanced_result(delta_bank_dealer=Decimal("0.10"))
        assert r.bank_dealer_effect == Decimal("0.40")

    def test_bank_dealer_effect_none(self):
        """bank_dealer_effect is None when delta_bank_dealer is None."""
        r = self._make_balanced_result()
        assert r.bank_dealer_effect is None

    def test_bank_dealer_nbfi_effect(self):
        """bank_dealer_nbfi_effect = delta_passive - delta_bank_dealer_nbfi."""
        r = self._make_balanced_result(delta_bank_dealer_nbfi=Decimal("0.05"))
        assert r.bank_dealer_nbfi_effect == Decimal("0.45")

    def test_bank_dealer_nbfi_effect_none(self):
        """bank_dealer_nbfi_effect is None when delta_bank_dealer_nbfi is None."""
        r = self._make_balanced_result()
        assert r.bank_dealer_nbfi_effect is None

    def test_adjusted_trading_effect(self):
        """adjusted_trading_effect subtracts intermediary loss differential."""
        r = self._make_balanced_result(
            intermediary_loss_pct_passive=0.01,
            intermediary_loss_pct_active=0.03,
        )
        # trading_effect = 0.20, intermediary diff = 0.03 - 0.01 = 0.02
        assert r.adjusted_trading_effect == pytest.approx(0.18)

    def test_adjusted_trading_effect_none_when_missing(self):
        """adjusted_trading_effect is None when intermediary data missing."""
        r = self._make_balanced_result()
        assert r.adjusted_trading_effect is None

    def test_adjusted_lending_effect(self):
        """adjusted_lending_effect subtracts intermediary loss differential."""
        r = self._make_balanced_result(
            delta_lender=Decimal("0.20"),
            intermediary_loss_pct_passive=0.01,
            intermediary_loss_pct_lender=0.04,
        )
        # lending_effect = 0.30, intermediary diff = 0.04 - 0.01 = 0.03
        assert r.adjusted_lending_effect == pytest.approx(0.27)

    def test_adjusted_lending_effect_none(self):
        """adjusted_lending_effect is None when lending data missing."""
        r = self._make_balanced_result()
        assert r.adjusted_lending_effect is None

    def test_adjusted_combined_effect(self):
        """adjusted_combined_effect subtracts intermediary loss differential."""
        r = self._make_balanced_result(
            delta_dealer_lender=Decimal("0.10"),
            intermediary_loss_pct_passive=0.01,
            intermediary_loss_pct_dealer_lender=0.05,
        )
        # combined_effect = 0.40, intermediary diff = 0.05 - 0.01 = 0.04
        assert r.adjusted_combined_effect == pytest.approx(0.36)

    def test_adjusted_combined_effect_none(self):
        """adjusted_combined_effect is None when combined data missing."""
        r = self._make_balanced_result()
        assert r.adjusted_combined_effect is None

    def test_adjusted_bank_passive_effect(self):
        """adjusted_bank_passive_effect subtracts intermediary loss differential."""
        r = self._make_balanced_result(
            delta_bank_passive=Decimal("0.25"),
            intermediary_loss_pct_passive=0.02,
            intermediary_loss_pct_bank_passive=0.06,
        )
        # bank_passive_effect = 0.25, intermediary diff = 0.06 - 0.02 = 0.04
        assert r.adjusted_bank_passive_effect == pytest.approx(0.21)

    def test_adjusted_bank_passive_effect_none(self):
        """adjusted_bank_passive_effect is None when data missing."""
        r = self._make_balanced_result()
        assert r.adjusted_bank_passive_effect is None

    def test_adjusted_bank_dealer_effect(self):
        """adjusted_bank_dealer_effect subtracts intermediary loss differential."""
        r = self._make_balanced_result(
            delta_bank_dealer=Decimal("0.10"),
            intermediary_loss_pct_passive=0.01,
            intermediary_loss_pct_bank_dealer=0.03,
        )
        # bank_dealer_effect = 0.40, intermediary diff = 0.03 - 0.01 = 0.02
        assert r.adjusted_bank_dealer_effect == pytest.approx(0.38)

    def test_adjusted_bank_dealer_effect_none(self):
        """adjusted_bank_dealer_effect is None when data missing."""
        r = self._make_balanced_result()
        assert r.adjusted_bank_dealer_effect is None

    def test_adjusted_bank_dealer_nbfi_effect(self):
        """adjusted_bank_dealer_nbfi_effect subtracts intermediary loss differential."""
        r = self._make_balanced_result(
            delta_bank_dealer_nbfi=Decimal("0.05"),
            intermediary_loss_pct_passive=0.01,
            intermediary_loss_pct_bank_dealer_nbfi=0.04,
        )
        # bank_dealer_nbfi_effect = 0.45, intermediary diff = 0.04 - 0.01 = 0.03
        assert r.adjusted_bank_dealer_nbfi_effect == pytest.approx(0.42)

    def test_adjusted_bank_dealer_nbfi_effect_none(self):
        """adjusted_bank_dealer_nbfi_effect is None when data missing."""
        r = self._make_balanced_result()
        assert r.adjusted_bank_dealer_nbfi_effect is None

    def test_system_loss_trading_effect(self):
        """system_loss_trading_effect = passive - active system loss."""
        r = self._make_balanced_result(
            system_loss_pct_passive=0.10,
            system_loss_pct_active=0.06,
        )
        assert r.system_loss_trading_effect == pytest.approx(0.04)

    def test_system_loss_trading_effect_none(self):
        """system_loss_trading_effect is None when system_loss_pct_active is None."""
        r = self._make_balanced_result()
        assert r.system_loss_trading_effect is None

    def test_system_loss_lending_effect(self):
        """system_loss_lending_effect = passive - lender system loss."""
        r = self._make_balanced_result(
            system_loss_pct_passive=0.10,
            system_loss_pct_lender=0.04,
        )
        assert r.system_loss_lending_effect == pytest.approx(0.06)

    def test_system_loss_lending_effect_none(self):
        """system_loss_lending_effect is None when system_loss_pct_lender is None."""
        r = self._make_balanced_result()
        assert r.system_loss_lending_effect is None

    def test_system_loss_combined_effect(self):
        """system_loss_combined_effect = passive - dealer_lender system loss."""
        r = self._make_balanced_result(
            system_loss_pct_passive=0.10,
            system_loss_pct_dealer_lender=0.03,
        )
        assert r.system_loss_combined_effect == pytest.approx(0.07)

    def test_system_loss_combined_effect_none(self):
        """system_loss_combined_effect is None when data missing."""
        r = self._make_balanced_result()
        assert r.system_loss_combined_effect is None

    def test_system_loss_bank_passive_effect(self):
        """system_loss_bank_passive_effect = passive - bank_passive system loss."""
        r = self._make_balanced_result(
            system_loss_pct_passive=0.10,
            system_loss_pct_bank_passive=0.05,
        )
        assert r.system_loss_bank_passive_effect == pytest.approx(0.05)

    def test_system_loss_bank_passive_effect_none(self):
        """system_loss_bank_passive_effect is None when data missing."""
        r = self._make_balanced_result()
        assert r.system_loss_bank_passive_effect is None

    def test_system_loss_bank_dealer_effect(self):
        """system_loss_bank_dealer_effect = passive - bank_dealer system loss."""
        r = self._make_balanced_result(
            system_loss_pct_passive=0.10,
            system_loss_pct_bank_dealer=0.02,
        )
        assert r.system_loss_bank_dealer_effect == pytest.approx(0.08)

    def test_system_loss_bank_dealer_effect_none(self):
        """system_loss_bank_dealer_effect is None when data missing."""
        r = self._make_balanced_result()
        assert r.system_loss_bank_dealer_effect is None

    def test_system_loss_bank_dealer_nbfi_effect(self):
        """system_loss_bank_dealer_nbfi_effect = passive - bank_dealer_nbfi system loss."""
        r = self._make_balanced_result(
            system_loss_pct_passive=0.10,
            system_loss_pct_bank_dealer_nbfi=0.01,
        )
        assert r.system_loss_bank_dealer_nbfi_effect == pytest.approx(0.09)

    def test_system_loss_bank_dealer_nbfi_effect_none(self):
        """system_loss_bank_dealer_nbfi_effect is None when data missing."""
        r = self._make_balanced_result()
        assert r.system_loss_bank_dealer_nbfi_effect is None


# =============================================================================
# 33. Bank _run_all_sequential with mocked _run_pair
# =============================================================================


class TestBankRunAllSequential:
    """Tests for BankComparisonRunner._run_all_sequential."""

    def test_sequential_invokes_run_pair_for_each_combo(self, tmp_path: Path):
        """_run_all_sequential calls _run_pair for each kappa x concentration x mu combo."""
        from bilancio.experiments.bank_comparison import (
            BankComparisonConfig,
            BankComparisonResult,
            BankComparisonRunner,
        )

        config = BankComparisonConfig(
            kappas=[Decimal("1"), Decimal("2")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
        )
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        pair_calls = []

        def mock_run_pair(kappa, concentration, mu, monotonicity, outside_mid_ratio):
            pair_calls.append(kappa)
            return BankComparisonResult(
                kappa=kappa, concentration=concentration, mu=mu,
                monotonicity=monotonicity, seed=len(pair_calls),
                outside_mid_ratio=outside_mid_ratio,
                delta_idle=Decimal("0.40"), phi_idle=Decimal("0.60"),
                idle_run_id=f"i{len(pair_calls)}", idle_status="completed",
                delta_lend=Decimal("0.20"), phi_lend=Decimal("0.80"),
                lend_run_id=f"l{len(pair_calls)}", lend_status="completed",
            )

        with patch.object(runner, "_run_pair", mock_run_pair):
            results = runner._run_all_sequential()

        assert len(pair_calls) == 2
        assert len(results) == 2
        # Verify CSV and summary were written
        assert (tmp_path / "aggregate" / "comparison.csv").exists()
        assert (tmp_path / "aggregate" / "summary.json").exists()

    def test_sequential_skips_completed_combos(self, tmp_path: Path):
        """_run_all_sequential skips combos already in _completed_counts."""
        from bilancio.experiments.bank_comparison import (
            BankComparisonConfig,
            BankComparisonResult,
            BankComparisonRunner,
        )

        config = BankComparisonConfig(
            kappas=[Decimal("1"), Decimal("2")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
        )
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        # Mark first combo as completed
        key = runner._make_key(Decimal("1"), Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0.90"))
        runner._completed_counts[key] = 1

        pair_calls = []

        def mock_run_pair(kappa, concentration, mu, monotonicity, outside_mid_ratio):
            pair_calls.append(kappa)
            return BankComparisonResult(
                kappa=kappa, concentration=concentration, mu=mu,
                monotonicity=monotonicity, seed=len(pair_calls),
                outside_mid_ratio=outside_mid_ratio,
                delta_idle=Decimal("0.30"), phi_idle=Decimal("0.70"),
                idle_run_id=f"i{len(pair_calls)}", idle_status="completed",
                delta_lend=Decimal("0.10"), phi_lend=Decimal("0.90"),
                lend_run_id=f"l{len(pair_calls)}", lend_status="completed",
            )

        with patch.object(runner, "_run_pair", mock_run_pair):
            runner._run_all_sequential()

        # Only kappa=2 should run, kappa=1 was already completed
        assert len(pair_calls) == 1
        assert pair_calls[0] == Decimal("2")


# =============================================================================
# 34. NBFI _run_all_sequential with mocked _run_pair
# =============================================================================


class TestNBFIRunAllSequential:
    """Tests for NBFIComparisonRunner._run_all_sequential."""

    def test_sequential_invokes_run_pair(self, tmp_path: Path):
        """_run_all_sequential calls _run_pair for each combo."""
        from bilancio.experiments.nbfi_comparison import (
            NBFIComparisonConfig,
            NBFIComparisonResult,
            NBFIComparisonRunner,
        )

        config = NBFIComparisonConfig(
            kappas=[Decimal("0.5"), Decimal("1")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
        )
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        pair_calls = []

        def mock_run_pair(kappa, concentration, mu, monotonicity, outside_mid_ratio):
            pair_calls.append(kappa)
            return NBFIComparisonResult(
                kappa=kappa, concentration=concentration, mu=mu,
                monotonicity=monotonicity, seed=len(pair_calls),
                outside_mid_ratio=outside_mid_ratio,
                delta_idle=Decimal("0.40"), phi_idle=Decimal("0.60"),
                idle_run_id=f"i{len(pair_calls)}", idle_status="completed",
                delta_lend=Decimal("0.20"), phi_lend=Decimal("0.80"),
                lend_run_id=f"l{len(pair_calls)}", lend_status="completed",
            )

        with patch.object(runner, "_run_pair", mock_run_pair):
            results = runner._run_all_sequential()

        assert len(pair_calls) == 2
        assert len(results) == 2
        assert (tmp_path / "aggregate" / "comparison.csv").exists()
        assert (tmp_path / "aggregate" / "summary.json").exists()


# =============================================================================
# 35. Balanced _run_all_sequential with mocked _run_single_combo
# =============================================================================


class TestBalancedRunAllSequential:
    """Tests for BalancedComparisonRunner._run_all_sequential."""

    def test_sequential_invokes_run_pair(self, tmp_path: Path):
        """_run_all_sequential calls _run_pair for each combo."""
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonResult,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
            outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        pair_calls = []

        def mock_run_pair(kappa, concentration, mu, monotonicity, outside_mid_ratio, topology="ring", seed=None):
            pair_calls.append(kappa)
            return BalancedComparisonResult(
                kappa=kappa, concentration=concentration, mu=mu,
                monotonicity=monotonicity, seed=seed or 1,
                face_value=Decimal("20"), outside_mid_ratio=outside_mid_ratio,
                big_entity_share=Decimal("0.25"),
                delta_passive=Decimal("0.50"), phi_passive=Decimal("0.50"),
                passive_run_id="p1", passive_status="completed",
                delta_active=Decimal("0.30"), phi_active=Decimal("0.70"),
                active_run_id="a1", active_status="completed",
                n_defaults_passive=5, n_defaults_active=3,
                cascade_fraction_passive=Decimal("0.10"),
                cascade_fraction_active=Decimal("0.05"),
                topology=topology,
            )

        with patch.object(runner, "_run_pair", mock_run_pair):
            results = runner._run_all_sequential()

        assert len(pair_calls) == 1
        assert len(results) == 1


# =============================================================================
# 36. Bank _write_stats_analysis and _write_activity_analysis with actual data
# =============================================================================


class TestBankWriteAnalysesWithData:
    """Tests for bank _write_stats_analysis and _write_activity_analysis with actual result data."""

    def test_stats_analysis_runs_with_data(self, tmp_path: Path):
        """_write_stats_analysis attempts to run when results exist (may require pandas)."""
        from bilancio.experiments.bank_comparison import (
            BankComparisonConfig,
            BankComparisonResult,
            BankComparisonRunner,
        )

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        # Create two results for the same kappa (to ensure min_replicates check)
        runner.comparison_results = [
            BankComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=i, outside_mid_ratio=Decimal("0.90"),
                delta_idle=Decimal("0.50"), phi_idle=Decimal("0.50"),
                idle_run_id=f"i{i}", idle_status="completed",
                delta_lend=Decimal("0.30"), phi_lend=Decimal("0.70"),
                lend_run_id=f"l{i}", lend_status="completed",
            )
            for i in range(3)
        ]

        # Write CSV first (stats reads from CSV)
        runner._write_comparison_csv()

        # This should not raise even if analysis internals encounter issues
        runner._write_stats_analysis()

    def test_activity_analysis_runs_with_data(self, tmp_path: Path):
        """_write_activity_analysis attempts to run when results exist."""
        from bilancio.experiments.bank_comparison import (
            BankComparisonConfig,
            BankComparisonResult,
            BankComparisonRunner,
        )

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            BankComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=1, outside_mid_ratio=Decimal("0.90"),
                delta_idle=Decimal("0.40"), phi_idle=Decimal("0.60"),
                idle_run_id="i1", idle_status="completed",
                delta_lend=Decimal("0.20"), phi_lend=Decimal("0.80"),
                lend_run_id="l1", lend_status="completed",
            ),
        ]
        runner._write_comparison_csv()

        # Should not raise
        runner._write_activity_analysis()


# =============================================================================
# 37. NBFI _write_stats_analysis and _write_activity_analysis with actual data
# =============================================================================


class TestNBFIWriteAnalysesWithData:
    """Tests for NBFI _write_stats_analysis and _write_activity_analysis with actual result data."""

    def test_stats_analysis_runs_with_data(self, tmp_path: Path):
        """_write_stats_analysis attempts to run when results exist."""
        from bilancio.experiments.nbfi_comparison import (
            NBFIComparisonConfig,
            NBFIComparisonResult,
            NBFIComparisonRunner,
        )

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            NBFIComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=i, outside_mid_ratio=Decimal("0.90"),
                delta_idle=Decimal("0.45"), phi_idle=Decimal("0.55"),
                idle_run_id=f"i{i}", idle_status="completed",
                delta_lend=Decimal("0.25"), phi_lend=Decimal("0.75"),
                lend_run_id=f"l{i}", lend_status="completed",
            )
            for i in range(3)
        ]
        runner._write_comparison_csv()

        runner._write_stats_analysis()

    def test_activity_analysis_runs_with_data(self, tmp_path: Path):
        """_write_activity_analysis attempts to run when results exist."""
        from bilancio.experiments.nbfi_comparison import (
            NBFIComparisonConfig,
            NBFIComparisonResult,
            NBFIComparisonRunner,
        )

        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            NBFIComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=1, outside_mid_ratio=Decimal("0.90"),
                delta_idle=Decimal("0.35"), phi_idle=Decimal("0.65"),
                idle_run_id="i1", idle_status="completed",
                delta_lend=Decimal("0.15"), phi_lend=Decimal("0.85"),
                lend_run_id="l1", lend_status="completed",
            ),
        ]
        runner._write_comparison_csv()

        runner._write_activity_analysis()


# =============================================================================
# 38. RingSweepRunner._prepare_run local path
# =============================================================================


class TestRingPrepareRunLocalPath:
    """Tests for RingSweepRunner._prepare_run local directory creation and file writing."""

    def test_prepare_run_creates_local_dirs(self, tmp_path: Path):
        """_prepare_run creates run_dir and out_dir locally."""
        from bilancio.experiments.ring import RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=1,
        )

        prepared = runner._prepare_run(
            phase="grid", kappa=Decimal("1"), concentration=Decimal("1"),
            mu=Decimal("0"), monotonicity=Decimal("0"), seed=42,
        )

        assert prepared.run_dir.exists()
        assert prepared.out_dir.exists()
        assert prepared.scenario_path.exists()
        assert prepared.kappa == Decimal("1")
        assert prepared.seed == 42
        assert prepared.S1 >= 0
        assert prepared.L0 >= 0

    def test_prepare_run_cloud_skips_dirs(self, tmp_path: Path):
        """_prepare_run in cloud mode skips directory creation."""
        from bilancio.experiments.ring import RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=1,
        )
        runner.skip_local_processing = True

        prepared = runner._prepare_run(
            phase="grid", kappa=Decimal("1"), concentration=Decimal("1"),
            mu=Decimal("0"), monotonicity=Decimal("0"), seed=42,
        )

        # Cloud mode uses /tmp placeholder paths - they should not exist
        assert not prepared.run_dir.exists()
        assert prepared.run_id.startswith("grid_")

    def test_prepare_run_with_label(self, tmp_path: Path):
        """_prepare_run includes label in run_id when provided."""
        from bilancio.experiments.ring import RingSweepRunner

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=1,
        )

        prepared = runner._prepare_run(
            phase="grid", kappa=Decimal("1"), concentration=Decimal("1"),
            mu=Decimal("0"), monotonicity=Decimal("0"), seed=42,
            label="test_label",
        )

        assert "test_label" in prepared.run_id

    def test_prepare_run_applies_adaptive_flag_overrides(self, tmp_path: Path):
        """Explicit adaptive overrides should be applied after preset defaults."""
        from bilancio.experiments.ring import RingSweepRunner
        import yaml

        runner = RingSweepRunner(
            out_dir=tmp_path,
            name_prefix="Test",
            n_agents=6,
            maturity_days=8,
            Q_total=Decimal("200"),
            liquidity_mode="uniform",
            liquidity_agent=None,
            base_seed=1,
            dealer_enabled=True,
            balanced_mode=True,
            adapt_preset="full",
            adapt_overrides={
                "adaptive_planning_horizon": False,
                "adaptive_risk_aversion": False,
                "adaptive_lookback": False,
                "adaptive_issuer_specific": False,
                "adaptive_ev_term_structure": False,
                "adaptive_convex_spreads": False,
            },
        )

        prepared = runner._prepare_run(
            phase="grid",
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0"),
            monotonicity=Decimal("0"),
            seed=42,
        )

        scenario = yaml.safe_load(prepared.scenario_path.read_text(encoding="utf-8"))
        balanced = scenario["balanced_dealer"]
        risk = scenario["dealer"]["risk_assessment"]

        # Overrides force these off even though adapt_preset="full" turns them on.
        assert balanced["adaptive_planning_horizon"] is False
        assert balanced["adaptive_risk_aversion"] is False
        assert balanced["adaptive_ev_term_structure"] is False
        assert balanced["adaptive_convex_spreads"] is False
        assert risk["adaptive_lookback"] is False
        assert risk["adaptive_issuer_specific"] is False
        assert risk["adaptive_ev_term_structure"] is False

        # Non-overridden full-preset flags should remain enabled.
        assert balanced["adaptive_reserves"] is True


# =============================================================================
# 39. Bank _run_pair with mocked execute_run
# =============================================================================


class TestBankRunPair:
    """Tests for BankComparisonRunner._run_pair end-to-end."""

    def test_run_pair_returns_result_with_both_arms(self, tmp_path: Path):
        """_run_pair executes idle and lend arms and returns combined result."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")],
        )
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        idle_summary = _make_summary(
            run_id="idle_001", phase="bank_idle", kappa=Decimal("1"),
            concentration=Decimal("1"), mu=Decimal("0"), monotonicity=Decimal("0"),
            delta_total=Decimal("0.50"), phi_total=Decimal("0.50"),
        )
        lend_summary = _make_summary(
            run_id="lend_001", phase="bank_lend", kappa=Decimal("1"),
            concentration=Decimal("1"), mu=Decimal("0"), monotonicity=Decimal("0"),
            delta_total=Decimal("0.30"), phi_total=Decimal("0.70"),
        )

        call_count = [0]

        def mock_execute_run(phase, kappa, concentration, mu, monotonicity, seed, **kwargs):
            call_count[0] += 1
            if phase == "bank_idle":
                return idle_summary
            return lend_summary

        # We need to mock the idle/lend runners' _execute_run
        idle_runner = MagicMock()
        idle_runner._execute_run = lambda **kw: idle_summary
        lend_runner = MagicMock()
        lend_runner._execute_run = lambda **kw: lend_summary

        with patch.object(runner, "_get_idle_runner", return_value=idle_runner), \
             patch.object(runner, "_get_lend_runner", return_value=lend_runner):
            result = runner._run_pair(
                Decimal("1"), Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0.90")
            )

        assert result.delta_idle == Decimal("0.50")
        assert result.delta_lend == Decimal("0.30")
        assert result.bank_lending_effect == Decimal("0.20")


# =============================================================================
# 40. Bank _persist_run_to_supabase
# =============================================================================


class TestBankPersistToSupabase:
    """Tests for BankComparisonRunner._persist_run_to_supabase."""

    def test_persist_skips_when_no_store(self, tmp_path: Path):
        """_persist_run_to_supabase is a no-op when Supabase is disabled."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        summary = _make_summary(
            run_id="test", phase="bank_idle", kappa=Decimal("1"),
            concentration=Decimal("1"), mu=Decimal("0"), monotonicity=Decimal("0"),
        )

        # Should not raise
        runner._persist_run_to_supabase(
            summary, "bank_idle", Decimal("1"), Decimal("1"), Decimal("0"), Decimal("0.90"), 42
        )

    def test_persist_calls_store_when_enabled(self, tmp_path: Path):
        """_persist_run_to_supabase calls store.upsert when store is available."""
        from bilancio.experiments.bank_comparison import BankComparisonConfig, BankComparisonRunner

        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        mock_store = MagicMock()
        runner._supabase_store = mock_store

        summary = _make_summary(
            run_id="test", phase="bank_idle", kappa=Decimal("1"),
            concentration=Decimal("1"), mu=Decimal("0"), monotonicity=Decimal("0"),
        )

        runner._persist_run_to_supabase(
            summary, "bank_idle", Decimal("1"), Decimal("1"), Decimal("0"), Decimal("0.90"), 42
        )

        mock_store.upsert.assert_called_once()


# =============================================================================
# 41. Ring _finalize_run local path (non-cloud)
# =============================================================================


class TestFinalizeRunLocalPath:
    """Tests for RingSweepRunner._finalize_run local (non-cloud) path."""

    def test_finalize_run_local_with_artifacts(self, tmp_path: Path):
        """_finalize_run computes metrics from local artifacts when not cloud-only."""
        from bilancio.experiments.ring import PreparedRun, RingSweepRunner
        from bilancio.runners import ExecutionResult, RunOptions
        from bilancio.storage.models import RunStatus

        runner = RingSweepRunner(
            out_dir=tmp_path, name_prefix="Test",
            n_agents=3, maturity_days=3, Q_total=Decimal("100"),
            liquidity_mode="uniform", liquidity_agent=None, base_seed=1,
        )
        # NOT cloud mode
        runner.skip_local_processing = False

        run_dir = tmp_path / "runs" / "local_001"
        out_dir = run_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        prepared = PreparedRun(
            run_id="local_001",
            phase="grid",
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0"),
            monotonicity=Decimal("0"),
            seed=42,
            scenario_config={"initial_actions": []},
            options=RunOptions(mode="until_stable", max_days=10),
            run_dir=run_dir,
            out_dir=out_dir,
            scenario_path=run_dir / "scenario.yaml",
            base_params={"phase": "grid"},
            S1=Decimal("1000"),
            L0=Decimal("500"),
        )

        result = ExecutionResult(
            run_id="local_001",
            status=RunStatus.COMPLETED,
            artifacts={"events_jsonl": str(out_dir / "events.jsonl")},
            storage_type="local",
            storage_base=str(tmp_path),
        )

        # Mock MetricsComputer and extract_initial_capitals since we don't have actual events
        mock_bundle = MagicMock()
        mock_bundle.summary = {
            "delta_total": Decimal("0.25"),
            "phi_total": Decimal("0.75"),
            "max_day": 4,
            "n_defaults": 1,
            "cascade_fraction": Decimal("0.05"),
            "cb_loans_created_count": 0,
            "cb_interest_total_paid": 0,
            "cb_loans_outstanding_pre_final": 0,
            "bank_defaults_final": 0,
            "cb_reserve_destruction_pct": 0.0,
            "delta_bank": None,
            "deposit_loss_gross": 0,
            "deposit_loss_pct": None,
            "payable_default_loss": 0,
            "total_loss": 0,
            "S_total": 1000,
            "nbfi_loan_loss": 0,
            "bank_credit_loss": 0,
            "cb_backstop_loss": 0,
        }

        mock_computer_cls = MagicMock()
        mock_computer_instance = MagicMock()
        mock_computer_instance.compute.return_value = mock_bundle
        mock_computer_instance.write_outputs.return_value = {
            "metrics_csv": out_dir / "metrics.csv",
            "metrics_html": out_dir / "metrics.html",
        }
        mock_computer_cls.return_value = mock_computer_instance

        with patch("bilancio.experiments.ring.MetricsComputer", mock_computer_cls), \
             patch("bilancio.analysis.report.extract_initial_capitals", return_value={
                 "intermediary_capital": 0, "dealer_capital": 0, "vbt_capital": 0,
                 "lender_capital": 0, "bank_capital": 0,
             }):
            summary = runner._finalize_run(prepared, result)

        assert summary.delta_total == Decimal("0.25")
        assert summary.phi_total == Decimal("0.75")
        assert summary.time_to_stability == 4
        assert summary.n_defaults == 1


# =============================================================================
# 42. BankComparisonResult computed properties
# =============================================================================


class TestBankComparisonResultProperties:
    """Tests for BankComparisonResult computed properties."""

    def test_bank_lending_effect(self):
        """bank_lending_effect = delta_idle - delta_lend."""
        from bilancio.experiments.bank_comparison import BankComparisonResult

        r = BankComparisonResult(
            kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
            monotonicity=Decimal("0"), seed=1, outside_mid_ratio=Decimal("0.90"),
            delta_idle=Decimal("0.50"), phi_idle=Decimal("0.50"),
            idle_run_id="i", idle_status="completed",
            delta_lend=Decimal("0.30"), phi_lend=Decimal("0.70"),
            lend_run_id="l", lend_status="completed",
        )
        assert r.bank_lending_effect == Decimal("0.20")

    def test_bank_lending_effect_none(self):
        """bank_lending_effect is None when delta_idle is None."""
        from bilancio.experiments.bank_comparison import BankComparisonResult

        r = BankComparisonResult(
            kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
            monotonicity=Decimal("0"), seed=1, outside_mid_ratio=Decimal("0.90"),
            delta_idle=None, phi_idle=None,
            idle_run_id="i", idle_status="failed",
            delta_lend=Decimal("0.30"), phi_lend=Decimal("0.70"),
            lend_run_id="l", lend_status="completed",
        )
        assert r.bank_lending_effect is None

    def test_bank_lending_relief_ratio(self):
        """bank_lending_relief_ratio = effect / delta_idle."""
        from bilancio.experiments.bank_comparison import BankComparisonResult

        r = BankComparisonResult(
            kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
            monotonicity=Decimal("0"), seed=1, outside_mid_ratio=Decimal("0.90"),
            delta_idle=Decimal("0.50"), phi_idle=Decimal("0.50"),
            idle_run_id="i", idle_status="completed",
            delta_lend=Decimal("0.30"), phi_lend=Decimal("0.70"),
            lend_run_id="l", lend_status="completed",
        )
        assert r.bank_lending_relief_ratio == Decimal("0.40")

    def test_system_loss_bank_lending_effect(self):
        """system_loss_bank_lending_effect = system_loss_pct_idle - system_loss_pct_lend."""
        from bilancio.experiments.bank_comparison import BankComparisonResult

        r = BankComparisonResult(
            kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
            monotonicity=Decimal("0"), seed=1, outside_mid_ratio=Decimal("0.90"),
            delta_idle=Decimal("0.50"), phi_idle=Decimal("0.50"),
            idle_run_id="i", idle_status="completed",
            delta_lend=Decimal("0.30"), phi_lend=Decimal("0.70"),
            lend_run_id="l", lend_status="completed",
            system_loss_pct_idle=0.08, system_loss_pct_lend=0.03,
        )
        assert r.system_loss_bank_lending_effect == pytest.approx(0.05)


# =============================================================================
# 43. Comparison _run_pair
# =============================================================================


class TestComparisonRunPair:
    """Tests for ComparisonSweepRunner._run_pair."""

    def test_run_pair_returns_result(self, tmp_path: Path):
        """_run_pair returns a ComparisonResult with control and treatment data."""
        from bilancio.experiments.comparison import ComparisonSweepConfig, ComparisonSweepRunner

        config = ComparisonSweepConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")], mus=[Decimal("0")],
        )
        runner = ComparisonSweepRunner(config, tmp_path)

        control_summary = _make_summary(
            run_id="c1", phase="passive", kappa=Decimal("1"),
            concentration=Decimal("1"), mu=Decimal("0"), monotonicity=Decimal("0"),
            delta_total=Decimal("0.40"), phi_total=Decimal("0.60"),
        )
        treatment_summary = _make_summary(
            run_id="t1", phase="active", kappa=Decimal("1"),
            concentration=Decimal("1"), mu=Decimal("0"), monotonicity=Decimal("0"),
            delta_total=Decimal("0.20"), phi_total=Decimal("0.80"),
        )

        mock_control_runner = MagicMock()
        mock_control_runner._execute_run = MagicMock(return_value=control_summary)
        mock_control_runner._next_seed = MagicMock(return_value=42)
        mock_treatment_runner = MagicMock()
        mock_treatment_runner._execute_run = MagicMock(return_value=treatment_summary)

        with patch.object(runner, "_get_control_runner", return_value=mock_control_runner), \
             patch.object(runner, "_get_treatment_runner", return_value=mock_treatment_runner):
            result = runner._run_pair(Decimal("1"), Decimal("1"), Decimal("0"), Decimal("0"))

        assert result.delta_control == Decimal("0.40")
        assert result.delta_treatment == Decimal("0.20")


# =============================================================================
# 44. BalancedComparisonResult - None path for relief ratio and adjusted effects
# =============================================================================


class TestBalancedResultNonePaths:
    """Tests for BalancedComparisonResult None returns on missing data."""

    def _make_result(self, **overrides):
        from bilancio.experiments.balanced_comparison import BalancedComparisonResult
        defaults = dict(
            kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
            monotonicity=Decimal("0"), seed=1,
            face_value=Decimal("20"), outside_mid_ratio=Decimal("0.90"),
            big_entity_share=Decimal("0.25"),
            delta_passive=Decimal("0.50"), phi_passive=Decimal("0.50"),
            passive_run_id="p1", passive_status="completed",
            delta_active=Decimal("0.30"), phi_active=Decimal("0.70"),
            active_run_id="a1", active_status="completed",
            n_defaults_passive=5, n_defaults_active=3,
            cascade_fraction_passive=Decimal("0.10"),
            cascade_fraction_active=Decimal("0.05"),
        )
        defaults.update(overrides)
        return BalancedComparisonResult(**defaults)

    def test_trading_relief_ratio_none_when_delta_missing(self):
        """trading_relief_ratio is None when delta_passive is None."""
        r = self._make_result(delta_passive=None)
        assert r.trading_relief_ratio is None

    def test_adjusted_trading_effect_none_when_trading_effect_none(self):
        """adjusted_trading_effect is None when trading_effect is None (delta missing)."""
        r = self._make_result(delta_active=None, intermediary_loss_pct_passive=0.01, intermediary_loss_pct_active=0.02)
        assert r.adjusted_trading_effect is None

    def test_adjusted_lending_effect_none_when_lending_effect_none(self):
        """adjusted_lending_effect is None when lending_effect is None."""
        r = self._make_result(intermediary_loss_pct_passive=0.01, intermediary_loss_pct_lender=0.02)
        # delta_lender is None by default, so lending_effect is None
        assert r.adjusted_lending_effect is None

    def test_adjusted_combined_effect_none_when_combined_none(self):
        """adjusted_combined_effect is None when combined_effect is None."""
        r = self._make_result(intermediary_loss_pct_passive=0.01, intermediary_loss_pct_dealer_lender=0.02)
        assert r.adjusted_combined_effect is None

    def test_adjusted_bank_passive_none_when_bank_passive_none(self):
        """adjusted_bank_passive_effect is None when bank_passive_effect is None."""
        r = self._make_result(intermediary_loss_pct_passive=0.01, intermediary_loss_pct_bank_passive=0.02)
        assert r.adjusted_bank_passive_effect is None

    def test_adjusted_bank_dealer_none_when_bank_dealer_none(self):
        """adjusted_bank_dealer_effect is None when bank_dealer_effect is None."""
        r = self._make_result(intermediary_loss_pct_passive=0.01, intermediary_loss_pct_bank_dealer=0.02)
        assert r.adjusted_bank_dealer_effect is None

    def test_adjusted_bank_dealer_nbfi_none_when_base_none(self):
        """adjusted_bank_dealer_nbfi_effect is None when bank_dealer_nbfi_effect is None."""
        r = self._make_result(intermediary_loss_pct_passive=0.01, intermediary_loss_pct_bank_dealer_nbfi=0.02)
        assert r.adjusted_bank_dealer_nbfi_effect is None


# =============================================================================
# 45. Balanced _write_comparison_csv
# =============================================================================


class TestBalancedWriteComparisonCsv:
    """Tests for BalancedComparisonRunner._write_comparison_csv."""

    def test_writes_csv_with_results(self, tmp_path: Path):
        """_write_comparison_csv writes CSV with all key fields."""
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonResult,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            BalancedComparisonResult(
                kappa=Decimal("1"), concentration=Decimal("1"), mu=Decimal("0"),
                monotonicity=Decimal("0"), seed=42,
                face_value=Decimal("20"), outside_mid_ratio=Decimal("0.90"),
                big_entity_share=Decimal("0.25"),
                delta_passive=Decimal("0.50"), phi_passive=Decimal("0.50"),
                passive_run_id="p1", passive_status="completed",
                delta_active=Decimal("0.30"), phi_active=Decimal("0.70"),
                active_run_id="a1", active_status="completed",
                n_defaults_passive=5, n_defaults_active=3,
                cascade_fraction_passive=Decimal("0.10"),
                cascade_fraction_active=Decimal("0.05"),
            ),
        ]

        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        assert csv_path.exists()

        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["kappa"] == "1"
        assert rows[0]["delta_passive"] == "0.50"
        assert rows[0]["delta_active"] == "0.30"
        assert rows[0]["trading_effect"] == "0.20"


# =============================================================================
# 46. Balanced _make_key and _next_seed
# =============================================================================


class TestBalancedMakeKeyAndSeed:
    """Tests for BalancedComparisonRunner._make_key and _next_seed."""

    def test_balanced_make_key(self, tmp_path: Path):
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )
        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        key = runner._make_key(Decimal("1"), Decimal("2"), Decimal("0"), Decimal("0"), Decimal("0.9"))
        assert key == ("1", "2", "0", "0", "0.9", "ring")
        key2 = runner._make_key(Decimal("1"), Decimal("2"), Decimal("0"), Decimal("0"), Decimal("0.9"), topology="k_regular")
        assert key2 == ("1", "2", "0", "0", "0.9", "k_regular")

    def test_balanced_next_seed(self, tmp_path: Path):
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )
        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
            base_seed=200,
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        s1 = runner._next_seed()
        s2 = runner._next_seed()
        assert s1 == 200
        assert s2 == 201


# =============================================================================
# 47. Balanced _format_time
# =============================================================================


class TestBalancedFormatTime:
    """Tests for BalancedComparisonRunner._format_time."""

    def test_seconds(self, tmp_path: Path):
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )
        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(45) == "45s"

    def test_minutes(self, tmp_path: Path):
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )
        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(150) == "2.5m"

    def test_hours(self, tmp_path: Path):
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )
        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(3720) == "1h 2m"


# =============================================================================
# 48. Balanced _load_existing_results
# =============================================================================


class TestBalancedLoadExistingResults:
    """Tests for BalancedComparisonRunner._load_existing_results."""

    def test_loads_existing_csv(self, tmp_path: Path):
        """BalancedComparisonRunner loads existing CSV and advances seed counter."""
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )

        agg_dir = tmp_path / "aggregate"
        agg_dir.mkdir(parents=True)
        csv_path = agg_dir / "comparison.csv"

        fields = BalancedComparisonRunner.COMPARISON_FIELDS
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            row = {f: "" for f in fields}
            row.update({
                "kappa": "1", "concentration": "1", "mu": "0",
                "monotonicity": "0", "outside_mid_ratio": "0.90",
                "seed": "75",
                "delta_passive": "0.30", "delta_active": "0.20",
                "trading_effect": "0.10",
                "passive_run_id": "p_x", "passive_status": "completed",
                "active_run_id": "a_x", "active_status": "completed",
            })
            writer.writerow(row)

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        assert runner.seed_counter >= 76
        assert len(runner._completed_counts) > 0

    def test_skips_load_when_no_csv(self, tmp_path: Path):
        """BalancedComparisonRunner handles missing CSV gracefully."""
        from bilancio.experiments.balanced_comparison import (
            BalancedComparisonConfig,
            BalancedComparisonRunner,
        )

        config = BalancedComparisonConfig(
            kappas=[Decimal("1")], concentrations=[Decimal("1")],
            mus=[Decimal("0")], outside_mid_ratios=[Decimal("0.90")],
        )
        runner = BalancedComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        assert runner.seed_counter == config.base_seed
        assert len(runner._completed_counts) == 0
