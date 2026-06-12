"""Sweep-pipeline tests for the central counterparty via novation (Plan 061).

Covers:
- the `ccp_totals` metrics helper on synthetic event lists,
- the new `cascade_depth_max` metric (incl. origin_* attribution on novated legs),
- run-level metric wiring (compute_run_level_metrics),
- CLI wiring of `--clearing-ccp` / `--ccp-fund-share` down to the runner,
- mutual exclusion with `--clearing` and `--clearing-certificates`,
- the runner's clearinghouse block emission (mode "ccp"),
- the new aggregate results columns (fund_drawdowns_total, vmgh_haircut_total,
  ccp_member_defaults, cascade_depth_max) with 0.0 fallback for legacy registries.

The kernel side (v2 novation, CCP fund/waterfall/VMGH, ClearinghouseConfig
mode "ccp") is implemented separately; tests that would require a live kernel
run are guarded so they skip until the kernel lands.
"""

from __future__ import annotations

from decimal import Decimal
from typing import get_args
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bilancio.analysis.metrics import cascade_depth_max, ccp_totals
from bilancio.analysis.report import aggregate_runs, compute_run_level_metrics
from bilancio.ui.cli import cli


def _kernel_supports_ccp() -> bool:
    """True once the v2 kernel's ClearinghouseConfig accepts mode='ccp'."""
    from bilancio.config.models import ClearinghouseConfig

    annotation = ClearinghouseConfig.model_fields["mode"].annotation
    return "ccp" in get_args(annotation)


# ---------------------------------------------------------------------------
# ccp_totals helper
# ---------------------------------------------------------------------------


class TestCCPTotals:
    def test_empty_events_returns_zeros(self):
        assert ccp_totals([]) == (Decimal("0"), Decimal("0"), 0)

    def test_non_ccp_events_ignored(self):
        events = [
            {"kind": "PayableCreated", "amount": "100", "due_day": 1},
            {"kind": "PayableSettled", "day": 1, "amount": "100"},
            {"kind": "CCPFundContribution", "day": 0, "member": "H1", "amount": "5"},
            {"kind": "CCPFundReplenished", "day": 2, "member": "H2", "amount": "3"},
        ]
        # Contributions/replenishments move fund accounting but are not
        # drawdowns, haircuts, or defaults.
        assert ccp_totals(events) == (Decimal("0"), Decimal("0"), 0)

    def test_drawdowns_sum_own_and_mutualized_tranches(self):
        events = [
            {
                "kind": "CCPFundDrawdown",
                "day": 2,
                "member": "H1",
                "own_tranche": "25",
                "mutualized_tranche": "35",
                "vmgh_residual": "0",
                "fund_total": "40",
            },
            {
                "kind": "CCPFundDrawdown",
                "day": 4,
                "member": "H3",
                "own_tranche": "10",
                "mutualized_tranche": "0",
                "vmgh_residual": "5",
                "fund_total": "30",
            },
        ]
        drawdowns, vmgh, defaults = ccp_totals(events)
        # vmgh_residual is NOT part of fund drawdowns — it is captured via
        # the VMGHHaircutApplied events it causes.
        assert drawdowns == Decimal("70")
        assert vmgh == Decimal("0")
        assert defaults == 0

    def test_vmgh_sums_haircut_payloads(self):
        events = [
            {
                "kind": "VMGHHaircutApplied",
                "day": 2,
                "creditor": "H4",
                "face": "120",
                "paid": "96",
                "haircut": "24",
                "haircut_factor": "0.8",
            },
            {
                "kind": "VMGHHaircutApplied",
                "day": 2,
                "creditor": "H2",
                "face": "30",
                "paid": "24",
                "haircut": "6",
                "haircut_factor": "0.8",
            },
        ]
        drawdowns, vmgh, defaults = ccp_totals(events)
        assert drawdowns == Decimal("0")
        assert vmgh == Decimal("30")
        assert defaults == 0

    def test_member_defaults_counts_distinct_agents(self):
        events = [
            {"kind": "AgentDefaulted", "day": 2, "agent": "H1"},
            {"kind": "AgentDefaulted", "day": 2, "agent": "H1"},  # duplicate
            {"kind": "AgentDefaulted", "day": 3, "agent": "H3"},
            {"kind": "AgentDefaulted", "day": 4, "frm": "H5"},  # legacy key
        ]
        drawdowns, vmgh, defaults = ccp_totals(events)
        assert defaults == 3

    def test_mixed_stream(self):
        events = [
            {"kind": "CCPFundContribution", "day": 0, "member": "H1", "amount": "25"},
            {"kind": "AgentDefaulted", "day": 2, "agent": "H1"},
            {
                "kind": "CCPFundDrawdown",
                "day": 2,
                "member": "H1",
                "own_tranche": "25",
                "mutualized_tranche": "5",
                "vmgh_residual": "30",
                "fund_total": "0",
            },
            {
                "kind": "VMGHHaircutApplied",
                "day": 2,
                "creditor": "H4",
                "face": "150",
                "paid": "120",
                "haircut": "30",
                "haircut_factor": "0.8",
            },
        ]
        assert ccp_totals(events) == (Decimal("30"), Decimal("30"), 1)


# ---------------------------------------------------------------------------
# cascade_depth_max metric
# ---------------------------------------------------------------------------


def _payable(debtor, creditor, *, origin_debtor=None, origin_creditor=None):
    e = {"kind": "PayableCreated", "debtor": debtor, "creditor": creditor, "amount": "10", "due_day": 1}
    if origin_debtor is not None:
        e["origin_debtor"] = origin_debtor
    if origin_creditor is not None:
        e["origin_creditor"] = origin_creditor
    return e


def _default(agent, day=1):
    return {"kind": "AgentDefaulted", "day": day, "agent": agent}


class TestCascadeDepthMax:
    def test_no_defaults_returns_zero(self):
        events = [_payable("A", "B"), _payable("B", "C")]
        assert cascade_depth_max(events) == 0

    def test_empty_events_returns_zero(self):
        assert cascade_depth_max([]) == 0

    def test_isolated_default_returns_one(self):
        events = [
            _payable("A", "B"),
            _payable("B", "C"),
            _default("B"),
        ]
        # B's debtor A never defaulted -> isolated.
        assert cascade_depth_max(events) == 1

    def test_chain_of_three(self):
        # Ring precedence A -> B -> C (A owes B, B owes C), defaults in order.
        events = [
            _payable("A", "B"),
            _payable("B", "C"),
            _default("A", day=1),
            _default("B", day=2),
            _default("C", day=3),
        ]
        assert cascade_depth_max(events) == 3

    def test_creditor_defaulting_first_breaks_precedence(self):
        # C defaults BEFORE its debtor B: no edge B -> C; both isolated.
        events = [
            _payable("B", "C"),
            _default("C", day=1),
            _default("B", day=2),
        ]
        assert cascade_depth_max(events) == 1

    def test_parallel_chains_take_longest(self):
        # Chain 1: A -> B (depth 2). Chain 2: X alone (depth 1).
        events = [
            _payable("A", "B"),
            _payable("X", "Y"),
            _default("A", day=1),
            _default("X", day=1),
            _default("B", day=2),
        ]
        assert cascade_depth_max(events) == 2

    def test_duplicate_default_events_count_once(self):
        events = [
            _payable("A", "B"),
            _default("A", day=1),
            _default("A", day=1),
            _default("B", day=2),
            _default("B", day=3),
        ]
        assert cascade_depth_max(events) == 2

    def test_novated_run_uses_origin_pair(self):
        # Novated star: A -> CCP1 and CCP1 -> B legs, both carrying the
        # origin pair (A, B). Defaults A then B must attribute the edge to
        # the ORIGIN pair (depth 2), not to CCP1 (which never defaults).
        events = [
            _payable("A", "CCP1", origin_debtor="A", origin_creditor="B"),
            _payable("CCP1", "B", origin_debtor="A", origin_creditor="B"),
            _payable("B", "CCP1", origin_debtor="B", origin_creditor="C"),
            _payable("CCP1", "C", origin_debtor="B", origin_creditor="C"),
            _default("A", day=1),
            _default("B", day=2),
        ]
        assert cascade_depth_max(events) == 2

    def test_non_novated_star_attributes_to_ccp_only(self):
        # Same star WITHOUT origin keys: B's only debtor is CCP1, which never
        # defaults, so the two defaults are independent (depth 1). This is
        # exactly the misattribution the origin_* keys exist to fix.
        events = [
            _payable("A", "CCP1"),
            _payable("CCP1", "B"),
            _default("A", day=1),
            _default("B", day=2),
        ]
        assert cascade_depth_max(events) == 1


# ---------------------------------------------------------------------------
# Run-level metric wiring
# ---------------------------------------------------------------------------


class TestRunLevelMetricsWiring:
    def test_ccp_keys_present_and_zero_without_events(self):
        summary = compute_run_level_metrics([])
        assert summary["fund_drawdowns_total"] == 0.0
        assert summary["vmgh_haircut_total"] == 0.0
        assert summary["ccp_member_defaults"] == 0
        assert summary["cascade_depth_max"] == 0

    def test_ccp_key_types(self):
        events = [
            {
                "kind": "CCPFundDrawdown",
                "day": 2,
                "member": "H1",
                "own_tranche": "25",
                "mutualized_tranche": "35",
                "vmgh_residual": "0",
                "fund_total": "40",
            },
            {
                "kind": "VMGHHaircutApplied",
                "day": 2,
                "creditor": "H4",
                "face": "120",
                "paid": "96",
                "haircut": "24",
                "haircut_factor": "0.8",
            },
            {"kind": "AgentDefaulted", "day": 2, "agent": "H1"},
        ]
        summary = compute_run_level_metrics(events)
        assert summary["fund_drawdowns_total"] == 60.0
        assert summary["vmgh_haircut_total"] == 24.0
        assert summary["ccp_member_defaults"] == 1
        assert summary["cascade_depth_max"] == 1
        assert isinstance(summary["fund_drawdowns_total"], float)
        assert isinstance(summary["vmgh_haircut_total"], float)
        assert isinstance(summary["ccp_member_defaults"], int)
        assert isinstance(summary["cascade_depth_max"], int)

    def test_cascade_depth_computed_for_non_ccp_runs(self):
        # cascade_depth_max needs no CCP events: a plain ring chain must
        # produce a depth so the with/without-CCP comparison exists.
        events = [
            {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": "10", "due_day": 1},
            {"kind": "PayableCreated", "debtor": "B", "creditor": "C", "amount": "10", "due_day": 1},
            {"kind": "AgentDefaulted", "day": 1, "agent": "A"},
            {"kind": "AgentDefaulted", "day": 2, "agent": "B"},
            {"kind": "AgentDefaulted", "day": 3, "agent": "C"},
        ]
        summary = compute_run_level_metrics(events)
        assert summary["cascade_depth_max"] == 3
        assert summary["fund_drawdowns_total"] == 0.0
        assert summary["vmgh_haircut_total"] == 0.0


# ---------------------------------------------------------------------------
# CLI wiring (--clearing-ccp / --ccp-fund-share)
# ---------------------------------------------------------------------------


class TestSweepRingCCPCLI:
    def test_sweep_ring_help_lists_ccp_options(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "ring", "--help"])
        assert result.exit_code == 0
        assert "--clearing-ccp" in result.output
        assert "--ccp-fund-share" in result.output

    @patch("bilancio.ui.cli.sweep.aggregate_runs")
    @patch("bilancio.ui.cli.sweep.render_dashboard")
    @patch("bilancio.ui.cli.sweep.RingSweepRunner")
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-ccp-job")
    def test_clearing_ccp_flag_passed_to_runner(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_cls,
        mock_render,
        mock_agg,
        tmp_path,
    ):
        mock_create_jm.return_value = MagicMock()
        mock_runner = MagicMock()
        mock_runner.registry_dir = tmp_path / "registry"
        mock_runner.aggregate_dir = tmp_path / "aggregate"
        (tmp_path / "registry").mkdir()
        (tmp_path / "aggregate").mkdir()
        mock_runner_cls.return_value = mock_runner

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "ring",
                "--out-dir",
                str(tmp_path / "ring_out"),
                "--clearing-ccp",
                "--ccp-fund-share",
                "0.1",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        kwargs = mock_runner_cls.call_args[1]
        assert kwargs["clearing_ccp"] is True
        assert kwargs["ccp_fund_share"] == Decimal("0.1")
        # ccp does NOT imply the netting block
        assert kwargs["clearing_enabled"] is False
        assert kwargs["clearing_certificates"] is False

    @patch("bilancio.ui.cli.sweep.aggregate_runs")
    @patch("bilancio.ui.cli.sweep.render_dashboard")
    @patch("bilancio.ui.cli.sweep.RingSweepRunner")
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-ccp-defaults")
    def test_ccp_defaults_match_plan(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_cls,
        mock_render,
        mock_agg,
        tmp_path,
    ):
        mock_create_jm.return_value = MagicMock()
        mock_runner = MagicMock()
        mock_runner.registry_dir = tmp_path / "registry"
        mock_runner.aggregate_dir = tmp_path / "aggregate"
        (tmp_path / "registry").mkdir()
        (tmp_path / "aggregate").mkdir()
        mock_runner_cls.return_value = mock_runner

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "ring",
                "--out-dir",
                str(tmp_path / "ring_out"),
                "--clearing-ccp",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        kwargs = mock_runner_cls.call_args[1]
        assert kwargs["clearing_ccp"] is True
        assert kwargs["ccp_fund_share"] == Decimal("0.05")

    @patch("bilancio.ui.cli.sweep.aggregate_runs")
    @patch("bilancio.ui.cli.sweep.render_dashboard")
    @patch("bilancio.ui.cli.sweep.RingSweepRunner")
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-ccp-off")
    def test_ccp_off_by_default(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_cls,
        mock_render,
        mock_agg,
        tmp_path,
    ):
        mock_create_jm.return_value = MagicMock()
        mock_runner = MagicMock()
        mock_runner.registry_dir = tmp_path / "registry"
        mock_runner.aggregate_dir = tmp_path / "aggregate"
        (tmp_path / "registry").mkdir()
        (tmp_path / "aggregate").mkdir()
        mock_runner_cls.return_value = mock_runner

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["sweep", "ring", "--out-dir", str(tmp_path / "ring_out")],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        kwargs = mock_runner_cls.call_args[1]
        assert kwargs["clearing_ccp"] is False
        assert kwargs["ccp_fund_share"] == Decimal("0.05")

    @pytest.mark.parametrize(
        "conflicting_flag",
        ["--clearing", "--clearing-certificates"],
    )
    def test_ccp_mutually_exclusive_with_other_modes(self, tmp_path, conflicting_flag):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "ring",
                "--out-dir",
                str(tmp_path / "ring_out"),
                "--clearing-ccp",
                conflicting_flag,
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output


# ---------------------------------------------------------------------------
# Runner scenario block emission
# ---------------------------------------------------------------------------


class TestRunnerScenarioBlock:
    """RingSweepRunner emits the ccp clearinghouse block."""

    def _make_runner(self, tmp_path, **kwargs):
        from bilancio.experiments.ring import RingSweepRunner

        return RingSweepRunner(
            out_dir=tmp_path,
            name_prefix="CCPTest",
            n_agents=4,
            maturity_days=3,
            Q_total=Decimal("100"),
            liquidity_mode="uniform",
            liquidity_agent=None,
            base_seed=42,
            **kwargs,
        )

    def test_ccp_block_in_prepared_scenario(self, tmp_path):
        runner = self._make_runner(
            tmp_path,
            clearing_ccp=True,
            ccp_fund_share=Decimal("0.05"),
        )
        prepared = runner._prepare_run(
            "grid", Decimal("0.5"), Decimal("1"), Decimal("0.75"), Decimal("0"), 42
        )
        block = prepared.scenario_config.get("clearinghouse")
        assert block == {
            "enabled": True,
            "mode": "ccp",
            "ccp_fund_share": 0.05,
        }

    def test_ccp_does_not_set_clearing_enabled(self, tmp_path):
        # The kernel runs netting automatically in ccp mode; the runner must
        # not turn on the separate netting flag.
        runner = self._make_runner(tmp_path, clearing_ccp=True)
        assert runner.clearing_ccp is True
        assert runner.clearing_enabled is False
        assert runner.clearing_certificates is False

    def test_ccp_fund_share_threaded_into_block(self, tmp_path):
        runner = self._make_runner(
            tmp_path, clearing_ccp=True, ccp_fund_share=Decimal("0.1")
        )
        prepared = runner._prepare_run(
            "grid", Decimal("0.5"), Decimal("1"), Decimal("0"), Decimal("0"), 42
        )
        assert prepared.scenario_config["clearinghouse"]["ccp_fund_share"] == 0.1

    def test_no_block_when_ccp_off(self, tmp_path):
        runner = self._make_runner(tmp_path)
        prepared = runner._prepare_run(
            "grid", Decimal("0.5"), Decimal("1"), Decimal("0"), Decimal("0"), 42
        )
        assert "clearinghouse" not in prepared.scenario_config

    def test_ccp_param_in_registry_base_params(self, tmp_path):
        runner = self._make_runner(tmp_path, clearing_ccp=True)
        prepared = runner._prepare_run(
            "grid", Decimal("0.5"), Decimal("1"), Decimal("0"), Decimal("0"), 42
        )
        assert prepared.base_params["clearing_ccp"] is True
        assert prepared.base_params["clearing_enabled"] is False

    @pytest.mark.parametrize(
        "conflict_kwargs",
        [
            {"clearing_enabled": True},
            {"clearing_certificates": True},
        ],
    )
    def test_runner_rejects_ccp_combined_with_other_modes(self, tmp_path, conflict_kwargs):
        with pytest.raises(ValueError, match="mutually exclusive"):
            self._make_runner(tmp_path, clearing_ccp=True, **conflict_kwargs)

    def test_netting_block_unchanged_by_plan_061(self, tmp_path):
        runner = self._make_runner(tmp_path, clearing_enabled=True)
        prepared = runner._prepare_run(
            "grid", Decimal("0.5"), Decimal("1"), Decimal("0"), Decimal("0"), 42
        )
        assert prepared.scenario_config.get("clearinghouse") == {
            "enabled": True,
            "mode": "netting",
        }


# ---------------------------------------------------------------------------
# Aggregate results columns
# ---------------------------------------------------------------------------


def test_aggregate_runs_carries_ccp_columns(tmp_path):
    registry_dir = tmp_path / "registry"
    out_dir = tmp_path / "runs" / "grid_ccp1" / "out"
    aggregate_dir = tmp_path / "aggregate"
    registry_dir.mkdir()
    out_dir.mkdir(parents=True)
    aggregate_dir.mkdir()

    metrics_csv = out_dir / "metrics.csv"
    metrics_csv.write_text(
        "day,S_t,phi_t,delta_t,face_netted_t\n"
        "1,100,1,0,0\n"
    )

    registry_csv = registry_dir / "experiments.csv"
    registry_csv.write_text(
        "run_id,phase,seed,n_agents,kappa,concentration,mu,monotonicity,S1,L0,"
        "metrics_csv,status,time_to_stability,n_defaults,cascade_fraction,"
        "fund_drawdowns_total,vmgh_haircut_total,ccp_member_defaults,cascade_depth_max\n"
        "grid_ccp1,grid,42,5,0.5,1,0.75,0,100,50,"
        "../runs/grid_ccp1/out/metrics.csv,completed,2,1,,"
        "60.0,24.0,1,1\n"
    )

    results_csv = aggregate_dir / "results.csv"
    rows = aggregate_runs(registry_csv, results_csv)

    assert rows
    row = rows[0]
    assert row["fund_drawdowns_total"] == "60.0"
    assert row["vmgh_haircut_total"] == "24.0"
    assert row["ccp_member_defaults"] == "1"
    assert row["cascade_depth_max"] == "1"

    header = results_csv.read_text().splitlines()[0]
    assert "fund_drawdowns_total" in header
    assert "vmgh_haircut_total" in header
    assert "ccp_member_defaults" in header
    assert "cascade_depth_max" in header


def test_aggregate_runs_defaults_to_zero_for_legacy_registries(tmp_path):
    """Registries written before Plan 061 lack the CCP columns."""
    registry_dir = tmp_path / "registry"
    out_dir = tmp_path / "runs" / "grid_old1" / "out"
    aggregate_dir = tmp_path / "aggregate"
    registry_dir.mkdir()
    out_dir.mkdir(parents=True)
    aggregate_dir.mkdir()

    metrics_csv = out_dir / "metrics.csv"
    metrics_csv.write_text("day,S_t,phi_t,delta_t\n1,100,1,0\n")

    registry_csv = registry_dir / "experiments.csv"
    registry_csv.write_text(
        "run_id,phase,seed,n_agents,kappa,concentration,mu,monotonicity,S1,L0,"
        "metrics_csv,status,time_to_stability,n_defaults,cascade_fraction\n"
        "grid_old1,grid,42,5,0.5,1,0,0,100,50,"
        "../runs/grid_old1/out/metrics.csv,completed,2,0,\n"
    )

    results_csv = aggregate_dir / "results.csv"
    rows = aggregate_runs(registry_csv, results_csv)

    assert rows
    row = rows[0]
    assert row["fund_drawdowns_total"] == 0.0
    assert row["vmgh_haircut_total"] == 0.0
    assert row["ccp_member_defaults"] == 0.0
    assert row["cascade_depth_max"] == 0.0


# ---------------------------------------------------------------------------
# CCPProfile
# ---------------------------------------------------------------------------


class TestCCPProfile:
    def test_defaults_match_plan(self):
        from bilancio.decision import CCPProfile

        profile = CCPProfile()
        assert profile.ccp_fund_share == Decimal("0.05")
        assert profile.vmgh_enabled is True
        assert profile.replenishment_enabled is True
        assert profile.ccp_can_fail is False

    def test_vmgh_disabled_rejected(self):
        from bilancio.decision.profiles import CCPProfile

        with pytest.raises(ValueError, match="vmgh_enabled"):
            CCPProfile(vmgh_enabled=False)

    def test_ccp_can_fail_rejected(self):
        from bilancio.decision.profiles import CCPProfile

        with pytest.raises(ValueError, match="ccp_can_fail"):
            CCPProfile(ccp_can_fail=True)

    def test_invalid_fund_share_rejected(self):
        from bilancio.decision.profiles import CCPProfile

        with pytest.raises(ValueError, match="ccp_fund_share"):
            CCPProfile(ccp_fund_share=Decimal("1.0"))
        with pytest.raises(ValueError, match="ccp_fund_share"):
            CCPProfile(ccp_fund_share=Decimal("-0.01"))

    def test_frozen(self):
        from bilancio.decision.profiles import CCPProfile

        profile = CCPProfile()
        with pytest.raises(AttributeError):
            profile.ccp_fund_share = Decimal("0.1")


# ---------------------------------------------------------------------------
# Termination impact events
# ---------------------------------------------------------------------------


def test_ccp_events_counted_as_impactful():
    from bilancio.engines.termination import IMPACT_EVENTS

    assert "CCPFundDrawdown" in IMPACT_EVENTS
    assert "VMGHHaircutApplied" in IMPACT_EVENTS


# ---------------------------------------------------------------------------
# Tiny end-to-end run (requires the kernel side of Plan 061)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _kernel_supports_ccp(),
    # KERNEL DEPENDENCY: the sibling Plan 061 kernel change widens
    # ClearinghouseConfig.mode to include "ccp". Until it lands, scenario
    # validation rejects the ccp block, so this end-to-end test is skipped.
    # The orchestrator should unskip after integration.
    reason="v2 kernel does not yet accept clearinghouse mode 'ccp'",
)
def test_end_to_end_tiny_ccp_run(tmp_path):
    """Smallest possible local run: the ccp block flows through the runner,
    the run completes, and the registry carries the four CCP metrics."""
    import csv

    from bilancio.experiments.ring import RingSweepRunner

    runner = RingSweepRunner(
        out_dir=tmp_path,
        name_prefix="CCPE2E",
        n_agents=4,
        maturity_days=3,
        Q_total=Decimal("100"),
        liquidity_mode="uniform",
        liquidity_agent=None,
        base_seed=42,
        default_handling="expel-agent",
        clearing_ccp=True,
    )
    summaries = runner.run_grid(
        [Decimal("0.5")], [Decimal("1")], [Decimal("0.75")], [Decimal("0")]
    )
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.fund_drawdowns_total >= 0.0
    assert summary.vmgh_haircut_total >= 0.0
    assert summary.ccp_member_defaults >= 0
    assert summary.cascade_depth_max >= 0

    registry_csv = tmp_path / "registry" / "experiments.csv"
    with registry_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert rows[0]["status"] == "completed"
    for column in (
        "fund_drawdowns_total",
        "vmgh_haircut_total",
        "ccp_member_defaults",
        "cascade_depth_max",
    ):
        assert column in rows[0]
        assert rows[0][column] != ""


class TestOriginAwareDeltaMetrics:
    """Novated obligations count once, on the out-leg (end-creditor claim)."""

    def _novated_pair(self, pid: str, debtor: str, creditor: str, amount: str, due_day: int):
        return [
            {
                "kind": "PayableCreated",
                "contract_id": pid,
                "debtor": debtor,
                "creditor": "CCP1",
                "origin_debtor": debtor,
                "origin_creditor": creditor,
                "amount": amount,
                "due_day": due_day,
            },
            {
                "kind": "PayableCreated",
                "contract_id": f"{pid}_out",
                "debtor": "CCP1",
                "creditor": creditor,
                "origin_debtor": debtor,
                "origin_creditor": creditor,
                "amount": amount,
                "due_day": due_day,
            },
        ]

    def test_dues_count_each_novated_obligation_once(self):
        from bilancio.analysis.metrics import dues_for_day

        events = self._novated_pair("PAY_1", "H1", "H2", "100", 1)
        dues = dues_for_day(events, 1)
        assert len(dues) == 1
        assert dues[0]["creditor"] == "H2"
        assert dues[0]["pid"] == "PAY_1_out"

    def test_gross_face_counts_each_novated_obligation_once(self):
        from bilancio.analysis.metrics import netting_totals

        events = self._novated_pair("PAY_1", "H1", "H2", "100", 1)
        gross, netted = netting_totals(events)
        assert gross == Decimal("100")
        assert netted == Decimal("0")

    def test_phi_credits_out_leg_settlement_only(self):
        """In-leg default with fund-paid out-leg: the obligation counts as
        honored — delta measures end-creditor losses."""
        from bilancio.analysis.metrics import dues_for_day, phi_delta

        events = self._novated_pair("PAY_1", "H1", "H2", "100", 1) + [
            # in-leg defaulted (no settled event); out-leg paid by the CCP
            {"kind": "PayableSettled", "day": 1, "pid": "PAY_1_out", "amount": "100"},
        ]
        dues = dues_for_day(events, 1)
        phi, delta = phi_delta(events, dues, 1)
        assert phi == Decimal("1")
        assert delta == Decimal("0")

    def test_non_novated_events_unaffected(self):
        from bilancio.analysis.metrics import dues_for_day

        events = [
            {
                "kind": "PayableCreated",
                "contract_id": "PAY_9",
                "debtor": "H1",
                "creditor": "H2",
                "amount": "50",
                "due_day": 1,
            }
        ]
        assert len(dues_for_day(events, 1)) == 1
