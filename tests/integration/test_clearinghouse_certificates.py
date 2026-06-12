"""Sweep-pipeline tests for clearinghouse loan certificates (Plan 060).

Covers:
- the `certificate_totals` metrics helper on synthetic event lists,
- run-level metric wiring (compute_run_level_metrics),
- CLI wiring of `--clearing-certificates` down to the generator/scenario config,
- the new aggregate results columns (certificates_issued_total,
  certificates_outstanding_peak, cert_default_losses).

The kernel side (v2 CertificateFacilityPhase, ClearinghouseConfig mode
"certificates") is implemented separately; tests that would require a live
kernel run are guarded so they skip until the kernel lands.
"""

from __future__ import annotations

from decimal import Decimal
from typing import get_args
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bilancio.analysis.metrics import certificate_totals
from bilancio.analysis.report import aggregate_runs, compute_run_level_metrics
from bilancio.ui.cli import cli


def _kernel_supports_certificates() -> bool:
    """True once the v2 kernel's ClearinghouseConfig accepts mode='certificates'."""
    from bilancio.config.models import ClearinghouseConfig

    annotation = ClearinghouseConfig.model_fields["mode"].annotation
    return "certificates" in get_args(annotation)


# ---------------------------------------------------------------------------
# certificate_totals helper
# ---------------------------------------------------------------------------


class TestCertificateTotals:
    def test_empty_events_returns_zeros(self):
        assert certificate_totals([]) == (Decimal("0"), Decimal("0"), Decimal("0"))

    def test_non_certificate_events_ignored(self):
        events = [
            {"kind": "PayableCreated", "amount": "100", "due_day": 1},
            {"kind": "PayableSettled", "day": 1, "amount": "100"},
        ]
        assert certificate_totals(events) == (Decimal("0"), Decimal("0"), Decimal("0"))

    def test_issued_total_sums_issuance_amounts(self):
        events = [
            {"kind": "CertificatesIssued", "day": 1, "member": "H1", "amount": "75"},
            {"kind": "CertificatesIssued", "day": 2, "member": "H2", "amount": "25"},
        ]
        issued, peak, losses = certificate_totals(events)
        assert issued == Decimal("100")
        assert peak == Decimal("100")
        assert losses == Decimal("0")

    def test_outstanding_peak_replays_day_by_day(self):
        # Day 1: +75; day 2: +50 (peak 125); day 3: -100 retired (outstanding 25)
        events = [
            {"kind": "CertificatesIssued", "day": 1, "member": "H1", "amount": "75"},
            {"kind": "CertificatesIssued", "day": 2, "member": "H2", "amount": "50"},
            {"kind": "CertificatesRetired", "day": 3, "member": "H1", "amount": "100"},
        ]
        issued, peak, losses = certificate_totals(events)
        assert issued == Decimal("125")
        assert peak == Decimal("125")
        assert losses == Decimal("0")

    def test_same_day_issue_and_retire_nets_within_day(self):
        # Redemption (retire) and new issuance occur in the same facility
        # phase, so peak is tracked at end-of-day granularity.
        events = [
            {"kind": "CertificatesIssued", "day": 1, "member": "H1", "amount": "60"},
            {"kind": "CertificatesRetired", "day": 2, "member": "H1", "amount": "60"},
            {"kind": "CertificatesIssued", "day": 2, "member": "H2", "amount": "40"},
        ]
        issued, peak, losses = certificate_totals(events)
        assert issued == Decimal("100")
        assert peak == Decimal("60")
        assert losses == Decimal("0")

    def test_haircut_counts_as_loss_and_reduces_outstanding(self):
        # Real kernel payload: `loss` is the total written down (margin
        # absorption + holder haircut + uncovered); `haircut_total` is the
        # holder-balance reduction that counts as retired face.
        events = [
            {"kind": "CertificatesIssued", "day": 1, "member": "H1", "amount": "100"},
            {"kind": "CertificatesRetired", "day": 2, "member": "H1", "amount": "70"},
            {
                "kind": "CertificateHaircutApplied",
                "day": 3,
                "loss": "30",
                "absorbed_by_margin": "5",
                "haircut_total": "25",
                "uncovered": "0",
            },
        ]
        issued, peak, losses = certificate_totals(events)
        assert issued == Decimal("100")
        assert peak == Decimal("100")
        assert losses == Decimal("30")

    def test_out_of_order_days_are_sorted(self):
        events = [
            {"kind": "CertificatesRetired", "day": 3, "amount": "50"},
            {"kind": "CertificatesIssued", "day": 1, "amount": "50"},
        ]
        issued, peak, losses = certificate_totals(events)
        assert issued == Decimal("50")
        assert peak == Decimal("50")
        assert losses == Decimal("0")

    def test_other_certificate_events_do_not_affect_totals(self):
        # Transfers/interest/recourse events move certificates between agents
        # or record charges; they do not change the outstanding total.
        events = [
            {"kind": "CertificatesIssued", "day": 1, "amount": "100"},
            {"kind": "CertificatesTransferred", "day": 1, "amount": "100"},
            {"kind": "CertificateInterestCharged", "day": 2, "amount": "3"},
            {"kind": "CertificateRecourseCreated", "day": 2, "amount": "40"},
            {"kind": "CertificateExcessReturned", "day": 2, "amount": "10"},
            {"kind": "CertificateRedemptionWindow", "day": 2, "amount": "20"},
            {"kind": "CertificatePledgeCreated", "day": 1, "amount": "133"},
        ]
        issued, peak, losses = certificate_totals(events)
        assert issued == Decimal("100")
        assert peak == Decimal("100")
        assert losses == Decimal("0")


class TestRunLevelMetricsWiring:
    def test_certificate_keys_present_and_zero_without_events(self):
        summary = compute_run_level_metrics([])
        assert summary["certificates_issued_total"] == 0.0
        assert summary["certificates_outstanding_peak"] == 0.0
        assert summary["cert_default_losses"] == 0.0

    def test_certificate_keys_are_floats(self):
        events = [
            {"kind": "CertificatesIssued", "day": 1, "member": "H1", "amount": "75"},
            {
                "kind": "CertificateHaircutApplied",
                "day": 2,
                "loss": "5",
                "absorbed_by_margin": "0",
                "haircut_total": "5",
                "uncovered": "0",
            },
        ]
        summary = compute_run_level_metrics(events)
        assert summary["certificates_issued_total"] == 75.0
        assert summary["certificates_outstanding_peak"] == 75.0
        assert summary["cert_default_losses"] == 5.0
        assert isinstance(summary["certificates_issued_total"], float)
        assert isinstance(summary["certificates_outstanding_peak"], float)
        assert isinstance(summary["cert_default_losses"], float)


# ---------------------------------------------------------------------------
# CLI wiring (--clearing-certificates)
# ---------------------------------------------------------------------------


class TestSweepRingCertificatesCLI:
    def test_sweep_ring_help_lists_certificate_options(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "ring", "--help"])
        assert result.exit_code == 0
        assert "--clearing-certificates" in result.output
        assert "--cert-haircut" in result.output
        assert "--cert-rate" in result.output
        assert "--cert-max-issuance" in result.output

    @patch("bilancio.ui.cli.sweep.aggregate_runs")
    @patch("bilancio.ui.cli.sweep.render_dashboard")
    @patch("bilancio.ui.cli.sweep.RingSweepRunner")
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-cert-job")
    def test_clearing_certificates_flag_passed_to_runner(
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
                "--clearing-certificates",
                "--cert-haircut",
                "0.3",
                "--cert-rate",
                "0.05",
                "--cert-max-issuance",
                "0.8",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        kwargs = mock_runner_cls.call_args[1]
        assert kwargs["clearing_certificates"] is True
        assert kwargs["cert_haircut"] == Decimal("0.3")
        assert kwargs["cert_rate"] == Decimal("0.05")
        assert kwargs["cert_max_issuance"] == Decimal("0.8")

    @patch("bilancio.ui.cli.sweep.aggregate_runs")
    @patch("bilancio.ui.cli.sweep.render_dashboard")
    @patch("bilancio.ui.cli.sweep.RingSweepRunner")
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-cert-defaults")
    def test_certificate_defaults_match_plan(
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
                "--clearing-certificates",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        kwargs = mock_runner_cls.call_args[1]
        assert kwargs["clearing_certificates"] is True
        assert kwargs["cert_haircut"] == Decimal("0.25")
        assert kwargs["cert_rate"] == Decimal("0.06")
        assert kwargs["cert_max_issuance"] == Decimal("1.0")


class TestRunnerScenarioBlock:
    """RingSweepRunner emits the certificates clearinghouse block."""

    def _make_runner(self, tmp_path, **kwargs):
        from bilancio.experiments.ring import RingSweepRunner

        return RingSweepRunner(
            out_dir=tmp_path,
            name_prefix="CertTest",
            n_agents=4,
            maturity_days=3,
            Q_total=Decimal("100"),
            liquidity_mode="uniform",
            liquidity_agent=None,
            base_seed=42,
            **kwargs,
        )

    def test_certificates_block_in_prepared_scenario(self, tmp_path):
        runner = self._make_runner(
            tmp_path,
            clearing_certificates=True,
            cert_haircut=Decimal("0.25"),
            cert_rate=Decimal("0.06"),
            cert_max_issuance=Decimal("1.0"),
        )
        prepared = runner._prepare_run(
            "grid", Decimal("0.5"), Decimal("1"), Decimal("0.75"), Decimal("0"), 42
        )
        block = prepared.scenario_config.get("clearinghouse")
        assert block == {
            "enabled": True,
            "mode": "certificates",
            "cert_haircut": 0.25,
            "cert_rate": 0.06,
            "max_issuance_per_member": 1.0,
        }

    def test_certificates_implies_clearing(self, tmp_path):
        runner = self._make_runner(tmp_path, clearing_certificates=True)
        assert runner.clearing_enabled is True
        assert runner.clearing_certificates is True

    def test_netting_block_unchanged_without_certificates(self, tmp_path):
        runner = self._make_runner(tmp_path, clearing_enabled=True)
        prepared = runner._prepare_run(
            "grid", Decimal("0.5"), Decimal("1"), Decimal("0"), Decimal("0"), 42
        )
        assert prepared.scenario_config.get("clearinghouse") == {
            "enabled": True,
            "mode": "netting",
        }

    def test_no_block_when_clearing_off(self, tmp_path):
        runner = self._make_runner(tmp_path)
        prepared = runner._prepare_run(
            "grid", Decimal("0.5"), Decimal("1"), Decimal("0"), Decimal("0"), 42
        )
        assert "clearinghouse" not in prepared.scenario_config


class TestCompilerPassthrough:
    """RingExplorerParamsModel.clearinghouse passthrough to compiled scenarios."""

    def _generator_config(self, clearinghouse=None):
        from bilancio.config.models import RingExplorerGeneratorConfig

        params = {
            "n_agents": 4,
            "seed": 42,
            "kappa": "0.5",
            "Q_total": "100",
            "maturity": {"days": 3, "mode": "lead_lag", "mu": "0.75"},
        }
        if clearinghouse is not None:
            params["clearinghouse"] = clearinghouse
        return RingExplorerGeneratorConfig.model_validate(
            {
                "version": 1,
                "generator": "ring_explorer_v1",
                "name_prefix": "CertCompiler",
                "params": params,
                "compile": {"emit_yaml": False},
            }
        )

    def test_clearinghouse_block_emitted(self):
        from bilancio.scenarios import compile_ring_explorer

        block = {
            "enabled": True,
            "mode": "certificates",
            "cert_haircut": 0.25,
            "cert_rate": 0.06,
            "max_issuance_per_member": 1.0,
        }
        scenario = compile_ring_explorer(self._generator_config(block))
        assert scenario["clearinghouse"] == block

    def test_no_block_by_default(self):
        from bilancio.scenarios import compile_ring_explorer

        scenario = compile_ring_explorer(self._generator_config())
        assert "clearinghouse" not in scenario


# ---------------------------------------------------------------------------
# Aggregate results columns
# ---------------------------------------------------------------------------


def test_aggregate_runs_carries_certificate_columns(tmp_path):
    registry_dir = tmp_path / "registry"
    out_dir = tmp_path / "runs" / "grid_cert1" / "out"
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
        "certificates_issued_total,certificates_outstanding_peak,cert_default_losses\n"
        "grid_cert1,grid,42,5,0.5,1,0.75,0,100,50,"
        "../runs/grid_cert1/out/metrics.csv,completed,2,0,,"
        "75.0,75.0,5.0\n"
    )

    results_csv = aggregate_dir / "results.csv"
    rows = aggregate_runs(registry_csv, results_csv)

    assert rows
    row = rows[0]
    assert row["certificates_issued_total"] == "75.0"
    assert row["certificates_outstanding_peak"] == "75.0"
    assert row["cert_default_losses"] == "5.0"

    header = results_csv.read_text().splitlines()[0]
    assert "certificates_issued_total" in header
    assert "certificates_outstanding_peak" in header
    assert "cert_default_losses" in header


def test_aggregate_runs_defaults_to_zero_for_legacy_registries(tmp_path):
    """Registries written before Plan 060 lack the certificate columns."""
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
    assert row["certificates_issued_total"] == 0.0
    assert row["certificates_outstanding_peak"] == 0.0
    assert row["cert_default_losses"] == 0.0


# ---------------------------------------------------------------------------
# Profile and instrument-kind surface
# ---------------------------------------------------------------------------


class TestClearinghouseProfile:
    def test_defaults_match_plan(self):
        from bilancio.decision.profiles import ClearinghouseProfile

        profile = ClearinghouseProfile()
        assert profile.cert_haircut == Decimal("0.25")
        assert profile.cert_rate == Decimal("0.06")
        assert profile.max_issuance_per_member == Decimal("1.0")
        assert profile.cert_max_tenor is None
        assert profile.mandatory_acceptance is True

    def test_invalid_haircut_rejected(self):
        from bilancio.decision.profiles import ClearinghouseProfile

        with pytest.raises(ValueError):
            ClearinghouseProfile(cert_haircut=Decimal("1.0"))

    def test_voluntary_acceptance_rejected_in_stage_2(self):
        from bilancio.decision.profiles import ClearinghouseProfile

        with pytest.raises(ValueError):
            ClearinghouseProfile(mandatory_acceptance=False)


def test_instrument_kind_member_exists():
    from bilancio.domain.instruments.base import InstrumentKind

    assert InstrumentKind.CLEARINGHOUSE_CERTIFICATE == "clearinghouse_certificate"
    assert str(InstrumentKind.CLEARINGHOUSE_CERTIFICATE) == "clearinghouse_certificate"


# ---------------------------------------------------------------------------
# Tiny end-to-end run (requires the kernel side of Plan 060)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _kernel_supports_certificates(),
    # KERNEL DEPENDENCY: the sibling Plan 060 kernel change widens
    # ClearinghouseConfig.mode to include "certificates". Until it lands,
    # scenario validation rejects the certificates block, so this end-to-end
    # test is skipped. The orchestrator should unskip after integration.
    reason="v2 kernel does not yet accept clearinghouse mode 'certificates'",
)
def test_end_to_end_tiny_certificates_run(tmp_path):
    """Smallest possible local run: certificates block flows through the
    runner, the run completes, and the registry carries the three metrics."""
    import csv

    from bilancio.experiments.ring import RingSweepRunner

    runner = RingSweepRunner(
        out_dir=tmp_path,
        name_prefix="CertE2E",
        n_agents=4,
        maturity_days=3,
        Q_total=Decimal("100"),
        liquidity_mode="uniform",
        liquidity_agent=None,
        base_seed=42,
        default_handling="expel-agent",
        clearing_certificates=True,
    )
    summaries = runner.run_grid(
        [Decimal("0.5")], [Decimal("1")], [Decimal("0.75")], [Decimal("0")]
    )
    assert len(summaries) == 1
    summary = summaries[0]
    # Metrics must be populated floats (possibly 0.0 depending on dynamics)
    assert summary.certificates_issued_total >= 0.0
    assert summary.certificates_outstanding_peak >= 0.0
    assert summary.cert_default_losses >= 0.0

    registry_csv = tmp_path / "registry" / "experiments.csv"
    with registry_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert rows[0]["status"] == "completed"
    for column in (
        "certificates_issued_total",
        "certificates_outstanding_peak",
        "cert_default_losses",
    ):
        assert column in rows[0]
        assert rows[0][column] != ""
