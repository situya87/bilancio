"""Tests for sweep_diagnostics — pre-flight and post-sweep validation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from bilancio.scenarios.sweep_diagnostics import (
    PostflightReport,
    PreflightReport,
    run_postsweep_validation,
    run_preflight_checks,
)

# ---------------------------------------------------------------------------
# Minimal config stubs matching the fields read by run_preflight_checks
# ---------------------------------------------------------------------------


@dataclass
class _BalancedConfigStub:
    """Minimal stand-in for BalancedComparisonConfig."""

    n_agents: int = 50
    maturity_days: int = 10
    face_value: Decimal = Decimal("20")
    Q_total: Decimal = Decimal("10000")
    kappas: list[Decimal] | None = None
    concentrations: list[Decimal] | None = None
    mus: list[Decimal] | None = None
    outside_mid_ratios: list[Decimal] | None = None
    dealer_share_per_bucket: Decimal = Decimal("0.125")
    vbt_share_per_bucket: Decimal = Decimal("0.25")
    risk_aversion: Decimal = Decimal("0")
    enable_lender: bool = False
    lender_base_rate: Decimal = Decimal("0.05")
    lender_risk_premium_scale: Decimal = Decimal("0.20")
    enable_bank_passive: bool = False
    enable_bank_dealer: bool = False
    enable_bank_dealer_nbfi: bool = False

    def __post_init__(self) -> None:
        if self.kappas is None:
            self.kappas = [Decimal("0.5"), Decimal("1"), Decimal("2")]
        if self.concentrations is None:
            self.concentrations = [Decimal("1")]
        if self.mus is None:
            self.mus = [Decimal("0")]
        if self.outside_mid_ratios is None:
            self.outside_mid_ratios = [Decimal("0.90")]


@dataclass
class _BankConfigStub:
    """Minimal stand-in for BankComparisonConfig."""

    n_agents: int = 50
    maturity_days: int = 10
    face_value: Decimal = Decimal("20")
    Q_total: Decimal = Decimal("10000")
    kappas: list[Decimal] | None = None
    concentrations: list[Decimal] | None = None
    mus: list[Decimal] | None = None
    outside_mid_ratios: list[Decimal] | None = None
    risk_aversion: Decimal = Decimal("0")
    n_banks: int = 5
    reserve_ratio: Decimal = Decimal("0.50")
    credit_risk_loading: Decimal = Decimal("0.5")
    cb_rate_escalation_slope: Decimal = Decimal("0.05")
    cb_max_outstanding_ratio: Decimal = Decimal("2.0")

    def __post_init__(self) -> None:
        if self.kappas is None:
            self.kappas = [Decimal("0.5"), Decimal("1"), Decimal("2")]
        if self.concentrations is None:
            self.concentrations = [Decimal("1")]
        if self.mus is None:
            self.mus = [Decimal("0")]
        if self.outside_mid_ratios is None:
            self.outside_mid_ratios = [Decimal("0.90")]


# ---------------------------------------------------------------------------
# Minimal result stubs for post-sweep validation
# ---------------------------------------------------------------------------


@dataclass
class _BalancedResultStub:
    """Minimal stand-in for BalancedComparisonResult."""

    kappa: Decimal = Decimal("1")
    delta_passive: Decimal | None = Decimal("0.15")
    delta_active: Decimal | None = Decimal("0.10")
    phi_passive: Decimal | None = Decimal("0.85")
    phi_active: Decimal | None = Decimal("0.90")
    total_trades: int | None = 25
    dealer_total_return: Decimal | None = Decimal("0.03")
    cascade_fraction_passive: Decimal | None = Decimal("0.1")
    cascade_fraction_active: Decimal | None = Decimal("0.05")
    delta_lender: Decimal | None = None
    total_loans: int | None = None
    delta_bank_passive: Decimal | None = None

    @property
    def trading_effect(self) -> Decimal | None:
        if self.delta_passive is None or self.delta_active is None:
            return None
        return self.delta_passive - self.delta_active

    @property
    def lending_effect(self) -> Decimal | None:
        if self.delta_passive is None or self.delta_lender is None:
            return None
        return self.delta_passive - self.delta_lender

    @property
    def bank_passive_effect(self) -> Decimal | None:
        if self.delta_passive is None or self.delta_bank_passive is None:
            return None
        return self.delta_passive - self.delta_bank_passive


@dataclass
class _BankResultStub:
    """Minimal stand-in for BankComparisonResult."""

    kappa: Decimal = Decimal("1")
    delta_idle: Decimal | None = Decimal("0.20")
    delta_lend: Decimal | None = Decimal("0.12")
    phi_idle: Decimal | None = Decimal("0.80")
    phi_lend: Decimal | None = Decimal("0.88")
    cascade_fraction_idle: Decimal | None = Decimal("0.15")
    cascade_fraction_lend: Decimal | None = Decimal("0.08")
    cb_loans_created_idle: int = 3
    cb_loans_created_lend: int = 5

    @property
    def bank_lending_effect(self) -> Decimal | None:
        if self.delta_idle is None or self.delta_lend is None:
            return None
        return self.delta_idle - self.delta_lend


# ---------------------------------------------------------------------------
# Tests: run_preflight_checks
# ---------------------------------------------------------------------------


class TestPreflightBalanced:
    """Pre-flight checks for balanced comparison configs."""

    def test_basic_preflight_returns_all_checks(self) -> None:
        config = _BalancedConfigStub()
        report = run_preflight_checks(config)
        assert isinstance(report, PreflightReport)
        check_ids = [c.check_id for c in report.checks]
        for vid in ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11"):
            assert vid in check_ids, f"{vid} missing from checks"

    def test_all_pass_with_reasonable_params(self) -> None:
        config = _BalancedConfigStub(
            n_agents=100,
            maturity_days=10,
            kappas=[Decimal("0.5"), Decimal("1"), Decimal("2")],
        )
        report = run_preflight_checks(config)
        for c in report.checks:
            assert c.level in ("pass", "warn"), f"{c.check_id} failed: {c.detail}"

    def test_v7_warns_with_single_kappa(self) -> None:
        config = _BalancedConfigStub(kappas=[Decimal("1")])
        report = run_preflight_checks(config)
        v7 = next(c for c in report.checks if c.check_id == "V7")
        assert v7.level == "warn"  # <2 distinct kappas

    def test_v3_warns_all_high_kappas(self) -> None:
        config = _BalancedConfigStub(kappas=[Decimal("3"), Decimal("5")])
        report = run_preflight_checks(config)
        v3 = next(c for c in report.checks if c.check_id == "V3")
        assert v3.level == "warn"  # all kappas > 2

    def test_trade_reports_populated(self) -> None:
        config = _BalancedConfigStub(
            kappas=[Decimal("0.5"), Decimal("1"), Decimal("4")],
        )
        report = run_preflight_checks(config)
        assert len(report.kappa_trade_reports) >= 2

    def test_lender_v5_shows_rates(self) -> None:
        config = _BalancedConfigStub(enable_lender=True)
        report = run_preflight_checks(config)
        v5 = next(c for c in report.checks if c.check_id == "V5")
        assert "rate=" in v5.detail

    def test_v8_na_without_banking(self) -> None:
        config = _BalancedConfigStub()
        report = run_preflight_checks(config)
        v8 = next(c for c in report.checks if c.check_id == "V8")
        assert "n/a" in v8.detail

    def test_v8_runs_with_banking(self) -> None:
        config = _BalancedConfigStub(enable_bank_passive=True)
        report = run_preflight_checks(config)
        v8 = next(c for c in report.checks if c.check_id == "V8")
        assert "n/a" not in v8.detail

    def test_v9_vbt_mid_stability(self) -> None:
        config = _BalancedConfigStub()
        report = run_preflight_checks(config)
        v9 = next(c for c in report.checks if c.check_id == "V9")
        assert v9.level == "pass"  # M should be well above M_MIN with omr=0.90
        assert "M=" in v9.detail
        assert "M_MIN=" in v9.detail

    def test_v10_na_without_lender(self) -> None:
        config = _BalancedConfigStub()
        report = run_preflight_checks(config)
        v10 = next(c for c in report.checks if c.check_id == "V10")
        assert "n/a" in v10.detail

    def test_v10_runs_with_lender(self) -> None:
        config = _BalancedConfigStub(enable_lender=True)
        report = run_preflight_checks(config)
        v10 = next(c for c in report.checks if c.check_id == "V10")
        assert "n/a" not in v10.detail
        assert "loan=" in v10.detail

    def test_v11_na_without_banking(self) -> None:
        config = _BalancedConfigStub()
        report = run_preflight_checks(config)
        v11 = next(c for c in report.checks if c.check_id == "V11")
        assert "n/a" in v11.detail

    def test_v11_runs_with_banking(self) -> None:
        config = _BalancedConfigStub(enable_bank_passive=True)
        report = run_preflight_checks(config)
        v11 = next(c for c in report.checks if c.check_id == "V11")
        assert "n/a" not in v11.detail
        assert "R=" in v11.detail

    def test_print_summary_no_error(self, capsys) -> None:
        config = _BalancedConfigStub()
        report = run_preflight_checks(config)
        report.print_summary()
        out = capsys.readouterr().out
        assert "PRE-FLIGHT" in out

    def test_to_dict(self) -> None:
        config = _BalancedConfigStub()
        report = run_preflight_checks(config)
        d = report.to_dict()
        assert "checks" in d
        assert len(d["checks"]) == len(report.checks)


class TestPreflightBank:
    """Pre-flight checks for bank comparison configs."""

    def test_bank_config_all_checks(self) -> None:
        config = _BankConfigStub()
        report = run_preflight_checks(config)
        check_ids = [c.check_id for c in report.checks]
        for vid in ("V1", "V8", "V9", "V11"):
            assert vid in check_ids

    def test_v8_runs_for_bank(self) -> None:
        config = _BankConfigStub()
        report = run_preflight_checks(config)
        v8 = next(c for c in report.checks if c.check_id == "V8")
        assert "n/a" not in v8.detail

    def test_interbank_reports_populated(self) -> None:
        config = _BankConfigStub()
        report = run_preflight_checks(config)
        assert len(report.interbank_reports) > 0

    def test_v11_bank_reserves(self) -> None:
        config = _BankConfigStub()
        report = run_preflight_checks(config)
        v11 = next(c for c in report.checks if c.check_id == "V11")
        assert "n/a" not in v11.detail
        assert "R=" in v11.detail


# ---------------------------------------------------------------------------
# Tests: run_postsweep_validation
# ---------------------------------------------------------------------------


class TestPostsweepBalanced:
    """Post-sweep validation with balanced results."""

    def test_basic_validation(self) -> None:
        config = _BalancedConfigStub()
        preflight = run_preflight_checks(config)
        results = [
            _BalancedResultStub(kappa=Decimal("0.5"), total_trades=30),
            _BalancedResultStub(kappa=Decimal("1"), total_trades=15),
            _BalancedResultStub(kappa=Decimal("2"), total_trades=5),
        ]
        post = run_postsweep_validation(preflight, results)
        assert isinstance(post, PostflightReport)
        assert len(post.checks) > 0

    def test_all_match_with_trades(self) -> None:
        config = _BalancedConfigStub()
        preflight = run_preflight_checks(config)
        results = [
            _BalancedResultStub(
                kappa=Decimal("0.5"),
                delta_passive=Decimal("0.3"),
                delta_active=Decimal("0.2"),
                phi_passive=Decimal("0.7"),
                phi_active=Decimal("0.8"),
                total_trades=50,
            ),
            _BalancedResultStub(
                kappa=Decimal("1"),
                delta_passive=Decimal("0.15"),
                delta_active=Decimal("0.10"),
                phi_passive=Decimal("0.85"),
                phi_active=Decimal("0.90"),
                total_trades=25,
            ),
        ]
        post = run_postsweep_validation(preflight, results)
        assert post.match_rate > 0.5

    def test_empty_results_returns_empty(self) -> None:
        config = _BalancedConfigStub()
        preflight = run_preflight_checks(config)
        post = run_postsweep_validation(preflight, [])
        assert len(post.checks) == 0
        assert post.match_rate == 1.0

    def test_v1_settlement_with_phi(self) -> None:
        """V1 checks settlement occurred via phi values."""
        config = _BalancedConfigStub()
        preflight = run_preflight_checks(config)
        results = [_BalancedResultStub(phi_passive=Decimal("0.9"))]
        post = run_postsweep_validation(preflight, results)
        v1 = next((c for c in post.checks if c.check_id == "V1"), None)
        assert v1 is not None
        assert v1.actual  # phi > 0 → settlement occurred
        assert "phi>0" in v1.detail

    def test_v1_mismatch_when_no_settlement(self) -> None:
        """V1 mismatches when phi is zero everywhere."""
        config = _BalancedConfigStub()
        preflight = run_preflight_checks(config)
        results = [
            _BalancedResultStub(
                phi_passive=Decimal("0"),
                phi_active=Decimal("0"),
            ),
        ]
        post = run_postsweep_validation(preflight, results)
        v1 = next((c for c in post.checks if c.check_id == "V1"), None)
        assert v1 is not None
        assert not v1.actual

    def test_v2_trading_occurred(self) -> None:
        """V2 checks that total_trades > 0."""
        config = _BalancedConfigStub()
        preflight = run_preflight_checks(config)
        results = [_BalancedResultStub(total_trades=30)]
        post = run_postsweep_validation(preflight, results)
        v2 = next((c for c in post.checks if c.check_id == "V2"), None)
        assert v2 is not None
        assert v2.actual
        assert "total_trades=30" in v2.detail

    def test_v2_no_trading(self) -> None:
        """V2 reports no trading when total_trades=0."""
        config = _BalancedConfigStub()
        preflight = run_preflight_checks(config)
        results = [_BalancedResultStub(total_trades=0)]
        post = run_postsweep_validation(preflight, results)
        v2 = next((c for c in post.checks if c.check_id == "V2"), None)
        assert v2 is not None
        assert not v2.actual

    def test_v4_dealer_active(self) -> None:
        """V4 checks dealer metrics exist."""
        config = _BalancedConfigStub()
        preflight = run_preflight_checks(config)
        results = [
            _BalancedResultStub(dealer_total_return=Decimal("0.05"), total_trades=10),
        ]
        post = run_postsweep_validation(preflight, results)
        v4 = next((c for c in post.checks if c.check_id == "V4"), None)
        assert v4 is not None
        assert v4.actual
        assert "dealer_return" in v4.detail

    def test_v6_cascade_dynamics(self) -> None:
        """V6 checks cascade_fraction > 0."""
        config = _BalancedConfigStub()
        preflight = run_preflight_checks(config)
        results = [
            _BalancedResultStub(
                cascade_fraction_passive=Decimal("0.2"),
                cascade_fraction_active=Decimal("0.1"),
            ),
        ]
        post = run_postsweep_validation(preflight, results)
        v6 = next((c for c in post.checks if c.check_id == "V6"), None)
        assert v6 is not None
        assert v6.actual
        assert "cascade>0" in v6.detail

    def test_v6_no_cascades(self) -> None:
        """V6 mismatch when no cascades occurred."""
        config = _BalancedConfigStub()
        preflight = run_preflight_checks(config)
        results = [
            _BalancedResultStub(
                cascade_fraction_passive=Decimal("0"),
                cascade_fraction_active=Decimal("0"),
            ),
        ]
        post = run_postsweep_validation(preflight, results)
        v6 = next((c for c in post.checks if c.check_id == "V6"), None)
        assert v6 is not None
        assert not v6.actual

    def test_write_report(self, tmp_path: Path) -> None:
        config = _BalancedConfigStub()
        preflight = run_preflight_checks(config)
        results = [_BalancedResultStub()]
        run_postsweep_validation(
            preflight, results, aggregate_dir=tmp_path
        )
        report_path = tmp_path / "viability_report.txt"
        assert report_path.exists()
        content = report_path.read_text()
        assert "POST-SWEEP VALIDATION" in content

    def test_print_summary_no_error(self, capsys) -> None:
        config = _BalancedConfigStub()
        preflight = run_preflight_checks(config)
        results = [_BalancedResultStub()]
        post = run_postsweep_validation(preflight, results)
        post.print_summary()
        out = capsys.readouterr().out
        assert "POST-SWEEP" in out


class TestPostsweepBank:
    """Post-sweep validation with bank results."""

    def test_bank_validation(self) -> None:
        config = _BankConfigStub()
        preflight = run_preflight_checks(config)
        results = [
            _BankResultStub(kappa=Decimal("0.5")),
            _BankResultStub(kappa=Decimal("1")),
        ]
        post = run_postsweep_validation(preflight, results)
        assert isinstance(post, PostflightReport)
        assert len(post.checks) > 0

    def test_bank_v1_settlement(self) -> None:
        """V1 checks phi_idle/phi_lend for bank results."""
        config = _BankConfigStub()
        preflight = run_preflight_checks(config)
        results = [
            _BankResultStub(phi_idle=Decimal("0.8"), phi_lend=Decimal("0.9")),
        ]
        post = run_postsweep_validation(preflight, results)
        v1 = next((c for c in post.checks if c.check_id == "V1"), None)
        assert v1 is not None
        assert v1.actual

    def test_bank_v8_activity(self) -> None:
        """V8 checks both idle and lend arms completed."""
        config = _BankConfigStub()
        preflight = run_preflight_checks(config)
        results = [
            _BankResultStub(
                delta_idle=Decimal("0.30"),
                delta_lend=Decimal("0.20"),
            ),
        ]
        post = run_postsweep_validation(preflight, results)
        v8 = next((c for c in post.checks if c.check_id == "V8"), None)
        assert v8 is not None
        assert v8.actual  # both arms completed

    def test_bank_v11_cb_backstop(self) -> None:
        """V11 checks CB loans were created in banking runs."""
        config = _BankConfigStub()
        preflight = run_preflight_checks(config)
        results = [
            _BankResultStub(cb_loans_created_idle=3, cb_loans_created_lend=5),
        ]
        post = run_postsweep_validation(preflight, results)
        v11 = next((c for c in post.checks if c.check_id == "V11"), None)
        assert v11 is not None
        assert v11.actual
        assert "CB loans" in v11.detail

    def test_bank_v6_cascade(self) -> None:
        """V6 checks cascade_fraction in bank results."""
        config = _BankConfigStub()
        preflight = run_preflight_checks(config)
        results = [
            _BankResultStub(
                cascade_fraction_idle=Decimal("0.2"),
                cascade_fraction_lend=Decimal("0.1"),
            ),
        ]
        post = run_postsweep_validation(preflight, results)
        v6 = next((c for c in post.checks if c.check_id == "V6"), None)
        assert v6 is not None
        assert v6.actual

    def test_write_report(self, tmp_path: Path) -> None:
        config = _BankConfigStub()
        preflight = run_preflight_checks(config)
        results = [_BankResultStub()]
        run_postsweep_validation(
            preflight, results, aggregate_dir=tmp_path
        )
        assert (tmp_path / "viability_report.txt").exists()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for preflight/postflight."""

    def test_single_kappa(self) -> None:
        config = _BalancedConfigStub(kappas=[Decimal("1")])
        report = run_preflight_checks(config)
        # V1-V11 = 11 checks total
        assert len(report.checks) == 11

    def test_two_kappas(self) -> None:
        config = _BalancedConfigStub(kappas=[Decimal("0.5"), Decimal("2")])
        report = run_preflight_checks(config)
        assert len(report.kappa_trade_reports) == 2

    def test_many_kappas_picks_three_representatives(self) -> None:
        config = _BalancedConfigStub(
            kappas=[Decimal(str(x)) for x in [0.1, 0.3, 0.5, 1, 2, 4, 8]],
        )
        report = run_preflight_checks(config)
        assert len(report.kappa_trade_reports) == 3

    def test_postflight_match_rate_all_match(self) -> None:
        post = PostflightReport(checks=[])
        assert post.match_rate == 1.0

    def test_small_n_agents_warns(self) -> None:
        config = _BalancedConfigStub(n_agents=5, maturity_days=3)
        report = run_preflight_checks(config)
        v7 = next(c for c in report.checks if c.check_id == "V7")
        assert v7.level == "warn"

    def test_v9_na_checks_all_configs(self) -> None:
        """V9 should always run (VBT is always present)."""
        config = _BalancedConfigStub()
        report = run_preflight_checks(config)
        v9 = next(c for c in report.checks if c.check_id == "V9")
        assert v9.check_id == "V9"
        # With omr=0.90, M should be well above M_MIN
        assert v9.level == "pass"

    def test_postflight_no_v9_v10_validators(self) -> None:
        """V9/V10 are pre-flight only — no post-sweep validator."""
        config = _BalancedConfigStub()
        preflight = run_preflight_checks(config)
        results = [_BalancedResultStub()]
        post = run_postsweep_validation(preflight, results)
        post_ids = {c.check_id for c in post.checks}
        # V9 and V10 have no post-sweep validator
        assert "V9" not in post_ids
        assert "V10" not in post_ids
