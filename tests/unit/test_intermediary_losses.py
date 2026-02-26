"""Unit tests for intermediary loss metric functionality.

Tests cover:
- compute_intermediary_losses() from bilancio.analysis.report
- Integration with compute_run_level_metrics()
- Adjusted treatment effect properties on BalancedComparisonResult
"""

from decimal import Decimal

import pytest

from bilancio.analysis.report import compute_intermediary_losses, compute_run_level_metrics
from bilancio.experiments.balanced_comparison import BalancedComparisonResult


# ---------------------------------------------------------------------------
# Helper: minimal kwargs for BalancedComparisonResult
# ---------------------------------------------------------------------------


def _base_kwargs(**overrides):
    """Return minimal kwargs for constructing a BalancedComparisonResult."""
    defaults = {
        "kappa": Decimal("1"),
        "concentration": Decimal("1"),
        "mu": Decimal("0"),
        "monotonicity": Decimal("0"),
        "seed": 42,
        "face_value": Decimal("20"),
        "outside_mid_ratio": Decimal("0.9"),
        "big_entity_share": Decimal("0"),
        "delta_passive": Decimal("0.3"),
        "phi_passive": Decimal("0.7"),
        "passive_run_id": "test_passive",
        "passive_status": "completed",
        "delta_active": Decimal("0.2"),
        "phi_active": Decimal("0.8"),
        "active_run_id": "test_active",
        "active_status": "completed",
    }
    defaults.update(overrides)
    return defaults


# ===========================================================================
# Tests: compute_intermediary_losses()
# ===========================================================================


class TestComputeIntermediaryLosses:
    """Tests for the compute_intermediary_losses function."""

    def test_no_events_no_dealer_metrics(self):
        """All zeros when no events and no dealer_metrics are provided."""
        result = compute_intermediary_losses(events=[], dealer_metrics=None)
        assert result["dealer_vbt_loss"] == 0.0
        assert result["nbfi_loan_loss"] == 0.0
        assert result["bank_credit_loss"] == 0.0
        assert result["cb_backstop_loss"] == 0.0
        assert result["intermediary_loss_total"] == 0.0

    def test_dealer_metrics_negative_pnl(self):
        """Negative dealer PnL translates to dealer_vbt_loss."""
        result = compute_intermediary_losses(
            events=[],
            dealer_metrics={"dealer_total_pnl": -150.0},
        )
        assert result["dealer_vbt_loss"] == pytest.approx(150.0)
        assert result["intermediary_loss_total"] == pytest.approx(150.0)

    def test_dealer_metrics_positive_pnl(self):
        """Positive dealer PnL means zero dealer_vbt_loss (no loss when profitable)."""
        result = compute_intermediary_losses(
            events=[],
            dealer_metrics={"dealer_total_pnl": 200.0},
        )
        assert result["dealer_vbt_loss"] == 0.0
        assert result["intermediary_loss_total"] == 0.0

    def test_dealer_metrics_zero_pnl(self):
        """Zero PnL means zero dealer loss."""
        result = compute_intermediary_losses(
            events=[],
            dealer_metrics={"dealer_total_pnl": 0.0},
        )
        assert result["dealer_vbt_loss"] == 0.0

    def test_nbfi_loan_defaulted_events(self):
        """NonBankLoanDefaulted events compute nbfi_loan_loss correctly."""
        events = [
            {
                "kind": "NonBankLoanDefaulted",
                "day": 3,
                "loan_id": "NBL_001",
                "borrower_id": "H1",
                "lender_id": "lender",
                "amount_owed": 500,
                "cash_available": 300,
            },
            {
                "kind": "NonBankLoanDefaulted",
                "day": 4,
                "loan_id": "NBL_002",
                "borrower_id": "H2",
                "lender_id": "lender",
                "amount_owed": 200,
                "cash_available": 50,
            },
        ]
        result = compute_intermediary_losses(events=events)
        # Loss = (500-300) + (200-50) = 200 + 150 = 350
        assert result["nbfi_loan_loss"] == pytest.approx(350.0)
        assert result["intermediary_loss_total"] == pytest.approx(350.0)

    def test_bank_loan_default_events(self):
        """BankLoanDefault events compute bank_credit_loss correctly."""
        events = [
            {
                "kind": "BankLoanDefault",
                "day": 5,
                "bank": "bank_1",
                "borrower": "H2",
                "principal": 1000,
                "repayment_due": 800,
                "recovered": 500,
                "loan_id": "BL_001",
            },
        ]
        result = compute_intermediary_losses(events=events)
        # Loss = 800 - 500 = 300
        assert result["bank_credit_loss"] == pytest.approx(300.0)
        assert result["intermediary_loss_total"] == pytest.approx(300.0)

    def test_bank_loan_default_partial_recovery(self):
        """Partial recovery: loss = repayment_due - recovered."""
        events = [
            {
                "kind": "BankLoanDefault",
                "day": 5,
                "bank": "bank_1",
                "borrower": "H1",
                "principal": 1000,
                "repayment_due": 1050,
                "recovered": 900,
                "loan_id": "BL_002",
            },
        ]
        result = compute_intermediary_losses(events=events)
        assert result["bank_credit_loss"] == pytest.approx(150.0)

    def test_cb_loan_freeze_written_off_events(self):
        """CBLoanFreezeWrittenOff events set cb_backstop_loss."""
        events = [
            {"kind": "CBLoanFreezeWrittenOff", "day": 10, "amount": 200, "bank": "bank_1"},
            {"kind": "CBLoanFreezeWrittenOff", "day": 11, "amount": 100, "bank": "bank_2"},
        ]
        result = compute_intermediary_losses(events=events)
        assert result["cb_backstop_loss"] == pytest.approx(300.0)
        assert result["intermediary_loss_total"] == pytest.approx(300.0)

    def test_all_event_types_combined(self):
        """All loss channels sum to intermediary_loss_total."""
        events = [
            {
                "kind": "NonBankLoanDefaulted",
                "day": 3,
                "amount_owed": 500,
                "cash_available": 300,
            },
            {
                "kind": "BankLoanDefault",
                "day": 5,
                "repayment_due": 800,
                "recovered": 500,
            },
            {"kind": "CBLoanFreezeWrittenOff", "day": 10, "amount": 200},
        ]
        dealer_metrics = {"dealer_total_pnl": -100.0}
        result = compute_intermediary_losses(events=events, dealer_metrics=dealer_metrics)

        assert result["dealer_vbt_loss"] == pytest.approx(100.0)
        assert result["nbfi_loan_loss"] == pytest.approx(200.0)  # 500 - 300
        assert result["bank_credit_loss"] == pytest.approx(300.0)  # 800 - 500
        assert result["cb_backstop_loss"] == pytest.approx(200.0)
        expected_total = 100.0 + 200.0 + 300.0 + 200.0
        assert result["intermediary_loss_total"] == pytest.approx(expected_total)

    def test_nbfi_full_recovery_no_loss(self):
        """When cash_available >= amount_owed, NBFI loss is zero."""
        events = [
            {
                "kind": "NonBankLoanDefaulted",
                "day": 3,
                "amount_owed": 100,
                "cash_available": 100,
            },
        ]
        result = compute_intermediary_losses(events=events)
        assert result["nbfi_loan_loss"] == pytest.approx(0.0)


# ===========================================================================
# Tests: compute_run_level_metrics() integration
# ===========================================================================


class TestComputeRunLevelMetricsIntermediaryFields:
    """Verify compute_run_level_metrics includes intermediary loss keys."""

    def test_return_dict_has_intermediary_keys(self):
        """Return dict includes nbfi_loan_loss, bank_credit_loss, cb_backstop_loss."""
        result = compute_run_level_metrics(events=[])
        assert "nbfi_loan_loss" in result
        assert "bank_credit_loss" in result
        assert "cb_backstop_loss" in result

    def test_nbfi_loan_loss_from_events(self):
        """NonBankLoanDefaulted events are picked up by compute_run_level_metrics."""
        events = [
            {
                "kind": "NonBankLoanDefaulted",
                "day": 3,
                "amount_owed": 500,
                "cash_available": 300,
            },
        ]
        result = compute_run_level_metrics(events=events)
        assert result["nbfi_loan_loss"] == 200  # 500 - 300

    def test_bank_credit_loss_from_events(self):
        """BankLoanDefault events are picked up by compute_run_level_metrics."""
        events = [
            {
                "kind": "BankLoanDefault",
                "day": 5,
                "repayment_due": 800,
                "recovered": 500,
            },
        ]
        result = compute_run_level_metrics(events=events)
        assert result["bank_credit_loss"] == 300  # 800 - 500


# ===========================================================================
# Tests: Adjusted treatment effect properties
# ===========================================================================


class TestAdjustedTreatmentEffects:
    """Tests for adjusted_trading_effect and adjusted_lending_effect properties."""

    def test_adjusted_trading_effect(self):
        """adjusted_trading_effect = trading_effect - (loss_pct_active - loss_pct_passive)."""
        r = BalancedComparisonResult(
            **_base_kwargs(
                intermediary_loss_passive=100.0,
                intermediary_loss_active=300.0,
                intermediary_loss_pct_passive=0.01,
                intermediary_loss_pct_active=0.03,
            )
        )
        # trading_effect = 0.3 - 0.2 = 0.1
        assert r.trading_effect == Decimal("0.1")
        # adjusted = 0.1 - (0.03 - 0.01) = 0.1 - 0.02 = 0.08
        assert r.adjusted_trading_effect == pytest.approx(0.08)

    def test_adjusted_trading_effect_none_when_loss_pct_missing(self):
        """adjusted_trading_effect returns None when intermediary_loss_pct is None."""
        r = BalancedComparisonResult(
            **_base_kwargs(
                intermediary_loss_pct_passive=None,
                intermediary_loss_pct_active=None,
            )
        )
        assert r.trading_effect == Decimal("0.1")
        assert r.adjusted_trading_effect is None

    def test_adjusted_trading_effect_none_when_trading_effect_none(self):
        """adjusted_trading_effect returns None when delta values are missing."""
        r = BalancedComparisonResult(
            **_base_kwargs(
                delta_passive=None,
                delta_active=None,
                intermediary_loss_pct_passive=0.01,
                intermediary_loss_pct_active=0.03,
            )
        )
        assert r.trading_effect is None
        assert r.adjusted_trading_effect is None

    def test_adjusted_lending_effect(self):
        """adjusted_lending_effect = lending_effect - (loss_pct_lender - loss_pct_passive)."""
        r = BalancedComparisonResult(
            **_base_kwargs(
                delta_lender=Decimal("0.15"),
                phi_lender=Decimal("0.85"),
                lender_run_id="lender-1",
                lender_status="completed",
                intermediary_loss_pct_passive=0.01,
                intermediary_loss_pct_lender=0.05,
            )
        )
        # lending_effect = 0.3 - 0.15 = 0.15
        assert r.lending_effect == Decimal("0.15")
        # adjusted = 0.15 - (0.05 - 0.01) = 0.15 - 0.04 = 0.11
        assert r.adjusted_lending_effect == pytest.approx(0.11)

    def test_adjusted_lending_effect_none_when_loss_pct_missing(self):
        """adjusted_lending_effect returns None when intermediary_loss_pct is None."""
        r = BalancedComparisonResult(
            **_base_kwargs(
                delta_lender=Decimal("0.15"),
                phi_lender=Decimal("0.85"),
                lender_run_id="lender-1",
                lender_status="completed",
                intermediary_loss_pct_passive=None,
                intermediary_loss_pct_lender=None,
            )
        )
        assert r.lending_effect == Decimal("0.15")
        assert r.adjusted_lending_effect is None
