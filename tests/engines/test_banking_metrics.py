"""Tests for banking-specific default metrics (δ_bank + deposit_loss).

Plan 039: These metrics are only meaningful for banking arms.
"""

from __future__ import annotations

from bilancio.analysis.report import compute_run_level_metrics


def _evt(kind: str, **kwargs) -> dict:
    """Helper to create a minimal event dict."""
    return {"kind": kind, **kwargs}


class TestDeltaBank:
    """Tests for δ_bank = bank_writeoffs / bank_obligations_created."""

    def test_none_when_no_banking(self):
        """No banking events → delta_bank is None."""
        events = [
            _evt("PayableSettled", amount=1000),
            _evt("PayableSettled", amount=500),
        ]
        m = compute_run_level_metrics(events)
        assert m["delta_bank"] is None
        assert m["bank_obligations_created"] == 0
        assert m["bank_writeoffs"] == 0

    def test_zero_when_all_repaid(self):
        """CB loans created but none written off → delta_bank = 0."""
        events = [
            _evt("CBLoanCreated", amount=1000),
            _evt("CBLoanCreated", amount=500),
            _evt("CBLoanRepaid", amount=1000),
            _evt("CBLoanRepaid", amount=500),
        ]
        m = compute_run_level_metrics(events)
        assert m["delta_bank"] == 0.0
        assert m["bank_obligations_created"] == 1500
        assert m["bank_writeoffs"] == 0

    def test_partial_writeoff(self):
        """Some CB loans written off in final settlement."""
        events = [
            _evt("CBLoanCreated", amount=1000),
            _evt("CBLoanCreated", amount=500),
            _evt("CBFinalSettlementWrittenOff", amount=500),
        ]
        m = compute_run_level_metrics(events)
        assert m["delta_bank"] == 500 / 1500
        assert m["bank_obligations_created"] == 1500
        assert m["bank_writeoffs"] == 500

    def test_includes_freeze_defaults(self):
        """BankDefaultCBFreeze events count as bank writeoffs."""
        events = [
            _evt("CBLoanCreated", amount=1000),
            _evt("BankDefaultCBFreeze", amount=1000),
        ]
        m = compute_run_level_metrics(events)
        assert m["delta_bank"] == 1.0
        assert m["bank_writeoffs"] == 1000

    def test_includes_interbank_writeoffs(self):
        """ObligationWrittenOff with interbank contract_kind contributes to delta_bank."""
        events = [
            _evt("InterbankLoan", amount=200),
            _evt("ObligationWrittenOff", contract_kind="interbank_loan", amount=200),
        ]
        m = compute_run_level_metrics(events)
        assert m["delta_bank"] == 1.0
        assert m["bank_obligations_created"] == 200
        assert m["bank_writeoffs"] == 200

    def test_includes_interbank_overnight(self):
        """InterbankOvernightCreated events count in denominator, interbank_overnight writeoffs in numerator."""
        events = [
            _evt("InterbankOvernightCreated", amount=300),
            _evt("ObligationWrittenOff", contract_kind="interbank_overnight", amount=100),
        ]
        m = compute_run_level_metrics(events)
        assert m["delta_bank"] == 100 / 300
        assert m["bank_obligations_created"] == 300
        assert m["bank_writeoffs"] == 100

    def test_combined_sources(self):
        """Multiple event types combine correctly."""
        events = [
            _evt("CBLoanCreated", amount=1000),
            _evt("InterbankLoan", amount=200),
            _evt("CBFinalSettlementWrittenOff", amount=300),
            _evt("BankDefaultCBFreeze", amount=200),
            _evt("ObligationWrittenOff", contract_kind="interbank_loan", amount=100),
        ]
        m = compute_run_level_metrics(events)
        assert m["bank_obligations_created"] == 1200  # 1000 + 200
        assert m["bank_writeoffs"] == 600  # 300 + 200 + 100
        assert m["delta_bank"] == 600 / 1200


class TestDepositLoss:
    """Tests for deposit_loss metrics."""

    def test_zero_no_bank_failure(self):
        """No bank failures → deposit_loss_gross = 0."""
        events = [
            _evt("CashDeposited", amount=10000),
            _evt("PayableSettled", amount=5000),
        ]
        m = compute_run_level_metrics(events)
        assert m["deposit_loss_gross"] == 0
        assert m["deposit_loss_pct"] == 0.0

    def test_from_bank_expulsion(self):
        """ObligationWrittenOff with bank_deposit captures depositor losses."""
        events = [
            _evt("CashDeposited", amount=10000),
            _evt("ObligationWrittenOff", contract_kind="bank_deposit", amount=5000),
        ]
        m = compute_run_level_metrics(events)
        assert m["deposit_loss_gross"] == 5000
        assert m["deposit_loss_pct"] == 0.5
        assert m["total_deposits_created"] == 10000

    def test_pct_none_when_no_deposits(self):
        """No CashDeposited events → deposit_loss_pct is None."""
        events = [
            _evt("PayableSettled", amount=1000),
        ]
        m = compute_run_level_metrics(events)
        assert m["deposit_loss_pct"] is None
        assert m["total_deposits_created"] == 0

    def test_multiple_bank_failures(self):
        """Multiple bank failures accumulate deposit losses."""
        events = [
            _evt("CashDeposited", amount=5000),
            _evt("CashDeposited", amount=5000),
            _evt("ObligationWrittenOff", contract_kind="bank_deposit", amount=3000),
            _evt("ObligationWrittenOff", contract_kind="bank_deposit", amount=2000),
        ]
        m = compute_run_level_metrics(events)
        assert m["deposit_loss_gross"] == 5000
        assert m["deposit_loss_pct"] == 0.5

    def test_non_deposit_writeoff_ignored(self):
        """ObligationWrittenOff with non-deposit contract_kind doesn't count."""
        events = [
            _evt("CashDeposited", amount=10000),
            _evt("ObligationWrittenOff", contract_kind="payable", amount=5000),
            _evt("ObligationWrittenOff", contract_kind="cb_loan", amount=3000),
        ]
        m = compute_run_level_metrics(events)
        assert m["deposit_loss_gross"] == 0
        assert m["deposit_loss_pct"] == 0.0
