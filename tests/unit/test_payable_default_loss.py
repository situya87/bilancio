"""Tests for payable_default_loss metric in compute_run_level_metrics.

Verifies that trigger contract shortfalls (ObligationDefaulted) and
remaining payable write-offs (ObligationWrittenOff) are both counted
in payable_default_loss and flow through to total_loss.
"""

from bilancio.analysis.report import compute_run_level_metrics


def _obligation_defaulted(shortfall, contract_id="PAY_001", debtor="H1", creditor="H2"):
    """Create an ObligationDefaulted event for a payable."""
    return {
        "kind": "ObligationDefaulted",
        "day": 1,
        "contract_id": contract_id,
        "debtor": debtor,
        "creditor": creditor,
        "contract_kind": "payable",
        "shortfall": shortfall,
        "amount_paid": 100 - shortfall,
        "original_amount": 100,
        "amount": shortfall,
    }


def _obligation_written_off(amount, contract_id="PAY_002", debtor="H1", creditor="H3"):
    """Create an ObligationWrittenOff event for a payable."""
    return {
        "kind": "ObligationWrittenOff",
        "day": 1,
        "contract_id": contract_id,
        "debtor": debtor,
        "creditor": creditor,
        "contract_kind": "payable",
        "amount": amount,
    }


class TestPayableDefaultLossTriggerContract:
    """ObligationDefaulted shortfall counts toward payable_default_loss."""

    def test_single_trigger_default(self):
        events = [_obligation_defaulted(shortfall=72)]
        m = compute_run_level_metrics(events)
        assert m["payable_default_loss"] == 72
        assert m["total_loss"] == 72

    def test_multiple_trigger_defaults(self):
        events = [
            _obligation_defaulted(shortfall=72, contract_id="PAY_001", debtor="H1"),
            _obligation_defaulted(shortfall=50, contract_id="PAY_002", debtor="H3"),
        ]
        m = compute_run_level_metrics(events)
        assert m["payable_default_loss"] == 122
        assert m["total_loss"] == 122

    def test_zero_shortfall_no_loss(self):
        events = [_obligation_defaulted(shortfall=0)]
        m = compute_run_level_metrics(events)
        assert m["payable_default_loss"] == 0


class TestPayableDefaultLossWrittenOff:
    """ObligationWrittenOff for payables still counts (remaining liabilities)."""

    def test_written_off_payable(self):
        events = [_obligation_written_off(amount=200)]
        m = compute_run_level_metrics(events)
        assert m["payable_default_loss"] == 200
        assert m["total_loss"] == 200


class TestPayableDefaultLossBothChannels:
    """Trigger shortfall + remaining write-offs both contribute."""

    def test_trigger_plus_writeoff(self):
        events = [
            _obligation_defaulted(shortfall=72),
            _obligation_written_off(amount=150),
        ]
        m = compute_run_level_metrics(events)
        assert m["payable_default_loss"] == 222
        assert m["total_loss"] == 222

    def test_total_loss_includes_deposit_loss(self):
        """total_loss = payable_default_loss + deposit_loss_gross."""
        events = [
            _obligation_defaulted(shortfall=72),
            {
                "kind": "ObligationWrittenOff",
                "day": 1,
                "contract_kind": "bank_deposit",
                "amount": 500,
                "debtor": "bank_1",
                "creditor": "H1",
            },
        ]
        m = compute_run_level_metrics(events)
        assert m["payable_default_loss"] == 72
        assert m["deposit_loss_gross"] == 500
        assert m["total_loss"] == 572


class TestNonPayableDefaultsIgnored:
    """ObligationDefaulted for non-payable contracts doesn't add to payable_default_loss."""

    def test_loan_default_not_counted(self):
        events = [
            {
                "kind": "ObligationDefaulted",
                "day": 1,
                "contract_kind": "bank_loan",
                "shortfall": 300,
                "amount": 300,
            },
        ]
        m = compute_run_level_metrics(events)
        assert m["payable_default_loss"] == 0

    def test_written_off_deposit_not_counted_as_payable(self):
        events = [
            {
                "kind": "ObligationWrittenOff",
                "day": 1,
                "contract_kind": "bank_deposit",
                "amount": 400,
            },
        ]
        m = compute_run_level_metrics(events)
        assert m["payable_default_loss"] == 0
        assert m["deposit_loss_gross"] == 400


class TestNoEventsBaseline:
    """No events → zero payable_default_loss."""

    def test_empty_events(self):
        m = compute_run_level_metrics(events=[])
        assert m["payable_default_loss"] == 0
        assert m["total_loss"] == 0
