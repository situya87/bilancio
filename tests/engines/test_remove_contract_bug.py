"""Regression test: _remove_contract() must handle CB_LOAN contracts.

Before Plan 037 fix, _remove_contract() only updated counters for
CASH and RESERVE_DEPOSIT but not CB_LOAN, so cb_loans_outstanding
was never decremented when a CB loan was written off.
"""

from decimal import Decimal

from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.cb_loan import CBLoan
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.settlement import _remove_contract
from bilancio.engines.system import System


def _make_system_with_cb_loan() -> tuple[System, str]:
    """Create a system with a CB loan and return (system, loan_id)."""
    system = System(policy=PolicyEngine.default())
    cb = CentralBank(id="cb", name="CB", kind="central_bank")
    bank = Bank(id="bank_1", name="Bank 1", kind="bank")
    system.add_agent(cb)
    system.add_agent(bank)

    # Create a CB loan directly (bypassing cb_lend_reserves for simplicity)
    loan_id = "L_test_001"
    loan = CBLoan(
        id=loan_id,
        kind=InstrumentKind.CB_LOAN,
        amount=1000,
        denom="X",
        asset_holder_id="cb",
        liability_issuer_id="bank_1",
        cb_rate=Decimal("0.03"),
        issuance_day=0,
    )
    system.add_contract(loan)
    system.state.cb_loans_outstanding += 1000
    return system, loan_id


class TestRemoveContractCBLoan:
    """Verify _remove_contract decrements cb_loans_outstanding for CB_LOAN."""

    def test_cb_loan_counter_decremented(self):
        """After removing a CB loan, cb_loans_outstanding should decrease."""
        system, loan_id = _make_system_with_cb_loan()
        assert system.state.cb_loans_outstanding == 1000

        _remove_contract(system, loan_id)

        assert system.state.cb_loans_outstanding == 0
        assert loan_id not in system.state.contracts

    def test_multiple_cb_loans(self):
        """Removing one CB loan only decrements by that loan's amount."""
        system, loan_id1 = _make_system_with_cb_loan()

        # Add a second loan
        loan_id2 = "L_test_002"
        loan2 = CBLoan(
            id=loan_id2,
            kind=InstrumentKind.CB_LOAN,
            amount=500,
            denom="X",
            asset_holder_id="cb",
            liability_issuer_id="bank_1",
            cb_rate=Decimal("0.03"),
            issuance_day=0,
        )
        system.add_contract(loan2)
        system.state.cb_loans_outstanding += 500
        assert system.state.cb_loans_outstanding == 1500

        _remove_contract(system, loan_id1)
        assert system.state.cb_loans_outstanding == 500

        _remove_contract(system, loan_id2)
        assert system.state.cb_loans_outstanding == 0

    def test_removing_nonexistent_contract_is_noop(self):
        """Removing an already-removed contract does nothing."""
        system, loan_id = _make_system_with_cb_loan()
        _remove_contract(system, loan_id)
        # Second remove should be a no-op
        _remove_contract(system, loan_id)
        assert system.state.cb_loans_outstanding == 0
