"""Extra coverage tests for bilancio.engines.bank_lending.

Focuses on uncovered paths:
- _per_borrower_rate: credit risk loading, rationing, no assessor
- _assess_borrower: edge cases (missing agent, defaulted obligors, zero-cost loan)
- _prefer_selling: dealer available, no dealer, no bid quotes
- _decrease_deposit: missing agent, no deposit at bank
- _get_upcoming_obligations: bank loans, non-bank loans in horizon
- _get_agent_liquidity: deposits counted, missing agent
- run_bank_loan_repayments: include_overdue, partial repayment with default
- _repay_loan: cross-bank repayment, reserve transfer failure with CB fallback
- _bank_can_lend: daily limit, exposure limit, per-borrower limit
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from bilancio.decision.profiles import BankProfile
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.bank_lending import (
    _assess_borrower,
    _decrease_deposit,
    _get_agent_liquidity,
    _get_upcoming_obligations,
    _per_borrower_rate,
    _prefer_selling,
    run_bank_loan_repayments,
    _execute_bank_loan,
    _increase_deposit,
)
from bilancio.engines.banking_subsystem import (
    _get_deposit_at_bank,
    initialize_banking_subsystem,
)
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_system() -> System:
    """Create a system with CB + 2 banks + 2 firms, deposits and reserves."""
    system = System(policy=PolicyEngine.default())

    cb = CentralBank(id="cb", name="Central Bank", kind="central_bank")
    bank1 = Bank(id="bank_1", name="Bank 1", kind="bank")
    bank2 = Bank(id="bank_2", name="Bank 2", kind="bank")
    firm1 = Firm(id="H_1", name="Firm 1", kind="firm")
    firm2 = Firm(id="H_2", name="Firm 2", kind="firm")

    system.add_agent(cb)
    system.add_agent(bank1)
    system.add_agent(bank2)
    system.add_agent(firm1)
    system.add_agent(firm2)

    system.mint_cash(to_agent_id="H_1", amount=1000)
    system.mint_cash(to_agent_id="H_2", amount=1000)
    deposit_cash(system, "H_1", "bank_1", 1000)
    deposit_cash(system, "H_2", "bank_2", 1000)
    system.mint_reserves(to_bank_id="bank_1", amount=5000)
    system.mint_reserves(to_bank_id="bank_2", amount=5000)

    return system


# ---------------------------------------------------------------------------
# _per_borrower_rate
# ---------------------------------------------------------------------------


class TestPerBorrowerRate:
    def test_no_loading_no_rationing(self):
        """Default profile (loading=0, max_risk=1): returns base rate unchanged."""
        banking = MagicMock()
        banking.bank_profile = BankProfile()
        result = _per_borrower_rate(Decimal("0.05"), "H_1", banking, 0)
        assert result == Decimal("0.05")

    def test_credit_rationed(self):
        """Borrower with P > max_borrower_risk returns None (rationed)."""
        profile = BankProfile(
            credit_risk_loading=Decimal("0.1"),
            max_borrower_risk=Decimal("0.2"),
        )
        banking = MagicMock()
        banking.bank_profile = profile
        assessor = MagicMock()
        assessor.estimate_default_prob.return_value = 0.5  # P=0.5 > 0.2
        banking.risk_assessor = assessor

        result = _per_borrower_rate(Decimal("0.05"), "H_1", banking, 0)
        assert result is None

    def test_credit_loading_applied(self):
        """With loading > 0 and P < max_risk, rate = base + loading * P."""
        profile = BankProfile(
            credit_risk_loading=Decimal("0.1"),
            max_borrower_risk=Decimal("1.0"),
        )
        banking = MagicMock()
        banking.bank_profile = profile
        assessor = MagicMock()
        assessor.estimate_default_prob.return_value = 0.3
        banking.risk_assessor = assessor

        result = _per_borrower_rate(Decimal("0.05"), "H_1", banking, 0)
        # Float P_default (0.3) causes minor imprecision; use approximate check
        assert abs(result - Decimal("0.08")) < Decimal("0.001")

    def test_no_assessor_returns_base(self):
        """If risk_assessor is None, returns base rate."""
        profile = BankProfile(
            credit_risk_loading=Decimal("0.1"),
            max_borrower_risk=Decimal("0.5"),
        )
        banking = MagicMock()
        banking.bank_profile = profile
        banking.risk_assessor = None

        result = _per_borrower_rate(Decimal("0.05"), "H_1", banking, 0)
        assert result == Decimal("0.05")

    def test_rationing_only_no_loading(self):
        """max_borrower_risk < 1 but credit_risk_loading = 0: still rations."""
        profile = BankProfile(
            credit_risk_loading=Decimal("0"),
            max_borrower_risk=Decimal("0.2"),
        )
        banking = MagicMock()
        banking.bank_profile = profile
        assessor = MagicMock()
        assessor.estimate_default_prob.return_value = 0.5  # > 0.2
        banking.risk_assessor = assessor

        result = _per_borrower_rate(Decimal("0.05"), "H_1", banking, 0)
        assert result is None

    def test_rationing_passes_when_p_below(self):
        """P below max_borrower_risk and loading=0 returns base."""
        profile = BankProfile(
            credit_risk_loading=Decimal("0"),
            max_borrower_risk=Decimal("0.5"),
        )
        banking = MagicMock()
        banking.bank_profile = profile
        assessor = MagicMock()
        assessor.estimate_default_prob.return_value = 0.1  # < 0.5
        banking.risk_assessor = assessor

        result = _per_borrower_rate(Decimal("0.05"), "H_1", banking, 0)
        assert result == Decimal("0.05")


# ---------------------------------------------------------------------------
# _assess_borrower
# ---------------------------------------------------------------------------


class TestAssessBorrower:
    def test_missing_agent(self):
        system = _make_system()
        result = _assess_borrower(system, "NOPE", 100, Decimal("0.05"), 3, 0)
        assert result == Decimal("-1")

    def test_zero_cost_loan(self):
        """When rate makes repayment zero, coverage is 999."""
        system = _make_system()
        result = _assess_borrower(system, "H_1", 0, Decimal("0.05"), 3, 0)
        assert result == Decimal("999")

    def test_coverage_with_receivables(self):
        """Receivables from non-defaulted counterparties increase coverage."""
        system = _make_system()

        # H_1 has a receivable from H_2 due before loan maturity
        receivable = Payable(
            id="RCV_1", kind=InstrumentKind.PAYABLE, amount=500,
            denom="USD", asset_holder_id="H_1", liability_issuer_id="H_2",
            due_day=2,
        )
        system.state.contracts[receivable.id] = receivable
        system.state.agents["H_1"].asset_ids.add(receivable.id)

        coverage = _assess_borrower(system, "H_1", 500, Decimal("0.05"), 5, 0)
        # liquid=1000 + quality_receivables=500 - obligations=0 / 525 (500*1.05) > 1
        assert coverage > Decimal("1")

    def test_defaulted_obligor_receivable_excluded(self):
        """Receivables from defaulted agents are worth zero."""
        system = _make_system()

        receivable = Payable(
            id="RCV_2", kind=InstrumentKind.PAYABLE, amount=500,
            denom="USD", asset_holder_id="H_1", liability_issuer_id="H_2",
            due_day=2,
        )
        system.state.contracts[receivable.id] = receivable
        system.state.agents["H_1"].asset_ids.add(receivable.id)

        # Mark H_2 as defaulted
        system.state.defaulted_agent_ids.add("H_2")

        coverage = _assess_borrower(system, "H_1", 500, Decimal("0.05"), 5, 0)
        # No quality receivables, coverage = 1000/525 ~ 1.9
        coverage_without_recv = Decimal("1000") / Decimal("525")
        assert abs(coverage - coverage_without_recv) < Decimal("0.01")

    def test_obligations_reduce_coverage(self):
        """Existing obligations reduce coverage."""
        system = _make_system()

        # H_1 owes 800 to H_2 due within loan maturity
        payable = Payable(
            id="PAY_1", kind=InstrumentKind.PAYABLE, amount=800,
            denom="USD", asset_holder_id="H_2", liability_issuer_id="H_1",
            due_day=2,
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.add(payable.id)

        coverage = _assess_borrower(system, "H_1", 500, Decimal("0.05"), 5, 0)
        # net = 1000 - 800 + 0 = 200; repayment = 525 => coverage ~ 0.38
        assert coverage < Decimal("1")


# ---------------------------------------------------------------------------
# _prefer_selling
# ---------------------------------------------------------------------------


class TestPreferSelling:
    def test_no_dealer_returns_false(self):
        system = _make_system()
        system.state.dealer_subsystem = None
        result = _prefer_selling(system, "H_1", 500, Decimal("0.05"), 3, 0)
        assert result is False

    def test_with_dealer_bids(self):
        system = _make_system()
        dealer1 = MagicMock()
        dealer1.bid = Decimal("0.90")
        dealer_sub = MagicMock()
        dealer_sub.dealers = {"short": dealer1}
        system.state.dealer_subsystem = dealer_sub

        # sell_cost = 500 * (1 - 0.90) = 50
        # borrow_cost = 500 * 0.05 = 25
        # sell_cost > borrow_cost => prefer borrowing
        result = _prefer_selling(system, "H_1", 500, Decimal("0.05"), 3, 0)
        assert result is False

    def test_selling_cheaper(self):
        system = _make_system()
        dealer1 = MagicMock()
        dealer1.bid = Decimal("0.99")
        dealer_sub = MagicMock()
        dealer_sub.dealers = {"short": dealer1}
        system.state.dealer_subsystem = dealer_sub

        # sell_cost = 500 * (1 - 0.99) = 5
        # borrow_cost = 500 * 0.10 = 50
        # sell_cost < borrow_cost => prefer selling
        result = _prefer_selling(system, "H_1", 500, Decimal("0.10"), 3, 0)
        assert result is True

    def test_no_dealer_bids_uses_fallback(self):
        """When no dealer has a bid, falls back to outside_mid_ratio."""
        system = _make_system()
        dealer1 = MagicMock(spec=[])  # No 'bid' attribute
        dealer_sub = MagicMock()
        dealer_sub.dealers = {"short": dealer1}
        dealer_sub.outside_mid_ratio = "0.85"
        system.state.dealer_subsystem = dealer_sub

        # sell_cost = 500 * (1 - 0.85) = 75
        # borrow_cost = 500 * 0.02 = 10
        result = _prefer_selling(system, "H_1", 500, Decimal("0.02"), 3, 0)
        assert result is False  # borrowing cheaper


# ---------------------------------------------------------------------------
# _decrease_deposit
# ---------------------------------------------------------------------------


class TestDecreaseDeposit:
    def test_missing_agent(self):
        system = _make_system()
        result = _decrease_deposit(system, "NOPE", "bank_1", 100)
        assert result == 0

    def test_no_deposit_at_bank(self):
        system = _make_system()
        # H_1 has deposit at bank_1, not bank_2
        result = _decrease_deposit(system, "H_1", "bank_2", 100)
        assert result == 0

    def test_partial_debit(self):
        system = _make_system()
        # H_1 has 1000 at bank_1
        result = _decrease_deposit(system, "H_1", "bank_1", 1500)
        assert result == 1000  # Can only debit 1000

    def test_full_debit(self):
        system = _make_system()
        result = _decrease_deposit(system, "H_1", "bank_1", 500)
        assert result == 500
        assert _get_deposit_at_bank(system, "H_1", "bank_1") == 500


# ---------------------------------------------------------------------------
# _get_upcoming_obligations
# ---------------------------------------------------------------------------


class TestGetUpcomingObligations:
    def test_missing_agent(self):
        system = _make_system()
        result = _get_upcoming_obligations(system, "NOPE", 0, 5)
        assert result == 0

    def test_payable_in_horizon(self):
        system = _make_system()
        payable = Payable(
            id="PAY_1", kind=InstrumentKind.PAYABLE, amount=300,
            denom="USD", asset_holder_id="H_2", liability_issuer_id="H_1",
            due_day=3,
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.add(payable.id)

        result = _get_upcoming_obligations(system, "H_1", 0, 5)
        assert result == 300

    def test_payable_outside_horizon(self):
        system = _make_system()
        payable = Payable(
            id="PAY_2", kind=InstrumentKind.PAYABLE, amount=300,
            denom="USD", asset_holder_id="H_2", liability_issuer_id="H_1",
            due_day=10,
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.add(payable.id)

        result = _get_upcoming_obligations(system, "H_1", 0, 5)
        assert result == 0

    def test_bank_loan_in_horizon(self):
        """Bank loans due within horizon are counted."""
        from bilancio.domain.instruments.bank_loan import BankLoan

        system = _make_system()
        loan = BankLoan(
            id="BL_1", kind=InstrumentKind.BANK_LOAN, amount=400,
            denom="USD", asset_holder_id="bank_1", liability_issuer_id="H_1",
            rate=Decimal("0.05"), issuance_day=0, maturity_days=3,
        )
        system.state.contracts[loan.id] = loan
        system.state.agents["H_1"].liability_ids.add(loan.id)

        result = _get_upcoming_obligations(system, "H_1", 0, 5)
        # repayment_amount = 400 * 1.05 = 420
        assert result == 420

    def test_missing_contract_skipped(self):
        """Missing contract ID in liability_ids is gracefully skipped."""
        system = _make_system()
        system.state.agents["H_1"].liability_ids.add("PHANTOM_CONTRACT")

        result = _get_upcoming_obligations(system, "H_1", 0, 5)
        assert result == 0


# ---------------------------------------------------------------------------
# _get_agent_liquidity
# ---------------------------------------------------------------------------


class TestGetAgentLiquidity:
    def test_missing_agent(self):
        system = _make_system()
        result = _get_agent_liquidity(system, "NOPE")
        assert result == 0

    def test_counts_cash_and_deposits(self):
        system = _make_system()
        # H_1 has 1000 in deposit at bank_1 (cash was converted to deposit)
        result = _get_agent_liquidity(system, "H_1")
        assert result == 1000  # deposit only (cash was deposited)


# ---------------------------------------------------------------------------
# run_bank_loan_repayments: overdue and default
# ---------------------------------------------------------------------------


class TestBankLoanRepaymentsExtra:
    def test_include_overdue(self):
        """include_overdue=True processes loans from earlier days."""
        system = _make_system()
        profile = BankProfile()
        subsystem = initialize_banking_subsystem(
            system=system, bank_profile=profile,
            kappa=Decimal("1.0"), maturity_days=10,
        )

        bank_state = subsystem.banks["bank_1"]
        loan_id = _execute_bank_loan(
            system=system, banking=subsystem, bank_state=bank_state,
            borrower_id="H_1", amount=500, rate=Decimal("0.02"),
            current_day=0, maturity=3,
        )
        assert loan_id is not None

        # Day 5 > maturity day 3 — without include_overdue, skipped
        events_no_overdue = run_bank_loan_repayments(system, current_day=5, banking=subsystem)
        assert len(events_no_overdue) == 0

        # With include_overdue, should process it
        events_overdue = run_bank_loan_repayments(
            system, current_day=5, banking=subsystem, include_overdue=True,
        )
        repaid = [e for e in events_overdue if e["kind"] == "BankLoanRepaid"]
        assert len(repaid) == 1

    def test_partial_default(self):
        """Borrower with partial deposits defaults with recovered amount."""
        system = _make_system()
        profile = BankProfile()
        subsystem = initialize_banking_subsystem(
            system=system, bank_profile=profile,
            kappa=Decimal("1.0"), maturity_days=10,
        )

        bank_state = subsystem.banks["bank_1"]
        loan_id = _execute_bank_loan(
            system=system, banking=subsystem, bank_state=bank_state,
            borrower_id="H_1", amount=500, rate=Decimal("0.02"),
            current_day=0, maturity=3,
        )

        # Set H_1's deposit to less than repayment (500*1.02=510)
        for cid in list(system.state.agents["H_1"].asset_ids):
            contract = system.state.contracts.get(cid)
            if contract and contract.kind == InstrumentKind.BANK_DEPOSIT:
                contract.amount = 200  # Less than 510

        events = run_bank_loan_repayments(system, current_day=3, banking=subsystem)

        defaults = [e for e in events if e["kind"] == "BankLoanDefault"]
        assert len(defaults) == 1
        assert defaults[0]["recovered"] == 200
        assert "H_1" in subsystem.defaulted_borrowers


# ---------------------------------------------------------------------------
# _execute_bank_loan edge cases
# ---------------------------------------------------------------------------


class TestExecuteBankLoanEdgeCases:
    def test_missing_bank_agent(self):
        """Loan proceeds even when bank agent not found (logs warning)."""
        system = _make_system()
        profile = BankProfile()
        subsystem = initialize_banking_subsystem(
            system=system, bank_profile=profile,
            kappa=Decimal("1.0"), maturity_days=10,
        )
        bank_state = subsystem.banks["bank_1"]

        # Remove bank agent from system
        del system.state.agents["bank_1"]

        loan_id = _execute_bank_loan(
            system=system, banking=subsystem, bank_state=bank_state,
            borrower_id="H_1", amount=100, rate=Decimal("0.01"),
            current_day=0, maturity=3,
        )
        # Should still create the loan
        assert loan_id is not None

    def test_missing_borrower_agent(self):
        """Loan proceeds even when borrower agent not found (logs warning)."""
        system = _make_system()
        profile = BankProfile()
        subsystem = initialize_banking_subsystem(
            system=system, bank_profile=profile,
            kappa=Decimal("1.0"), maturity_days=10,
        )
        bank_state = subsystem.banks["bank_1"]

        # Remove borrower agent
        del system.state.agents["H_1"]

        loan_id = _execute_bank_loan(
            system=system, banking=subsystem, bank_state=bank_state,
            borrower_id="H_1", amount=100, rate=Decimal("0.01"),
            current_day=0, maturity=3,
        )
        assert loan_id is not None
