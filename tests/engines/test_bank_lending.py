"""Unit tests for bank lending phase.

Tests cover:
1. BankLoan instrument properties (maturity_day, repayment_amount, is_due)
2. _find_eligible_borrowers identifies firms with shortfalls
3. _execute_bank_loan creates instrument and credits deposit
4. Bank loan repayment retires instruments and debits deposits
5. _increase_deposit adds to existing deposit
6. _increase_deposit creates new deposit when none exists
"""

from decimal import Decimal

from bilancio.decision.profiles import BankProfile
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.instruments.bank_loan import BankLoan
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.bank_lending import (
    _execute_bank_loan,
    _find_eligible_borrowers,
    _increase_deposit,
    run_bank_loan_repayments,
)
from bilancio.engines.banking_subsystem import (
    BankLoanRecord,
    _get_deposit_at_bank,
    initialize_banking_subsystem,
)
from bilancio.engines.system import System
from bilancio.domain.policy import PolicyEngine
from bilancio.ops.banking import deposit_cash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lending_system() -> System:
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

    # Mint cash, deposit, and mint reserves
    system.mint_cash(to_agent_id="H_1", amount=1000)
    system.mint_cash(to_agent_id="H_2", amount=1000)
    deposit_cash(system, "H_1", "bank_1", 1000)
    deposit_cash(system, "H_2", "bank_2", 1000)
    system.mint_reserves(to_bank_id="bank_1", amount=5000)
    system.mint_reserves(to_bank_id="bank_2", amount=5000)

    return system


class TestBankLoanInstrument:
    """Tests for the BankLoan instrument type."""

    def test_bank_loan_properties(self):
        """Verify BankLoan computes maturity_day, repayment_amount correctly."""
        loan = BankLoan(
            id="BL_1",
            kind=InstrumentKind.BANK_LOAN,
            amount=1000,
            denom="USD",
            asset_holder_id="bank_1",
            liability_issuer_id="H_1",
            rate=Decimal("0.05"),
            issuance_day=3,
            maturity_days=5,
        )

        assert loan.maturity_day == 8  # 3 + 5
        assert loan.repayment_amount == 1050  # 1000 * 1.05
        assert loan.interest_amount == 50  # 1050 - 1000
        assert loan.principal == 1000

    def test_bank_loan_is_due(self):
        """Verify is_due returns True when current_day >= maturity_day."""
        loan = BankLoan(
            id="BL_2",
            kind=InstrumentKind.BANK_LOAN,
            amount=500,
            denom="USD",
            asset_holder_id="bank_1",
            liability_issuer_id="H_1",
            rate=Decimal("0.02"),
            issuance_day=0,
            maturity_days=5,
        )

        assert not loan.is_due(4)  # Before maturity
        assert loan.is_due(5)      # On maturity day
        assert loan.is_due(6)      # After maturity day

    def test_bank_loan_post_init_sets_kind(self):
        """__post_init__ sets kind to BANK_LOAN regardless of input."""
        loan = BankLoan(
            id="BL_3",
            kind=InstrumentKind.CASH,  # Intentionally wrong
            amount=100,
            denom="USD",
            asset_holder_id="bank_1",
            liability_issuer_id="H_1",
        )
        assert loan.kind == InstrumentKind.BANK_LOAN


class TestFindEligibleBorrowers:
    """Tests for _find_eligible_borrowers."""

    def test_find_eligible_borrowers(self):
        """Firm with payable due soon and insufficient cash appears as eligible."""
        system = _make_lending_system()
        profile = BankProfile()
        kappa = Decimal("1.0")

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        # Create a payable that H_1 owes (liability), due within the loan horizon
        payable = Payable(
            id="PAY_1",
            kind=InstrumentKind.PAYABLE,
            amount=2000,  # More than H_1's deposit of 1000
            denom="USD",
            asset_holder_id="H_2",   # creditor
            liability_issuer_id="H_1",  # debtor
            due_day=2,
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.append(payable.id)
        system.state.agents["H_2"].asset_ids.append(payable.id)

        eligible = _find_eligible_borrowers(system, subsystem, current_day=0)

        # H_1 should appear: owes 2000, has 1000 deposit => shortfall = 1000
        borrower_ids = [bid for bid, _ in eligible]
        assert "H_1" in borrower_ids

        # Verify the shortfall amount
        for bid, shortfall in eligible:
            if bid == "H_1":
                assert shortfall == 1000  # 2000 - 1000

    def test_find_eligible_borrowers_no_shortfall(self):
        """Firm with sufficient liquidity does not appear as eligible."""
        system = _make_lending_system()
        profile = BankProfile()
        kappa = Decimal("1.0")

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        # Payable of 500, but H_1 has 1000 in deposits => no shortfall
        payable = Payable(
            id="PAY_2",
            kind=InstrumentKind.PAYABLE,
            amount=500,
            denom="USD",
            asset_holder_id="H_2",
            liability_issuer_id="H_1",
            due_day=2,
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.append(payable.id)
        system.state.agents["H_2"].asset_ids.append(payable.id)

        eligible = _find_eligible_borrowers(system, subsystem, current_day=0)

        borrower_ids = [bid for bid, _ in eligible]
        assert "H_1" not in borrower_ids

    def test_defaulted_agent_excluded(self):
        """Defaulted agents are excluded from eligible borrowers."""
        system = _make_lending_system()
        profile = BankProfile()
        kappa = Decimal("1.0")

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        # Mark H_1 as defaulted
        system.state.agents["H_1"].defaulted = True

        # Create a payable that would cause a shortfall
        payable = Payable(
            id="PAY_3",
            kind=InstrumentKind.PAYABLE,
            amount=5000,
            denom="USD",
            asset_holder_id="H_2",
            liability_issuer_id="H_1",
            due_day=2,
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.append(payable.id)
        system.state.agents["H_2"].asset_ids.append(payable.id)

        eligible = _find_eligible_borrowers(system, subsystem, current_day=0)

        borrower_ids = [bid for bid, _ in eligible]
        assert "H_1" not in borrower_ids


class TestExecuteBankLoan:
    """Tests for _execute_bank_loan."""

    def test_execute_bank_loan(self):
        """Execute a loan: BankLoan instrument created, deposit increased, bank state updated."""
        system = _make_lending_system()
        profile = BankProfile()
        kappa = Decimal("1.0")

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        bank_state = subsystem.banks["bank_1"]
        amount = 500
        rate = Decimal("0.03")
        current_day = 1
        maturity = 5

        # Record deposit before loan
        deposit_before = _get_deposit_at_bank(system, "H_1", "bank_1")

        loan_id = _execute_bank_loan(
            system=system,
            banking=subsystem,
            bank_state=bank_state,
            borrower_id="H_1",
            amount=amount,
            rate=rate,
            current_day=current_day,
            maturity=maturity,
        )

        assert loan_id is not None

        # 1. BankLoan instrument should exist in system contracts
        contract = system.state.contracts[loan_id]
        assert contract.kind == InstrumentKind.BANK_LOAN
        assert contract.amount == amount
        assert contract.asset_holder_id == "bank_1"
        assert contract.liability_issuer_id == "H_1"

        # 2. Loan should be registered in agent registries
        assert loan_id in system.state.agents["bank_1"].asset_ids
        assert loan_id in system.state.agents["H_1"].liability_ids

        # 3. Borrower's deposit should have increased by loan amount
        deposit_after = _get_deposit_at_bank(system, "H_1", "bank_1")
        assert deposit_after == deposit_before + amount

        # 4. Bank state should track the loan
        assert loan_id in bank_state.outstanding_loans
        record = bank_state.outstanding_loans[loan_id]
        assert record.principal == amount
        assert record.rate == rate
        assert record.maturity_day == current_day + maturity
        assert bank_state.total_loan_principal == amount


class TestBankLoanRepayment:
    """Tests for bank loan repayment."""

    def test_bank_loan_repayment(self):
        """Create and repay a loan, verify instruments removed and deposits debited."""
        system = _make_lending_system()
        profile = BankProfile()
        kappa = Decimal("1.0")

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        bank_state = subsystem.banks["bank_1"]
        amount = 500
        rate = Decimal("0.02")
        current_day = 0
        maturity = 3

        # Execute a loan first
        loan_id = _execute_bank_loan(
            system=system,
            banking=subsystem,
            bank_state=bank_state,
            borrower_id="H_1",
            amount=amount,
            rate=rate,
            current_day=current_day,
            maturity=maturity,
        )

        assert loan_id is not None

        deposit_before_repay = _get_deposit_at_bank(system, "H_1", "bank_1")
        repayment_amount = bank_state.outstanding_loans[loan_id].repayment_amount

        # Advance to maturity day and run repayments
        repay_day = current_day + maturity
        events = run_bank_loan_repayments(system, repay_day, subsystem)

        # Should have a repayment event
        repaid_events = [e for e in events if e["kind"] == "BankLoanRepaid"]
        assert len(repaid_events) == 1
        assert repaid_events[0]["loan_id"] == loan_id
        assert repaid_events[0]["borrower"] == "H_1"
        assert repaid_events[0]["bank"] == "bank_1"

        # Loan should be removed from bank state
        assert loan_id not in bank_state.outstanding_loans

        # Loan instrument should be removed from system contracts
        assert loan_id not in system.state.contracts

        # Loan should be removed from agent registries
        assert loan_id not in system.state.agents["bank_1"].asset_ids
        assert loan_id not in system.state.agents["H_1"].liability_ids

        # Deposit should have been debited by the repayment amount
        deposit_after_repay = _get_deposit_at_bank(system, "H_1", "bank_1")
        assert deposit_after_repay == deposit_before_repay - repayment_amount


class TestIncreaseDeposit:
    """Tests for _increase_deposit helper."""

    def test_increase_deposit_existing(self):
        """_increase_deposit adds to existing deposit when one exists."""
        system = _make_lending_system()

        # H_1 already has a deposit at bank_1 of 1000
        deposit_before = _get_deposit_at_bank(system, "H_1", "bank_1")
        assert deposit_before == 1000

        _increase_deposit(system, "H_1", "bank_1", 500)

        deposit_after = _get_deposit_at_bank(system, "H_1", "bank_1")
        assert deposit_after == 1500

    def test_increase_deposit_new(self):
        """_increase_deposit creates new deposit when none exists at that bank."""
        system = _make_lending_system()

        # H_1 has no deposit at bank_2
        deposit_before = _get_deposit_at_bank(system, "H_1", "bank_2")
        assert deposit_before == 0

        _increase_deposit(system, "H_1", "bank_2", 300)

        deposit_after = _get_deposit_at_bank(system, "H_1", "bank_2")
        assert deposit_after == 300

        # New deposit instrument should be in H_1's assets and bank_2's liabilities
        found = False
        for cid in system.state.agents["H_1"].asset_ids:
            contract = system.state.contracts.get(cid)
            if (
                contract
                and contract.kind == InstrumentKind.BANK_DEPOSIT
                and contract.liability_issuer_id == "bank_2"
            ):
                found = True
                assert contract.amount == 300
                assert contract.asset_holder_id == "H_1"
                # Also verify it appears in bank_2's liabilities
                assert cid in system.state.agents["bank_2"].liability_ids
        assert found, "New deposit instrument should exist"

    def test_increase_deposit_nonexistent_agent(self):
        """_increase_deposit is a no-op for nonexistent agent (does not raise)."""
        system = _make_lending_system()
        # Should not raise
        _increase_deposit(system, "nonexistent", "bank_1", 100)
