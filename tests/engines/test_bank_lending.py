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
    run_bank_lending_phase,
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


# ---------------------------------------------------------------------------
# Helpers for lending-discipline tests
# ---------------------------------------------------------------------------


def _setup_lending_scenario():
    """Create system + subsystem with H_1 having a shortfall (eligible to borrow).

    Returns (system, subsystem) with:
    - H_1: deposit=1000, payable=2000 due day 2 → shortfall=1000
    - Banks have quotes refreshed so lending can proceed.
    """
    system = _make_lending_system()
    profile = BankProfile()
    kappa = Decimal("1.0")

    subsystem = initialize_banking_subsystem(
        system=system,
        bank_profile=profile,
        kappa=kappa,
        maturity_days=10,
    )

    # Create a payable that H_1 owes, due within lending horizon
    payable = Payable(
        id="PAY_test",
        kind=InstrumentKind.PAYABLE,
        amount=2000,  # More than H_1's 1000 deposit → shortfall=1000
        denom="USD",
        asset_holder_id="H_2",
        liability_issuer_id="H_1",
        due_day=2,
    )
    system.state.contracts[payable.id] = payable
    system.state.agents["H_1"].liability_ids.append(payable.id)
    system.state.agents["H_2"].asset_ids.append(payable.id)

    # Refresh quotes so banks can lend
    subsystem.refresh_all_quotes(system, current_day=0)

    return system, subsystem


class TestFoolMeOnce:
    """Tests for the 'fool me once' policy: borrowers who defaulted on a
    bank loan are permanently blocked from future borrowing."""

    def test_loan_default_populates_defaulted_borrowers(self):
        """When a borrower defaults on a loan, they are added to defaulted_borrowers."""
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

        # Issue a loan to H_1
        loan_id = _execute_bank_loan(
            system=system,
            banking=subsystem,
            bank_state=bank_state,
            borrower_id="H_1",
            amount=500,
            rate=Decimal("0.05"),
            current_day=0,
            maturity=3,
        )
        assert loan_id is not None
        assert "H_1" not in subsystem.defaulted_borrowers

        # Drain H_1's deposit so repayment fails (force default)
        deposit = _get_deposit_at_bank(system, "H_1", "bank_1")
        # Find the deposit instrument and zero it
        for cid in list(system.state.agents["H_1"].asset_ids):
            contract = system.state.contracts.get(cid)
            if (
                contract
                and contract.kind == InstrumentKind.BANK_DEPOSIT
                and contract.liability_issuer_id == "bank_1"
            ):
                contract.amount = 0

        # Run repayments at maturity → should default
        events = run_bank_loan_repayments(system, current_day=3, banking=subsystem)

        default_events = [e for e in events if e["kind"] == "BankLoanDefault"]
        assert len(default_events) == 1
        assert default_events[0]["borrower"] == "H_1"

        # H_1 should now be in defaulted_borrowers
        assert "H_1" in subsystem.defaulted_borrowers

    def test_defaulted_borrower_blocked_from_lending_phase(self):
        """A borrower in defaulted_borrowers receives no loan from run_bank_lending_phase."""
        system, subsystem = _setup_lending_scenario()

        # Pre-populate H_1 as a defaulted borrower
        subsystem.defaulted_borrowers.add("H_1")

        events = run_bank_lending_phase(system, current_day=0, banking=subsystem)

        loan_events = [e for e in events if e["kind"] == "BankLoanIssued"]
        borrowers = [e["borrower"] for e in loan_events]
        assert "H_1" not in borrowers


class TestOneLoanAtATime:
    """Tests for the one-loan-at-a-time policy: borrowers with an outstanding
    loan at any bank are skipped during the lending phase."""

    def test_has_outstanding_loan_false_initially(self):
        """has_outstanding_loan returns False when no loans exist."""
        system = _make_lending_system()
        profile = BankProfile()

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=Decimal("1.0"),
            maturity_days=10,
        )

        assert not subsystem.has_outstanding_loan("H_1")

    def test_has_outstanding_loan_true_after_loan(self):
        """has_outstanding_loan returns True after a loan is issued."""
        system = _make_lending_system()
        profile = BankProfile()

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=Decimal("1.0"),
            maturity_days=10,
        )

        bank_state = subsystem.banks["bank_1"]
        loan_id = _execute_bank_loan(
            system=system,
            banking=subsystem,
            bank_state=bank_state,
            borrower_id="H_1",
            amount=500,
            rate=Decimal("0.03"),
            current_day=0,
            maturity=5,
        )
        assert loan_id is not None
        assert subsystem.has_outstanding_loan("H_1")
        # H_2 has no loan
        assert not subsystem.has_outstanding_loan("H_2")

    def test_borrower_with_outstanding_loan_blocked(self):
        """A borrower with an outstanding loan receives no second loan."""
        system, subsystem = _setup_lending_scenario()

        # Issue a loan to H_1 directly (simulating a prior day's lending)
        bank_state = subsystem.banks["bank_1"]
        loan_id = _execute_bank_loan(
            system=system,
            banking=subsystem,
            bank_state=bank_state,
            borrower_id="H_1",
            amount=200,
            rate=Decimal("0.03"),
            current_day=0,
            maturity=5,
        )
        assert loan_id is not None
        assert subsystem.has_outstanding_loan("H_1")

        # Refresh quotes (the execute above already does, but be explicit)
        subsystem.refresh_all_quotes(system, current_day=0)

        # Now run the lending phase — H_1 still has a shortfall but
        # should be blocked by the one-loan-at-a-time rule.
        events = run_bank_lending_phase(system, current_day=0, banking=subsystem)

        loan_events = [e for e in events if e["kind"] == "BankLoanIssued"]
        borrowers = [e["borrower"] for e in loan_events]
        assert "H_1" not in borrowers


# ---------------------------------------------------------------------------
# Plan 045: Settlement forecast & projected reserve check
# ---------------------------------------------------------------------------


def _make_settlement_system(
    reserves_1: int = 300,
    reserves_2: int = 300,
    deposits_1: int = 600,
    deposits_2: int = 600,
    firm1_bank: str = "bank_1",
    firm2_bank: str = "bank_2",
) -> System:
    """Create a system with 2 banks and 2 firms, with configurable reserves/deposits.

    Used for testing settlement forecast and projected reserve checks.
    ``firm1_bank`` / ``firm2_bank`` control where each firm's deposit is created.
    """
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

    # Fund firms and deposit at their assigned banks
    system.mint_cash(to_agent_id="H_1", amount=deposits_1)
    system.mint_cash(to_agent_id="H_2", amount=deposits_2)
    deposit_cash(system, "H_1", firm1_bank, deposits_1)
    deposit_cash(system, "H_2", firm2_bank, deposits_2)

    # Mint reserves for banks
    system.mint_reserves(to_bank_id="bank_1", amount=reserves_1)
    system.mint_reserves(to_bank_id="bank_2", amount=reserves_2)

    return system


class TestSettlementForecast:
    """Tests for compute_settlement_forecasts (Plan 045)."""

    def test_cross_bank_payable_creates_outflow(self):
        """A payable from H_1→H_2 (different banks) creates outflow for bank_1."""
        system = _make_settlement_system()
        profile = BankProfile()

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=Decimal("0.3"),
            maturity_days=10,
            trader_banks={"H_1": ["bank_1"], "H_2": ["bank_2"]},
        )

        # H_1 (bank_1 client) owes H_2 (bank_2 client) 500 due today
        payable = Payable(
            id="PAY_settle",
            kind=InstrumentKind.PAYABLE,
            amount=500,
            denom="USD",
            asset_holder_id="H_2",
            liability_issuer_id="H_1",
            due_day=0,
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.append(payable.id)
        system.state.agents["H_2"].asset_ids.append(payable.id)

        net = subsystem.compute_settlement_forecasts(system, current_day=0)

        # bank_1 should have outflow of 500 (positive = reserves leave)
        assert net["bank_1"] == 500
        # bank_2 should have inflow of 500 (negative = reserves arrive)
        assert net["bank_2"] == -500

    def test_intra_bank_payable_no_outflow(self):
        """A payable between two clients of the same bank creates no outflow."""
        # Both firms deposit at bank_1 so forecast scans find same bank
        system = _make_settlement_system(firm2_bank="bank_1")
        profile = BankProfile()

        # Both firms at bank_1
        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=Decimal("1.0"),
            maturity_days=10,
            trader_banks={"H_1": ["bank_1"], "H_2": ["bank_1"]},
        )

        payable = Payable(
            id="PAY_intra",
            kind=InstrumentKind.PAYABLE,
            amount=500,
            denom="USD",
            asset_holder_id="H_2",
            liability_issuer_id="H_1",
            due_day=0,
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.append(payable.id)
        system.state.agents["H_2"].asset_ids.append(payable.id)

        net = subsystem.compute_settlement_forecasts(system, current_day=0)

        # Both banks should have zero net — no reserve movement for intra-bank
        assert net["bank_1"] == 0
        assert net["bank_2"] == 0

    def test_defaulted_debtor_excluded(self):
        """Payables from defaulted debtors are excluded from forecast."""
        system = _make_settlement_system()
        profile = BankProfile()

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=Decimal("1.0"),
            maturity_days=10,
            trader_banks={"H_1": ["bank_1"], "H_2": ["bank_2"]},
        )

        system.state.agents["H_1"].defaulted = True

        payable = Payable(
            id="PAY_def",
            kind=InstrumentKind.PAYABLE,
            amount=500,
            denom="USD",
            asset_holder_id="H_2",
            liability_issuer_id="H_1",
            due_day=0,
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.append(payable.id)
        system.state.agents["H_2"].asset_ids.append(payable.id)

        net = subsystem.compute_settlement_forecasts(system, current_day=0)

        assert net["bank_1"] == 0
        assert net["bank_2"] == 0

    def test_future_payable_not_counted(self):
        """Payables due on a different day are not counted."""
        system = _make_settlement_system()
        profile = BankProfile()

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=Decimal("1.0"),
            maturity_days=10,
            trader_banks={"H_1": ["bank_1"], "H_2": ["bank_2"]},
        )

        payable = Payable(
            id="PAY_future",
            kind=InstrumentKind.PAYABLE,
            amount=500,
            denom="USD",
            asset_holder_id="H_2",
            liability_issuer_id="H_1",
            due_day=5,  # Not today
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.append(payable.id)
        system.state.agents["H_2"].asset_ids.append(payable.id)

        net = subsystem.compute_settlement_forecasts(system, current_day=0)

        assert net["bank_1"] == 0
        assert net["bank_2"] == 0

    def test_effective_creditor_used_for_transferred_payable(self):
        """Forecast uses effective_creditor (secondary market holder), not asset_holder_id."""
        # H_1 at bank_1, H_2 at bank_2
        system = _make_settlement_system()
        profile = BankProfile()

        # Add a third firm at bank_1 to be the secondary market holder
        firm3 = Firm(id="H_3", name="Firm 3", kind="firm")
        system.add_agent(firm3)
        system.mint_cash(to_agent_id="H_3", amount=100)
        deposit_cash(system, "H_3", "bank_1", 100)

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=Decimal("0.3"),
            maturity_days=10,
            trader_banks={
                "H_1": ["bank_1"],
                "H_2": ["bank_2"],
                "H_3": ["bank_1"],
            },
        )

        # H_1 (bank_1) owes H_2 (bank_2 original creditor),
        # but payable was transferred to H_3 (bank_1)
        payable = Payable(
            id="PAY_transferred",
            kind=InstrumentKind.PAYABLE,
            amount=400,
            denom="USD",
            asset_holder_id="H_2",  # original creditor at bank_2
            liability_issuer_id="H_1",
            due_day=0,
            holder_id="H_3",  # secondary market holder at bank_1
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.append(payable.id)

        net = subsystem.compute_settlement_forecasts(system, current_day=0)

        # effective_creditor = H_3 (at bank_1), debtor H_1 also at bank_1
        # → intra-bank, no reserve movement
        # Without the fix (using asset_holder_id=H_2 at bank_2), this would
        # show 400 outflow from bank_1 and -400 inflow to bank_2 — wrong.
        assert net["bank_1"] == 0
        assert net["bank_2"] == 0

    def test_multi_bank_debtor_splits_payment(self):
        """When debtor has deposits at two banks, forecast splits by balance."""
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

        # H_1 has deposits at BOTH banks: 200 at bank_1, 300 at bank_2
        system.mint_cash(to_agent_id="H_1", amount=500)
        deposit_cash(system, "H_1", "bank_1", 200)
        deposit_cash(system, "H_1", "bank_2", 300)

        # H_2 has deposit at bank_1 only (creditor receives at bank_1)
        system.mint_cash(to_agent_id="H_2", amount=400)
        deposit_cash(system, "H_2", "bank_1", 400)

        system.mint_reserves(to_bank_id="bank_1", amount=300)
        system.mint_reserves(to_bank_id="bank_2", amount=300)

        profile = BankProfile()
        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=Decimal("1.0"),
            maturity_days=10,
            trader_banks={"H_1": ["bank_1", "bank_2"], "H_2": ["bank_1"]},
        )

        # H_1 owes H_2 400 — creditor at bank_1
        payable = Payable(
            id="PAY_multi",
            kind=InstrumentKind.PAYABLE,
            amount=400,
            denom="USD",
            asset_holder_id="H_2",
            liability_issuer_id="H_1",
            due_day=0,
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.append(payable.id)
        system.state.agents["H_2"].asset_ids.append(payable.id)

        net = subsystem.compute_settlement_forecasts(system, current_day=0)

        # After initialize_banking_subsystem:
        #   bank_1 r_D ≈ 0.003 (more deposits → higher rate)
        #   bank_2 r_D = 0
        # Creditor H_2 receives at bank_1 (highest r_D).
        # Debtor H_1 pays from lowest r_D first:
        #   bank_2 (r_D=0, balance=300) → 300 cross-bank to bank_1
        #   bank_1 (r_D=0.003, balance=200) → 100 intra-bank (skipped)
        assert net["bank_2"] == 300  # outflow: paid 300 cross-bank to bank_1
        assert net["bank_1"] == -300  # inflow: received 300 from bank_2


class TestProjectedReserveCheck:
    """Tests for the projected reserve check in _bank_can_lend (Plan 045)."""

    def test_bank_refuses_loan_when_settlement_drain_exceeds_reserves(self):
        """Bank refuses to lend when projected reserves (after settlement drain) are below floor."""
        # Low reserves, large settlement outflow expected
        system = _make_settlement_system(reserves_1=300, deposits_1=600)
        profile = BankProfile()

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=Decimal("0.3"),
            maturity_days=10,
            trader_banks={"H_1": ["bank_1"], "H_2": ["bank_2"]},
        )

        # Create a large cross-bank payable due today (H_1 → H_2, different banks)
        # This will drain bank_1's reserves
        payable = Payable(
            id="PAY_drain",
            kind=InstrumentKind.PAYABLE,
            amount=500,
            denom="USD",
            asset_holder_id="H_2",
            liability_issuer_id="H_1",
            due_day=0,
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.append(payable.id)
        system.state.agents["H_2"].asset_ids.append(payable.id)

        # Also create a payable that creates a shortfall for H_1
        payable2 = Payable(
            id="PAY_short",
            kind=InstrumentKind.PAYABLE,
            amount=2000,
            denom="USD",
            asset_holder_id="H_2",
            liability_issuer_id="H_1",
            due_day=2,
        )
        system.state.contracts[payable2.id] = payable2
        system.state.agents["H_1"].liability_ids.append(payable2.id)
        system.state.agents["H_2"].asset_ids.append(payable2.id)

        # Refresh quotes — this now includes settlement forecasts
        subsystem.refresh_all_quotes(system, current_day=0)

        # Verify bank_1 sees the settlement drain
        bank_state = subsystem.banks["bank_1"]
        assert bank_state._settlement_net_outflow == 500

        # min_projected_reserves should be low (reserves=300 minus settlement drain 500 => negative)
        assert bank_state.min_projected_reserves < 0

        # Run lending phase — bank_1 should refuse to lend to H_1
        events = run_bank_lending_phase(system, current_day=0, banking=subsystem)

        loan_events = [e for e in events if e["kind"] == "BankLoanIssued"]
        assert len(loan_events) == 0, (
            f"Bank should refuse lending when settlement drain exceeds reserves, "
            f"but {len(loan_events)} loans issued"
        )

    def test_bank_lends_when_reserves_sufficient_after_settlement(self):
        """Bank lends when it has ample reserves even after settlement drain."""
        # High reserves relative to settlement outflow
        system = _make_settlement_system(reserves_1=5000, deposits_1=1000)
        profile = BankProfile()

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=Decimal("2.0"),
            maturity_days=10,
            trader_banks={"H_1": ["bank_1"], "H_2": ["bank_2"]},
        )

        # Small cross-bank payable
        payable = Payable(
            id="PAY_small",
            kind=InstrumentKind.PAYABLE,
            amount=200,
            denom="USD",
            asset_holder_id="H_2",
            liability_issuer_id="H_1",
            due_day=0,
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.append(payable.id)
        system.state.agents["H_2"].asset_ids.append(payable.id)

        # Create shortfall for H_1
        payable2 = Payable(
            id="PAY_need",
            kind=InstrumentKind.PAYABLE,
            amount=2000,
            denom="USD",
            asset_holder_id="H_2",
            liability_issuer_id="H_1",
            due_day=2,
        )
        system.state.contracts[payable2.id] = payable2
        system.state.agents["H_1"].liability_ids.append(payable2.id)
        system.state.agents["H_2"].asset_ids.append(payable2.id)

        # Refresh quotes
        subsystem.refresh_all_quotes(system, current_day=0)

        bank_state = subsystem.banks["bank_1"]
        # With 5000 reserves and only 200 outflow, min_projected should be high
        assert bank_state.min_projected_reserves > 0

        # Run lending phase — bank_1 should lend to H_1
        events = run_bank_lending_phase(system, current_day=0, banking=subsystem)

        loan_events = [e for e in events if e["kind"] == "BankLoanIssued"]
        assert len(loan_events) > 0, (
            "Bank should lend when reserves are ample after settlement drain"
        )

    def test_settlement_drain_included_in_refresh_quote_path(self):
        """Verify refresh_quote incorporates settlement drain into path[0]."""
        system = _make_settlement_system(reserves_1=1000, deposits_1=600)
        profile = BankProfile()

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=Decimal("1.0"),
            maturity_days=10,
            trader_banks={"H_1": ["bank_1"], "H_2": ["bank_2"]},
        )

        # Add a cross-bank payable of 800
        payable = Payable(
            id="PAY_path",
            kind=InstrumentKind.PAYABLE,
            amount=800,
            denom="USD",
            asset_holder_id="H_2",
            liability_issuer_id="H_1",
            due_day=0,
        )
        system.state.contracts[payable.id] = payable
        system.state.agents["H_1"].liability_ids.append(payable.id)
        system.state.agents["H_2"].asset_ids.append(payable.id)

        # Refresh WITHOUT settlement forecast (manually set to 0)
        bank_state = subsystem.banks["bank_1"]
        bank_state._settlement_net_outflow = 0
        bank_state.refresh_quote(system, current_day=0, n_banks=2)
        min_without = bank_state.min_projected_reserves

        # Now refresh WITH settlement forecast
        subsystem.refresh_all_quotes(system, current_day=0)
        min_with = bank_state.min_projected_reserves

        # With settlement drain, projected reserves should be lower
        assert min_with < min_without, (
            f"Settlement drain should reduce projected reserves: "
            f"without={min_without}, with={min_with}"
        )
