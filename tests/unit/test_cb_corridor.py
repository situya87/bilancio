"""
Tests for Central Bank corridor mechanics.

Tests cover:
1. CBLoan instrument creation and properties
2. Reserve interest accrual
3. CB lending facility
4. CB loan repayment
5. CentralBank corridor parameters
"""

from decimal import Decimal

import pytest

from bilancio.core.errors import ValidationError
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.cb_loan import CBLoan
from bilancio.domain.instruments.means_of_payment import ReserveDeposit
from bilancio.engines.system import System

# =============================================================================
# CBLoan Instrument Tests
# =============================================================================


class TestCBLoanInstrument:
    """Tests for the CBLoan instrument type."""

    def test_cb_loan_properties(self):
        """Test CBLoan computes maturity and repayment correctly."""
        loan = CBLoan(
            id="L1",
            kind=InstrumentKind.CB_LOAN,
            amount=10000,
            denom="X",
            asset_holder_id="CB",
            liability_issuer_id="B1",
            cb_rate=Decimal("0.03"),
            issuance_day=5,
        )

        assert loan.principal == 10000
        assert loan.maturity_day == 7  # 5 + 2
        assert loan.repayment_amount == 10300  # 10000 * 1.03
        assert loan.interest_amount == 300

    def test_cb_loan_is_due(self):
        """Test CBLoan due date checking."""
        loan = CBLoan(
            id="L1",
            kind=InstrumentKind.CB_LOAN,
            amount=5000,
            denom="X",
            asset_holder_id="CB",
            liability_issuer_id="B1",
            cb_rate=Decimal("0.02"),
            issuance_day=0,
        )

        assert not loan.is_due(0)  # Not due on issuance day
        assert not loan.is_due(1)  # Not due day before maturity
        assert loan.is_due(2)  # Due on maturity day
        assert loan.is_due(3)  # Due after maturity day

    def test_cb_loan_different_rates(self):
        """Test CBLoan with different interest rates."""
        # Low rate
        loan_low = CBLoan(
            id="L1",
            kind=InstrumentKind.CB_LOAN,
            amount=1000,
            denom="X",
            asset_holder_id="CB",
            liability_issuer_id="B1",
            cb_rate=Decimal("0.01"),
            issuance_day=0,
        )
        assert loan_low.repayment_amount == 1010

        # High rate
        loan_high = CBLoan(
            id="L2",
            kind=InstrumentKind.CB_LOAN,
            amount=1000,
            denom="X",
            asset_holder_id="CB",
            liability_issuer_id="B2",
            cb_rate=Decimal("0.05"),
            issuance_day=0,
        )
        assert loan_high.repayment_amount == 1050


# =============================================================================
# Reserve Interest Accrual Tests
# =============================================================================


class TestReserveInterest:
    """Tests for reserve deposit interest accrual."""

    def test_reserve_with_interest_properties(self):
        """Test ReserveDeposit with interest tracking fields."""
        reserve = ReserveDeposit(
            id="R1",
            kind=InstrumentKind.RESERVE_DEPOSIT,
            amount=10000,
            denom="X",
            asset_holder_id="B1",
            liability_issuer_id="CB",
            remuneration_rate=Decimal("0.01"),
            issuance_day=0,
        )

        assert reserve.next_interest_day() == 2  # 0 + 2
        assert reserve.compute_interest() == 100  # 10000 * 0.01
        assert not reserve.is_interest_due(0)
        assert not reserve.is_interest_due(1)
        assert reserve.is_interest_due(2)
        assert reserve.is_interest_due(3)

    def test_reserve_without_interest(self):
        """Test ReserveDeposit without interest (backwards compatible)."""
        reserve = ReserveDeposit(
            id="R1",
            kind=InstrumentKind.RESERVE_DEPOSIT,
            amount=10000,
            denom="X",
            asset_holder_id="B1",
            liability_issuer_id="CB",
            # No remuneration_rate
        )

        assert reserve.next_interest_day() is None
        assert reserve.compute_interest() == 0
        assert not reserve.is_interest_due(100)

    def test_reserve_interest_day_tracking(self):
        """Test that last_interest_day updates affect next_interest_day."""
        reserve = ReserveDeposit(
            id="R1",
            kind=InstrumentKind.RESERVE_DEPOSIT,
            amount=5000,
            denom="X",
            asset_holder_id="B1",
            liability_issuer_id="CB",
            remuneration_rate=Decimal("0.02"),
            issuance_day=0,
        )

        # Initially: next interest at day 2
        assert reserve.next_interest_day() == 2

        # After crediting interest on day 2
        reserve.last_interest_day = 2
        assert reserve.next_interest_day() == 4

        # After crediting interest on day 4
        reserve.last_interest_day = 4
        assert reserve.next_interest_day() == 6


# =============================================================================
# CentralBank Agent Tests
# =============================================================================


class TestCentralBankCorridor:
    """Tests for CentralBank corridor parameters."""

    def test_default_corridor_rates(self):
        """Test default corridor rates."""
        cb = CentralBank(id="CB", name="Central Bank")

        assert cb.reserve_remuneration_rate == Decimal("0.01")
        assert cb.cb_lending_rate == Decimal("0.03")
        assert cb.corridor_width == Decimal("0.02")
        assert cb.corridor_mid == Decimal("0.02")

    def test_custom_corridor_rates(self):
        """Test custom corridor rates."""
        cb = CentralBank(
            id="CB",
            name="Central Bank",
            reserve_remuneration_rate=Decimal("0.005"),
            cb_lending_rate=Decimal("0.025"),
        )

        assert cb.corridor_width == Decimal("0.02")
        assert cb.corridor_mid == Decimal("0.015")

    def test_corridor_validation(self):
        """Test corridor validation."""
        cb = CentralBank(
            id="CB",
            name="Central Bank",
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
        )
        cb.validate_corridor()  # Should not raise

        # Invalid: ceiling below floor
        cb_invalid = CentralBank(
            id="CB2",
            name="Invalid CB",
            reserve_remuneration_rate=Decimal("0.05"),
            cb_lending_rate=Decimal("0.02"),
        )
        with pytest.raises(AssertionError):
            cb_invalid.validate_corridor()

    def test_cb_configuration_flags(self):
        """Test CB configuration flags."""
        # Default: both enabled
        cb_default = CentralBank(id="CB", name="CB")
        assert cb_default.issues_cash
        assert cb_default.reserves_accrue_interest

        # Disabled
        cb_custom = CentralBank(
            id="CB",
            name="CB",
            issues_cash=False,
            reserves_accrue_interest=False,
        )
        assert not cb_custom.issues_cash
        assert not cb_custom.reserves_accrue_interest


# =============================================================================
# CB Lending Facility Tests
# =============================================================================


class TestCBLendingFacility:
    """Tests for CB lending operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sys = System()
        self.cb = CentralBank(
            id="CB",
            name="Central Bank",
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
        )
        self.bank = Bank(id="B1", name="Bank 1", kind="bank")
        self.sys.add_agent(self.cb)
        self.sys.add_agent(self.bank)

    def test_cb_lend_reserves_creates_loan_and_reserves(self):
        """Test CB lending creates both loan and reserves."""
        loan_id = self.sys.cb_lend_reserves("B1", 10000, day=0)

        # Check loan created
        assert loan_id in self.sys.state.contracts
        loan = self.sys.state.contracts[loan_id]
        assert loan.kind == InstrumentKind.CB_LOAN
        assert loan.amount == 10000
        assert loan.asset_holder_id == "CB"
        assert loan.liability_issuer_id == "B1"
        assert loan.cb_rate == Decimal("0.03")

        # Check reserves created
        bank_reserves = sum(
            c.amount
            for c in self.sys.state.contracts.values()
            if c.kind == InstrumentKind.RESERVE_DEPOSIT and c.asset_holder_id == "B1"
        )
        assert bank_reserves == 10000

        # Check outstanding counters
        assert self.sys.state.cb_loans_outstanding == 10000
        assert self.sys.state.cb_reserves_outstanding == 10000

        # Check invariants
        self.sys.assert_invariants()

    def test_cb_lend_reserves_with_interest_tracking(self):
        """Test CB lending creates reserves with interest tracking."""
        self.sys.cb_lend_reserves("B1", 5000, day=3)

        # Find the reserve
        reserve = next(
            c
            for c in self.sys.state.contracts.values()
            if c.kind == InstrumentKind.RESERVE_DEPOSIT and c.asset_holder_id == "B1"
        )

        assert reserve.remuneration_rate == Decimal("0.01")
        assert reserve.issuance_day == 3
        assert reserve.next_interest_day() == 5

    def test_multiple_cb_loans(self):
        """Test multiple CB loans to same bank."""
        loan1_id = self.sys.cb_lend_reserves("B1", 5000, day=0)
        loan2_id = self.sys.cb_lend_reserves("B1", 3000, day=1)

        assert loan1_id != loan2_id
        assert self.sys.state.cb_loans_outstanding == 8000
        assert self.sys.state.cb_reserves_outstanding == 8000

        # Check both loans exist
        loan1 = self.sys.state.contracts[loan1_id]
        loan2 = self.sys.state.contracts[loan2_id]
        assert loan1.maturity_day == 2
        assert loan2.maturity_day == 3

        self.sys.assert_invariants()


# =============================================================================
# CB Loan Repayment Tests
# =============================================================================


class TestCBLoanRepayment:
    """Tests for CB loan repayment."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sys = System()
        self.cb = CentralBank(
            id="CB",
            name="Central Bank",
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
        )
        self.bank = Bank(id="B1", name="Bank 1", kind="bank")
        self.sys.add_agent(self.cb)
        self.sys.add_agent(self.bank)

    def test_repay_cb_loan_success(self):
        """Test successful CB loan repayment."""
        # Create loan
        loan_id = self.sys.cb_lend_reserves("B1", 10000, day=0)

        # Bank now has 10000 reserves
        # Add extra for interest (300)
        self.sys.mint_reserves("B1", 400)

        # Repay loan
        repaid = self.sys.cb_repay_loan(loan_id, "B1")

        assert repaid == 10300  # principal + interest
        assert self.sys.state.cb_loans_outstanding == 0

        # Bank should have remaining reserves
        bank_reserves = sum(
            c.amount
            for c in self.sys.state.contracts.values()
            if c.kind == InstrumentKind.RESERVE_DEPOSIT and c.asset_holder_id == "B1"
        )
        assert bank_reserves == 100  # 10400 - 10300

        # Loan should be cancelled
        assert loan_id not in self.sys.state.contracts

        self.sys.assert_invariants()

    def test_repay_cb_loan_insufficient_reserves(self):
        """Test CB loan repayment fails with insufficient reserves."""
        # Create loan but don't give bank enough to repay
        loan_id = self.sys.cb_lend_reserves("B1", 10000, day=0)
        # Bank has 10000 but needs 10300

        with pytest.raises(ValidationError, match="Insufficient reserves"):
            self.sys.cb_repay_loan(loan_id, "B1")

        # State should be unchanged (atomic rollback)
        assert loan_id in self.sys.state.contracts
        assert self.sys.state.cb_loans_outstanding == 10000

    def test_repay_cb_loan_wrong_bank(self):
        """Test CB loan repayment fails for wrong bank."""
        bank2 = Bank(id="B2", name="Bank 2", kind="bank")
        self.sys.add_agent(bank2)

        loan_id = self.sys.cb_lend_reserves("B1", 5000, day=0)

        with pytest.raises(ValidationError, match="not the borrower"):
            self.sys.cb_repay_loan(loan_id, "B2")

    def test_get_cb_loans_due(self):
        """Test getting loans due on a specific day."""
        loan1 = self.sys.cb_lend_reserves("B1", 5000, day=0)  # Due day 2
        loan2 = self.sys.cb_lend_reserves("B1", 3000, day=1)  # Due day 3
        loan3 = self.sys.cb_lend_reserves("B1", 2000, day=2)  # Due day 4

        assert self.sys.get_cb_loans_due(1) == []
        assert set(self.sys.get_cb_loans_due(2)) == {loan1}
        assert set(self.sys.get_cb_loans_due(3)) == {loan1, loan2}
        assert set(self.sys.get_cb_loans_due(4)) == {loan1, loan2, loan3}


# =============================================================================
# Reserve Interest Crediting Tests
# =============================================================================


class TestReserveInterestCrediting:
    """Tests for system-wide reserve interest crediting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sys = System()
        self.cb = CentralBank(
            id="CB",
            name="Central Bank",
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
            reserves_accrue_interest=True,
        )
        self.bank1 = Bank(id="B1", name="Bank 1", kind="bank")
        self.bank2 = Bank(id="B2", name="Bank 2", kind="bank")
        self.sys.add_agent(self.cb)
        self.sys.add_agent(self.bank1)
        self.sys.add_agent(self.bank2)

    def test_credit_reserve_interest_single_bank(self):
        """Test interest crediting for a single bank."""
        # Mint reserves with interest on day 0
        self.sys.mint_reserves_with_interest("B1", 10000, day=0)

        # Credit interest on day 2
        total_interest = self.sys.credit_reserve_interest(day=2)

        assert total_interest == 100  # 10000 * 0.01

        # Bank should have original + interest
        bank_reserves = sum(
            c.amount
            for c in self.sys.state.contracts.values()
            if c.kind == InstrumentKind.RESERVE_DEPOSIT and c.asset_holder_id == "B1"
        )
        assert bank_reserves == 10100

        self.sys.assert_invariants()

    def test_credit_reserve_interest_multiple_banks(self):
        """Test interest crediting for multiple banks."""
        self.sys.mint_reserves_with_interest("B1", 10000, day=0)
        self.sys.mint_reserves_with_interest("B2", 5000, day=0)

        total_interest = self.sys.credit_reserve_interest(day=2)

        assert total_interest == 150  # 100 + 50

        b1_reserves = sum(
            c.amount
            for c in self.sys.state.contracts.values()
            if c.kind == InstrumentKind.RESERVE_DEPOSIT and c.asset_holder_id == "B1"
        )
        b2_reserves = sum(
            c.amount
            for c in self.sys.state.contracts.values()
            if c.kind == InstrumentKind.RESERVE_DEPOSIT and c.asset_holder_id == "B2"
        )

        assert b1_reserves == 10100
        assert b2_reserves == 5050

        self.sys.assert_invariants()

    def test_credit_reserve_interest_not_due_yet(self):
        """Test no interest credited before due date."""
        self.sys.mint_reserves_with_interest("B1", 10000, day=0)

        # Day 1: not due yet
        total_interest = self.sys.credit_reserve_interest(day=1)

        assert total_interest == 0

        bank_reserves = sum(
            c.amount
            for c in self.sys.state.contracts.values()
            if c.kind == InstrumentKind.RESERVE_DEPOSIT and c.asset_holder_id == "B1"
        )
        assert bank_reserves == 10000

    def test_credit_reserve_interest_multiple_periods(self):
        """Test interest crediting over multiple periods."""
        self.sys.mint_reserves_with_interest("B1", 10000, day=0)

        # Day 2: first interest
        interest_d2 = self.sys.credit_reserve_interest(day=2)
        assert interest_d2 == 100

        # Day 4: second interest (on original only, new interest not compounded in same period)
        interest_d4 = self.sys.credit_reserve_interest(day=4)
        # Original reserve: 10000 * 0.01 = 100
        # Interest reserve from day 2: 100 * 0.01 = 1 (if it was created with interest tracking)
        assert interest_d4 >= 100  # At least 100 from original

        self.sys.assert_invariants()

    def test_credit_reserve_interest_disabled(self):
        """Test no interest when CB has it disabled."""
        # Create CB with interest disabled
        sys2 = System()
        cb_no_interest = CentralBank(
            id="CB",
            name="CB",
            reserves_accrue_interest=False,
        )
        bank = Bank(id="B1", name="Bank 1", kind="bank")
        sys2.add_agent(cb_no_interest)
        sys2.add_agent(bank)

        sys2.mint_reserves_with_interest("B1", 10000, day=0)

        total_interest = sys2.credit_reserve_interest(day=2)

        assert total_interest == 0

    def test_old_style_reserves_no_interest(self):
        """Test old-style reserves without interest fields don't accrue."""
        # Use regular mint_reserves (not with_interest)
        self.sys.mint_reserves("B1", 10000)

        total_interest = self.sys.credit_reserve_interest(day=2)

        # Old-style reserves don't have remuneration_rate, so no interest
        assert total_interest == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestCBCorridorIntegration:
    """Integration tests for the full CB corridor workflow."""

    def test_full_lending_cycle(self):
        """Test complete lending cycle: borrow -> earn interest -> repay."""
        sys = System()
        cb = CentralBank(
            id="CB",
            name="Central Bank",
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
        )
        bank = Bank(id="B1", name="Bank 1", kind="bank")
        sys.add_agent(cb)
        sys.add_agent(bank)

        # Day 0: Bank borrows 10,000
        loan_id = sys.cb_lend_reserves("B1", 10000, day=0)

        assert sys.state.cb_reserves_outstanding == 10000
        assert sys.state.cb_loans_outstanding == 10000

        # Day 2: Interest credited on reserves
        sys.state.day = 2
        interest = sys.credit_reserve_interest(day=2)

        # Bank earned 100 on 10,000 at 1%
        assert interest == 100
        assert sys.state.cb_reserves_outstanding == 10100

        # Day 2: Loan is due, repay it
        # Bank has 10,100 reserves, needs 10,300 to repay
        # So bank still needs to borrow more or we need to mint extra
        sys.mint_reserves("B1", 200)  # Give extra to cover interest

        repaid = sys.cb_repay_loan(loan_id, "B1")

        assert repaid == 10300
        assert sys.state.cb_loans_outstanding == 0

        # Bank should have: 10100 + 200 - 10300 = 0
        bank_reserves = sum(
            c.amount
            for c in sys.state.contracts.values()
            if c.kind == InstrumentKind.RESERVE_DEPOSIT and c.asset_holder_id == "B1"
        )
        assert bank_reserves == 0

        sys.assert_invariants()

    def test_two_banks_interbank_with_cb_corridor(self):
        """Test two banks with CB corridor: one borrows, one has excess."""
        sys = System()
        cb = CentralBank(
            id="CB",
            name="Central Bank",
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
        )
        bank1 = Bank(id="B1", name="Bank 1", kind="bank")
        bank2 = Bank(id="B2", name="Bank 2", kind="bank")
        sys.add_agent(cb)
        sys.add_agent(bank1)
        sys.add_agent(bank2)

        # Bank1 has excess reserves (earning interest)
        sys.mint_reserves_with_interest("B1", 20000, day=0)

        # Bank2 needs to borrow
        sys.cb_lend_reserves("B2", 10000, day=0)

        # Day 2: Both earn/owe
        interest = sys.credit_reserve_interest(day=2)

        # B1: 20000 * 0.01 = 200
        # B2: 10000 * 0.01 = 100
        assert interest == 300

        # Bank2's loan cost: 10000 * 0.03 = 300
        # Bank2 earns 100 on reserves, pays 300 on loan -> net cost 200

        # Check B1 earning at floor
        b1_reserves = sum(
            c.amount
            for c in sys.state.contracts.values()
            if c.kind == InstrumentKind.RESERVE_DEPOSIT and c.asset_holder_id == "B1"
        )
        assert b1_reserves == 20200  # 20000 + 200

        sys.assert_invariants()

    def test_corridor_spread_matters(self):
        """Test that corridor spread creates cost for borrowers vs savers."""
        sys = System()
        cb = CentralBank(
            id="CB",
            name="Central Bank",
            reserve_remuneration_rate=Decimal("0.01"),  # Floor: 1%
            cb_lending_rate=Decimal("0.05"),  # Ceiling: 5% (wide spread)
        )
        bank = Bank(id="B1", name="Bank 1", kind="bank")
        sys.add_agent(cb)
        sys.add_agent(bank)

        # Borrow 10,000 at 5%
        loan_id = sys.cb_lend_reserves("B1", 10000, day=0)

        # Loan costs 500 interest
        loan = sys.state.contracts[loan_id]
        assert loan.interest_amount == 500

        # Reserves earn 100 interest
        sys.state.day = 2
        interest = sys.credit_reserve_interest(day=2)
        assert interest == 100

        # Net cost to bank: 500 - 100 = 400
        # This is the corridor spread penalty for borrowing


# =============================================================================
# Dynamic Corridor Tests (Plan 041)
# =============================================================================


class TestDynamicCorridor:
    """Tests for compute_corridor with deviation-from-expectation."""

    def test_disabled_when_kappa_prior_zero(self):
        """compute_corridor returns static rates when kappa_prior=0."""
        cb = CentralBank(
            id="CB",
            name="CB",
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
            kappa_prior=Decimal("0"),
        )
        r_floor, r_ceiling = cb.compute_corridor(10, 100)
        assert r_floor == Decimal("0.01")
        assert r_ceiling == Decimal("0.03")

    def test_no_surprise_no_change(self):
        """When realized defaults <= P_0, surprise=0, corridor unchanged."""
        cb = CentralBank(
            id="CB",
            name="CB",
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
            kappa_prior=Decimal("0.15"),
            beta_mid=Decimal("0.50"),
            beta_width=Decimal("0.30"),
        )
        # 10 out of 100 = 0.10 < 0.15 = P_0 → surprise = 0
        r_floor, r_ceiling = cb.compute_corridor(10, 100)
        assert r_floor == Decimal("0.01")
        assert r_ceiling == Decimal("0.03")

    def test_surprise_widens_corridor(self):
        """When realized defaults > P_0, corridor mid rises and widens."""
        cb = CentralBank(
            id="CB",
            name="CB",
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
            kappa_prior=Decimal("0.10"),
            beta_mid=Decimal("0.50"),
            beta_width=Decimal("0.30"),
        )
        # mid_0 = 0.02, width_0 = 0.02
        # 30 of 100 = 0.30, surprise = 0.30 - 0.10 = 0.20
        # mid = 0.02 + 0.50 * 0.20 = 0.12
        # width = 0.02 + 0.30 * 0.20 = 0.08
        # r_floor = 0.12 - 0.04 = 0.08
        # r_ceiling = 0.12 + 0.04 = 0.16
        r_floor, r_ceiling = cb.compute_corridor(30, 100)
        assert r_floor == Decimal("0.08")
        assert r_ceiling == Decimal("0.16")

    def test_floor_clamped_at_zero(self):
        """Floor rate never goes negative."""
        cb = CentralBank(
            id="CB",
            name="CB",
            reserve_remuneration_rate=Decimal("0.005"),
            cb_lending_rate=Decimal("0.015"),
            kappa_prior=Decimal("0.10"),
            beta_mid=Decimal("0.01"),
            beta_width=Decimal("5.0"),  # extreme width sensitivity
        )
        # mid_0 = 0.01, width_0 = 0.01
        # surprise = 0.40 - 0.10 = 0.30
        # mid = 0.01 + 0.01*0.30 = 0.013
        # width = 0.01 + 5.0*0.30 = 1.51
        # r_floor = max(0, 0.013 - 0.755) = 0
        r_floor, r_ceiling = cb.compute_corridor(40, 100)
        assert r_floor == Decimal("0")
        assert r_ceiling > Decimal("0")

    def test_zero_total_agents_no_crash(self):
        """No division by zero when n_total=0."""
        cb = CentralBank(
            id="CB",
            name="CB",
            kappa_prior=Decimal("0.15"),
        )
        r_floor, r_ceiling = cb.compute_corridor(0, 0)
        # p_realized = 0/1 = 0, surprise = max(0, 0-0.15) = 0
        assert r_floor == cb.reserve_remuneration_rate
        assert r_ceiling == cb.cb_lending_rate
