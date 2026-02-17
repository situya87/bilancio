"""Unit tests for interbank lending and repayment.

Tests cover:
1. run_interbank_lending - surplus banks lend reserves to deficit banks
2. run_interbank_repayments - repayment of interbank loans at maturity
"""

from decimal import Decimal

from bilancio.banking.types import Quote
from bilancio.decision.profiles import BankProfile
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.banking_subsystem import (
    BankingSubsystem,
    BankTreynorState,
    InterbankLoan,
    _get_bank_reserves,
    initialize_banking_subsystem,
)
from bilancio.engines.interbank import run_interbank_lending, run_interbank_repayments
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_banking_system() -> System:
    """Create a minimal system with CB + 2 banks + 2 firms for testing."""
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
# TestRunInterbankLending
# ---------------------------------------------------------------------------


class TestRunInterbankLending:
    """Tests for run_interbank_lending."""

    def test_no_lending_when_balanced(self):
        """Both banks at target reserves -- no interbank loans should occur."""
        system = _make_banking_system()
        profile = BankProfile()
        kappa = Decimal("1.0")

        banking = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        # Both banks start with identical reserves and identical targets,
        # so neither has a surplus or deficit.
        events = run_interbank_lending(system, current_day=0, banking=banking)

        assert events == []
        assert banking.interbank_loans == []

    def test_surplus_lends_to_deficit(self):
        """One bank has surplus reserves, the other a deficit -- lending should occur."""
        system = _make_banking_system()
        profile = BankProfile()
        kappa = Decimal("1.0")

        banking = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        # Create asymmetry: move 3000 reserves from bank_2 to bank_1
        # so bank_1 has surplus and bank_2 has a deficit.
        system.transfer_reserves("bank_2", "bank_1", 3000)

        # Manually set reserve targets to amplify the imbalance.
        # bank_1 target low (so it has a large surplus),
        # bank_2 target high (so it has a large deficit).
        banking.banks["bank_1"].pricing_params.reserve_target = 2000
        banking.banks["bank_2"].pricing_params.reserve_target = 5000

        # Set quotes so that the surplus bank (bank_1) has a lower deposit rate
        # and the deficit bank (bank_2) has a higher midline -- trade should clear.
        banking.banks["bank_1"].current_quote = Quote(
            deposit_rate=Decimal("0.02"),
            loan_rate=Decimal("0.08"),
            day=0,
            midline=Decimal("0.05"),
        )
        banking.banks["bank_2"].current_quote = Quote(
            deposit_rate=Decimal("0.04"),
            loan_rate=Decimal("0.10"),
            day=0,
            midline=Decimal("0.07"),
        )

        reserves_1_before = _get_bank_reserves(system, "bank_1")
        reserves_2_before = _get_bank_reserves(system, "bank_2")

        events = run_interbank_lending(system, current_day=0, banking=banking)

        # Lending should have occurred
        assert len(events) >= 1
        assert len(banking.interbank_loans) >= 1

        # The loan should be from bank_1 (surplus) to bank_2 (deficit)
        loan = banking.interbank_loans[0]
        assert loan.lender_bank == "bank_1"
        assert loan.borrower_bank == "bank_2"
        assert loan.amount > 0
        assert loan.maturity_day == 2  # current_day (0) + 2

        # Reserves should have moved
        reserves_1_after = _get_bank_reserves(system, "bank_1")
        reserves_2_after = _get_bank_reserves(system, "bank_2")
        assert reserves_1_after < reserves_1_before
        assert reserves_2_after > reserves_2_before
        assert reserves_1_after == reserves_1_before - loan.amount
        assert reserves_2_after == reserves_2_before + loan.amount

    def test_no_lending_when_borrower_mid_below_lender_rate(self):
        """When borrower_mid <= lender deposit rate, no trade should occur."""
        system = _make_banking_system()
        profile = BankProfile()
        kappa = Decimal("1.0")

        banking = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        # Create asymmetry so that there IS a surplus/deficit pair
        system.transfer_reserves("bank_2", "bank_1", 3000)
        banking.banks["bank_1"].pricing_params.reserve_target = 2000
        banking.banks["bank_2"].pricing_params.reserve_target = 5000

        # Set quotes so the surplus bank's deposit rate is HIGHER than
        # the deficit bank's midline -- trade should NOT clear.
        banking.banks["bank_1"].current_quote = Quote(
            deposit_rate=Decimal("0.10"),
            loan_rate=Decimal("0.15"),
            day=0,
            midline=Decimal("0.12"),
        )
        banking.banks["bank_2"].current_quote = Quote(
            deposit_rate=Decimal("0.01"),
            loan_rate=Decimal("0.05"),
            day=0,
            midline=Decimal("0.01"),  # borrower midline below lender deposit rate
        )

        events = run_interbank_lending(system, current_day=0, banking=banking)

        assert events == []
        assert banking.interbank_loans == []

    def test_interbank_rate_negotiation(self):
        """Interbank rate should be (borrower_mid + lender_deposit_rate) / 2."""
        system = _make_banking_system()
        profile = BankProfile()
        kappa = Decimal("1.0")

        banking = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        # Create asymmetry
        system.transfer_reserves("bank_2", "bank_1", 3000)
        banking.banks["bank_1"].pricing_params.reserve_target = 2000
        banking.banks["bank_2"].pricing_params.reserve_target = 5000

        borrower_mid = Decimal("0.08")
        lender_deposit_rate = Decimal("0.02")
        expected_rate = (borrower_mid + lender_deposit_rate) / 2  # 0.05

        banking.banks["bank_1"].current_quote = Quote(
            deposit_rate=lender_deposit_rate,
            loan_rate=Decimal("0.10"),
            day=0,
            midline=Decimal("0.06"),
        )
        banking.banks["bank_2"].current_quote = Quote(
            deposit_rate=Decimal("0.04"),
            loan_rate=Decimal("0.12"),
            day=0,
            midline=borrower_mid,
        )

        events = run_interbank_lending(system, current_day=0, banking=banking)

        assert len(events) >= 1
        loan = banking.interbank_loans[0]
        assert loan.rate == expected_rate
        assert events[0]["rate"] == str(expected_rate)


# ---------------------------------------------------------------------------
# TestRunInterbankRepayments
# ---------------------------------------------------------------------------


class TestRunInterbankRepayments:
    """Tests for run_interbank_repayments."""

    def test_repayment_on_maturity(self):
        """Loan maturing today should be repaid and removed from the list."""
        system = _make_banking_system()
        profile = BankProfile()
        kappa = Decimal("1.0")

        banking = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        # Insert a loan: bank_1 lent 100 to bank_2, matures on day 2
        loan = InterbankLoan(
            lender_bank="bank_1",
            borrower_bank="bank_2",
            amount=100,
            rate=Decimal("0.05"),
            issuance_day=0,
            maturity_day=2,
        )
        banking.interbank_loans.append(loan)

        reserves_1_before = _get_bank_reserves(system, "bank_1")
        reserves_2_before = _get_bank_reserves(system, "bank_2")

        # Repayment amount = int(100 * (1 + 0.05)) = 105
        expected_repayment = loan.repayment_amount
        assert expected_repayment == 105

        events = run_interbank_repayments(system, current_day=2, banking=banking)

        # Loan should be removed
        assert len(banking.interbank_loans) == 0

        # Reserves should have transferred from borrower to lender
        reserves_1_after = _get_bank_reserves(system, "bank_1")
        reserves_2_after = _get_bank_reserves(system, "bank_2")
        assert reserves_1_after == reserves_1_before + expected_repayment
        assert reserves_2_after == reserves_2_before - expected_repayment

        # One event should have been emitted
        assert len(events) == 1

    def test_no_repayment_before_maturity(self):
        """Loan maturing on day 5 should not be repaid on day 3."""
        system = _make_banking_system()
        profile = BankProfile()
        kappa = Decimal("1.0")

        banking = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        loan = InterbankLoan(
            lender_bank="bank_1",
            borrower_bank="bank_2",
            amount=100,
            rate=Decimal("0.05"),
            issuance_day=0,
            maturity_day=5,
        )
        banking.interbank_loans.append(loan)

        reserves_1_before = _get_bank_reserves(system, "bank_1")
        reserves_2_before = _get_bank_reserves(system, "bank_2")

        events = run_interbank_repayments(system, current_day=3, banking=banking)

        # No events, loan still present, reserves unchanged
        assert events == []
        assert len(banking.interbank_loans) == 1
        assert banking.interbank_loans[0] is loan
        assert _get_bank_reserves(system, "bank_1") == reserves_1_before
        assert _get_bank_reserves(system, "bank_2") == reserves_2_before

    def test_repayment_event_fields(self):
        """Repayment event dict should have correct fields and values."""
        system = _make_banking_system()
        profile = BankProfile()
        kappa = Decimal("1.0")

        banking = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        loan = InterbankLoan(
            lender_bank="bank_1",
            borrower_bank="bank_2",
            amount=100,
            rate=Decimal("0.05"),
            issuance_day=0,
            maturity_day=2,
        )
        banking.interbank_loans.append(loan)

        events = run_interbank_repayments(system, current_day=2, banking=banking)

        assert len(events) == 1
        event = events[0]

        assert event["kind"] == "InterbankRepaid"
        assert event["day"] == 2
        assert event["lender"] == "bank_1"
        assert event["borrower"] == "bank_2"
        assert event["principal"] == 100
        assert event["repayment"] == 105  # int(100 * 1.05)
        assert event["interest"] == 5    # 105 - 100
