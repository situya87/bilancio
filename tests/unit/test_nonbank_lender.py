"""Comprehensive tests for the Non-Bank Lender system.

Covers:
1. NonBankLoan instrument
2. NonBankLender agent
3. PolicyEngine rules for NonBankLoan
4. System methods (nonbank_lend_cash, nonbank_repay_loan, get_nonbank_loans_due)
5. LendingConfig defaults and custom values
6. Lending strategy (run_lending_phase)
7. Simulation integration (run_day with enable_lender)
8. Scenario generation (compile_ring_explorer_balanced with mode="lender")
"""

from decimal import Decimal

import pytest

from bilancio.config.models import LenderScenarioConfig, ScenarioConfig
from bilancio.core.errors import ValidationError
from bilancio.domain.agent import Agent, AgentKind
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.agents.household import Household
from bilancio.domain.agents.non_bank_lender import NonBankLender
from bilancio.domain.instruments.base import Instrument, InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.non_bank_loan import NonBankLoan
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.lending import LendingConfig, run_lending_phase, run_loan_repayments
from bilancio.engines.system import System


# ── Helpers ──────────────────────────────────────────────────────────────


def _build_lending_system(
    lender_cash: int = 10000,
    firm_cash: int = 500,
    firm_payable_amount: int = 1000,
    payable_due_day: int = 2,
) -> System:
    """Build a minimal system suitable for lending tests.

    Agents:
        CB01  - Central bank
        B01   - Bank (required for CB but otherwise inactive)
        NBL01 - Non-bank lender with ``lender_cash``
        F01   - Firm with ``firm_cash`` and a payable of ``firm_payable_amount``
        F02   - Firm acting as creditor of F01's payable

    Returns a ``System`` in day 0 with all instruments in place.
    """
    system = System()

    cb = CentralBank(id="CB01", name="Central Bank", kind="central_bank")
    bank = Bank(id="B01", name="Bank 1", kind="bank")
    lender = NonBankLender(id="NBL01", name="Non-Bank Lender")
    firm1 = Firm(id="F01", name="Firm 1", kind="firm")
    firm2 = Firm(id="F02", name="Firm 2", kind="firm")

    system.bootstrap_cb(cb)
    system.add_agent(bank)
    system.add_agent(lender)
    system.add_agent(firm1)
    system.add_agent(firm2)

    # Mint cash
    if lender_cash > 0:
        system.mint_cash("NBL01", lender_cash)
    if firm_cash > 0:
        system.mint_cash("F01", firm_cash)

    # Create a payable: F01 owes F02
    if firm_payable_amount > 0:
        payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=firm_payable_amount,
            denom="X",
            asset_holder_id="F02",
            liability_issuer_id="F01",
            due_day=payable_due_day,
        )
        system.add_contract(payable)

    return system


# ═══════════════════════════════════════════════════════════════════════
# 1. NonBankLoan Instrument Tests
# ═══════════════════════════════════════════════════════════════════════


class TestNonBankLoan:
    """Tests for the NonBankLoan instrument dataclass."""

    def test_creation_defaults(self):
        """NonBankLoan created with default rate, issuance_day, maturity_days."""
        loan = NonBankLoan(
            id="NBL_1",
            kind=InstrumentKind.NON_BANK_LOAN,
            amount=1000,
            denom="X",
            asset_holder_id="lender",
            liability_issuer_id="borrower",
        )
        assert loan.rate == Decimal("0.05")
        assert loan.issuance_day == 0
        assert loan.maturity_days == 2

    def test_kind_is_non_bank_loan(self):
        """__post_init__ forces kind to NON_BANK_LOAN regardless of init value."""
        loan = NonBankLoan(
            id="NBL_2",
            kind=InstrumentKind.CASH,  # deliberate wrong kind
            amount=500,
            denom="X",
            asset_holder_id="lender",
            liability_issuer_id="borrower",
        )
        assert loan.kind == InstrumentKind.NON_BANK_LOAN

    def test_maturity_day_property(self):
        """maturity_day = issuance_day + maturity_days."""
        loan = NonBankLoan(
            id="NBL_3",
            kind=InstrumentKind.NON_BANK_LOAN,
            amount=1000,
            denom="X",
            asset_holder_id="lender",
            liability_issuer_id="borrower",
            issuance_day=3,
            maturity_days=5,
        )
        assert loan.maturity_day == 8

    def test_repayment_amount_property(self):
        """repayment_amount = int(amount * (1 + rate))."""
        loan = NonBankLoan(
            id="NBL_4",
            kind=InstrumentKind.NON_BANK_LOAN,
            amount=1000,
            denom="X",
            asset_holder_id="lender",
            liability_issuer_id="borrower",
            rate=Decimal("0.10"),
        )
        assert loan.repayment_amount == 1100

    def test_repayment_amount_truncates(self):
        """repayment_amount truncates to int (floor via int())."""
        loan = NonBankLoan(
            id="NBL_5",
            kind=InstrumentKind.NON_BANK_LOAN,
            amount=999,
            denom="X",
            asset_holder_id="lender",
            liability_issuer_id="borrower",
            rate=Decimal("0.05"),
        )
        # 999 * 1.05 = 1048.95 -> int(1048.95) = 1048
        assert loan.repayment_amount == 1048

    def test_interest_amount_property(self):
        """interest_amount = repayment_amount - amount."""
        loan = NonBankLoan(
            id="NBL_6",
            kind=InstrumentKind.NON_BANK_LOAN,
            amount=2000,
            denom="X",
            asset_holder_id="lender",
            liability_issuer_id="borrower",
            rate=Decimal("0.05"),
        )
        assert loan.interest_amount == loan.repayment_amount - 2000
        assert loan.interest_amount == 100  # 2000 * 0.05 = 100

    def test_principal_alias(self):
        """principal is an alias for amount."""
        loan = NonBankLoan(
            id="NBL_7",
            kind=InstrumentKind.NON_BANK_LOAN,
            amount=3000,
            denom="X",
            asset_holder_id="lender",
            liability_issuer_id="borrower",
        )
        assert loan.principal == 3000
        assert loan.principal == loan.amount

    def test_is_due_true_on_maturity_day(self):
        """is_due returns True when current_day >= maturity_day."""
        loan = NonBankLoan(
            id="NBL_8",
            kind=InstrumentKind.NON_BANK_LOAN,
            amount=1000,
            denom="X",
            asset_holder_id="lender",
            liability_issuer_id="borrower",
            issuance_day=1,
            maturity_days=3,
        )
        # maturity_day = 4
        assert not loan.is_due(3)
        assert loan.is_due(4)
        assert loan.is_due(5)

    def test_is_due_false_before_maturity(self):
        """is_due returns False when current_day < maturity_day."""
        loan = NonBankLoan(
            id="NBL_9",
            kind=InstrumentKind.NON_BANK_LOAN,
            amount=1000,
            denom="X",
            asset_holder_id="lender",
            liability_issuer_id="borrower",
            issuance_day=0,
            maturity_days=2,
        )
        assert not loan.is_due(0)
        assert not loan.is_due(1)
        assert loan.is_due(2)

    def test_custom_rate(self):
        """Loan with a custom rate computes interest correctly."""
        loan = NonBankLoan(
            id="NBL_10",
            kind=InstrumentKind.NON_BANK_LOAN,
            amount=5000,
            denom="X",
            asset_holder_id="lender",
            liability_issuer_id="borrower",
            rate=Decimal("0.20"),
        )
        assert loan.repayment_amount == 6000
        assert loan.interest_amount == 1000


# ═══════════════════════════════════════════════════════════════════════
# 2. NonBankLender Agent Tests
# ═══════════════════════════════════════════════════════════════════════


class TestNonBankLenderAgent:
    """Tests for the NonBankLender agent class."""

    def test_creation(self):
        """NonBankLender can be created with id and name."""
        lender = NonBankLender(id="NBL01", name="My Lender")
        assert lender.id == "NBL01"
        assert lender.name == "My Lender"

    def test_kind_defaults_to_non_bank_lender(self):
        """kind field defaults to AgentKind.NON_BANK_LENDER."""
        lender = NonBankLender(id="NBL02", name="Lender 2")
        assert lender.kind == AgentKind.NON_BANK_LENDER
        assert lender.kind == "non_bank_lender"

    def test_inherits_from_agent(self):
        """NonBankLender is a subclass of Agent."""
        lender = NonBankLender(id="NBL03", name="Lender 3")
        assert isinstance(lender, Agent)

    def test_starts_with_empty_ids(self):
        """Newly created lender has no assets or liabilities."""
        lender = NonBankLender(id="NBL04", name="Lender 4")
        assert lender.asset_ids == []
        assert lender.liability_ids == []
        assert lender.stock_ids == []
        assert lender.defaulted is False


# ═══════════════════════════════════════════════════════════════════════
# 3. PolicyEngine Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPolicyEngineNonBankLoan:
    """Tests for policy rules governing NonBankLoan."""

    @pytest.fixture()
    def policy(self) -> PolicyEngine:
        return PolicyEngine.default()

    def _make_loan(self, holder_id: str, issuer_id: str) -> NonBankLoan:
        return NonBankLoan(
            id="NBL_test",
            kind=InstrumentKind.NON_BANK_LOAN,
            amount=100,
            denom="X",
            asset_holder_id=holder_id,
            liability_issuer_id=issuer_id,
        )

    def test_nonbank_lender_can_hold_loan(self, policy: PolicyEngine):
        """NonBankLender is allowed to hold a NonBankLoan (as asset)."""
        lender = NonBankLender(id="NBL01", name="Lender")
        loan = self._make_loan("NBL01", "F01")
        assert policy.can_hold(lender, loan) is True

    def test_firm_cannot_hold_loan(self, policy: PolicyEngine):
        """A Firm cannot hold a NonBankLoan as an asset."""
        firm = Firm(id="F01", name="Firm 1", kind="firm")
        loan = self._make_loan("F01", "F02")
        assert policy.can_hold(firm, loan) is False

    def test_household_cannot_hold_loan(self, policy: PolicyEngine):
        """A Household cannot hold a NonBankLoan as an asset."""
        hh = Household(id="HH01", name="Household", kind="household")
        loan = self._make_loan("HH01", "F01")
        assert policy.can_hold(hh, loan) is False

    def test_bank_cannot_hold_loan(self, policy: PolicyEngine):
        """A Bank cannot hold a NonBankLoan as an asset."""
        bank = Bank(id="B01", name="Bank", kind="bank")
        loan = self._make_loan("B01", "F01")
        assert policy.can_hold(bank, loan) is False

    def test_any_agent_can_issue_loan(self, policy: PolicyEngine):
        """Any agent (as borrower) can issue a NonBankLoan liability."""
        loan = self._make_loan("NBL01", "F01")

        firm = Firm(id="F01", name="Firm 1", kind="firm")
        assert policy.can_issue(firm, loan) is True

        hh = Household(id="HH01", name="Household", kind="household")
        loan2 = self._make_loan("NBL01", "HH01")
        assert policy.can_issue(hh, loan2) is True

    def test_mop_rank_for_lender(self, policy: PolicyEngine):
        """NonBankLender uses CASH as its means of payment."""
        rank = policy.mop_rank.get(AgentKind.NON_BANK_LENDER)
        assert rank is not None
        assert InstrumentKind.CASH in rank


# ═══════════════════════════════════════════════════════════════════════
# 4. System Method Tests
# ═══════════════════════════════════════════════════════════════════════


class TestNonbankLendCash:
    """Tests for System.nonbank_lend_cash."""

    def test_successful_loan(self):
        """Lender cash decreases, borrower cash increases, loan instrument created."""
        system = _build_lending_system(lender_cash=5000, firm_cash=200)

        # Get initial cash
        lender_cash_before = _agent_cash(system, "NBL01")
        borrower_cash_before = _agent_cash(system, "F01")
        assert lender_cash_before == 5000
        assert borrower_cash_before == 200

        loan_id = system.nonbank_lend_cash(
            lender_id="NBL01",
            borrower_id="F01",
            amount=1000,
            rate=Decimal("0.05"),
            day=0,
            maturity_days=3,
        )

        # Lender lost cash, borrower gained cash
        assert _agent_cash(system, "NBL01") == 4000
        assert _agent_cash(system, "F01") == 1200

        # Loan contract exists
        loan = system.state.contracts[loan_id]
        assert isinstance(loan, NonBankLoan)
        assert loan.amount == 1000
        assert loan.rate == Decimal("0.05")
        assert loan.issuance_day == 0
        assert loan.maturity_days == 3
        assert loan.asset_holder_id == "NBL01"
        assert loan.liability_issuer_id == "F01"

        # Loan in lender assets and borrower liabilities
        assert loan_id in system.state.agents["NBL01"].asset_ids
        assert loan_id in system.state.agents["F01"].liability_ids

    def test_insufficient_cash_raises(self):
        """Lending more cash than lender has should raise ValidationError."""
        system = _build_lending_system(lender_cash=500, firm_cash=100)

        with pytest.raises(ValidationError, match="insufficient cash"):
            system.nonbank_lend_cash(
                lender_id="NBL01",
                borrower_id="F01",
                amount=1000,
                rate=Decimal("0.05"),
                day=0,
            )

    def test_loan_events_logged(self):
        """A NonBankLoanCreated event is logged."""
        system = _build_lending_system(lender_cash=5000, firm_cash=0)

        loan_id = system.nonbank_lend_cash(
            lender_id="NBL01",
            borrower_id="F01",
            amount=500,
            rate=Decimal("0.10"),
            day=1,
            maturity_days=2,
        )

        loan_events = [
            e for e in system.state.events if e.get("kind") == "NonBankLoanCreated"
        ]
        assert len(loan_events) == 1
        evt = loan_events[0]
        assert evt["lender_id"] == "NBL01"
        assert evt["borrower_id"] == "F01"
        assert evt["amount"] == 500
        assert evt["loan_id"] == loan_id

    def test_multiple_loans(self):
        """Multiple loans can be created to the same or different borrowers."""
        system = _build_lending_system(lender_cash=10000, firm_cash=0)

        # Add another firm
        firm3 = Firm(id="F03", name="Firm 3", kind="firm")
        system.add_agent(firm3)

        loan1 = system.nonbank_lend_cash(
            lender_id="NBL01", borrower_id="F01",
            amount=2000, rate=Decimal("0.05"), day=0,
        )
        loan2 = system.nonbank_lend_cash(
            lender_id="NBL01", borrower_id="F03",
            amount=3000, rate=Decimal("0.08"), day=0,
        )

        assert _agent_cash(system, "NBL01") == 5000
        assert _agent_cash(system, "F01") == 2000
        assert _agent_cash(system, "F03") == 3000
        assert loan1 != loan2


class TestNonbankRepayLoan:
    """Tests for System.nonbank_repay_loan."""

    def test_successful_repayment(self):
        """Borrower pays principal + interest; lender gets cash back; loan removed."""
        system = _build_lending_system(lender_cash=5000, firm_cash=0)

        loan_id = system.nonbank_lend_cash(
            lender_id="NBL01",
            borrower_id="F01",
            amount=1000,
            rate=Decimal("0.10"),
            day=0,
            maturity_days=2,
        )

        # Give borrower enough cash to repay (1000 * 1.10 = 1100)
        system.mint_cash("F01", 1100)

        lender_cash_before = _agent_cash(system, "NBL01")
        borrower_cash_before = _agent_cash(system, "F01")

        repaid = system.nonbank_repay_loan(loan_id, "F01")

        assert repaid is True
        # Lender got repayment (1100)
        assert _agent_cash(system, "NBL01") == lender_cash_before + 1100
        # Borrower lost repayment amount
        assert _agent_cash(system, "F01") == borrower_cash_before - 1100
        # Loan contract removed
        assert loan_id not in system.state.contracts
        assert loan_id not in system.state.agents["NBL01"].asset_ids
        assert loan_id not in system.state.agents["F01"].liability_ids

    def test_default_when_borrower_cannot_pay(self):
        """If borrower has insufficient cash, loan defaults (written off)."""
        system = _build_lending_system(lender_cash=5000, firm_cash=0)

        loan_id = system.nonbank_lend_cash(
            lender_id="NBL01",
            borrower_id="F01",
            amount=1000,
            rate=Decimal("0.10"),
            day=0,
            maturity_days=2,
        )

        # Borrower has 1000 from the loan, needs 1100 to repay
        borrower_cash = _agent_cash(system, "F01")
        assert borrower_cash == 1000  # Only from the loan itself

        repaid = system.nonbank_repay_loan(loan_id, "F01")

        assert repaid is False
        # Loan should be removed (written off)
        assert loan_id not in system.state.contracts

        # Default event logged
        default_events = [
            e for e in system.state.events if e.get("kind") == "NonBankLoanDefaulted"
        ]
        assert len(default_events) == 1

    def test_repayment_events_logged(self):
        """Successful repayment logs a NonBankLoanRepaid event."""
        system = _build_lending_system(lender_cash=5000, firm_cash=2000)

        loan_id = system.nonbank_lend_cash(
            lender_id="NBL01",
            borrower_id="F01",
            amount=1000,
            rate=Decimal("0.05"),
            day=0,
        )

        system.nonbank_repay_loan(loan_id, "F01")

        repaid_events = [
            e for e in system.state.events if e.get("kind") == "NonBankLoanRepaid"
        ]
        assert len(repaid_events) == 1
        evt = repaid_events[0]
        assert evt["loan_id"] == loan_id
        assert evt["borrower_id"] == "F01"
        assert evt["lender_id"] == "NBL01"
        assert evt["principal"] == 1000
        assert evt["interest"] == 50
        assert evt["total_repaid"] == 1050

    def test_repay_nonexistent_loan_raises(self):
        """Repaying a nonexistent loan raises ValidationError."""
        system = _build_lending_system()
        with pytest.raises(ValidationError, match="not found"):
            system.nonbank_repay_loan("FAKE_LOAN", "F01")

    def test_repay_wrong_borrower_raises(self):
        """Repaying with the wrong borrower ID raises ValidationError."""
        system = _build_lending_system(lender_cash=5000, firm_cash=0)

        loan_id = system.nonbank_lend_cash(
            lender_id="NBL01",
            borrower_id="F01",
            amount=1000,
            rate=Decimal("0.05"),
            day=0,
        )

        with pytest.raises(ValidationError, match="not the borrower"):
            system.nonbank_repay_loan(loan_id, "F02")


class TestGetNonbankLoansDue:
    """Tests for System.get_nonbank_loans_due."""

    def test_returns_due_loans(self):
        """Returns loan IDs when current_day >= maturity_day."""
        system = _build_lending_system(lender_cash=10000, firm_cash=0)

        loan_id = system.nonbank_lend_cash(
            lender_id="NBL01",
            borrower_id="F01",
            amount=1000,
            rate=Decimal("0.05"),
            day=0,
            maturity_days=2,
        )

        # Not due on day 1
        assert loan_id not in system.get_nonbank_loans_due(1)
        # Due on day 2
        assert loan_id in system.get_nonbank_loans_due(2)
        # Still due on day 3 (overdue)
        assert loan_id in system.get_nonbank_loans_due(3)

    def test_does_not_return_undue_loans(self):
        """Loans not yet mature are not returned."""
        system = _build_lending_system(lender_cash=10000, firm_cash=0)

        system.nonbank_lend_cash(
            lender_id="NBL01",
            borrower_id="F01",
            amount=1000,
            rate=Decimal("0.05"),
            day=0,
            maturity_days=5,
        )

        due = system.get_nonbank_loans_due(3)
        assert due == []

    def test_multiple_loans_different_maturities(self):
        """Only the loans that are due are returned, others are not."""
        system = _build_lending_system(lender_cash=10000, firm_cash=0)

        loan_short = system.nonbank_lend_cash(
            lender_id="NBL01", borrower_id="F01",
            amount=1000, rate=Decimal("0.05"), day=0, maturity_days=2,
        )
        loan_long = system.nonbank_lend_cash(
            lender_id="NBL01", borrower_id="F01",
            amount=1000, rate=Decimal("0.05"), day=0, maturity_days=5,
        )

        due_day2 = system.get_nonbank_loans_due(2)
        assert loan_short in due_day2
        assert loan_long not in due_day2

        due_day5 = system.get_nonbank_loans_due(5)
        assert loan_short in due_day5  # overdue
        assert loan_long in due_day5


# ═══════════════════════════════════════════════════════════════════════
# 5. LendingConfig Tests
# ═══════════════════════════════════════════════════════════════════════


class TestLendingConfig:
    """Tests for LendingConfig defaults and custom values."""

    def test_defaults(self):
        """Default config matches expected values."""
        cfg = LendingConfig()
        assert cfg.base_rate == Decimal("0.05")
        assert cfg.risk_premium_scale == Decimal("0.20")
        assert cfg.max_single_exposure == Decimal("0.15")
        assert cfg.max_total_exposure == Decimal("0.80")
        assert cfg.maturity_days == 2
        assert cfg.horizon == 3
        assert cfg.min_shortfall == 1
        assert cfg.max_default_prob == Decimal("0.50")

    def test_custom_values(self):
        """Custom config values are stored correctly."""
        cfg = LendingConfig(
            base_rate=Decimal("0.10"),
            risk_premium_scale=Decimal("0.50"),
            max_single_exposure=Decimal("0.30"),
            max_total_exposure=Decimal("0.90"),
            maturity_days=5,
            horizon=7,
            min_shortfall=100,
            max_default_prob=Decimal("0.80"),
        )
        assert cfg.base_rate == Decimal("0.10")
        assert cfg.risk_premium_scale == Decimal("0.50")
        assert cfg.max_single_exposure == Decimal("0.30")
        assert cfg.max_total_exposure == Decimal("0.90")
        assert cfg.maturity_days == 5
        assert cfg.horizon == 7
        assert cfg.min_shortfall == 100
        assert cfg.max_default_prob == Decimal("0.80")


# ═══════════════════════════════════════════════════════════════════════
# 6. Lending Strategy Tests (run_lending_phase)
# ═══════════════════════════════════════════════════════════════════════


class TestRunLendingPhase:
    """Tests for run_lending_phase and run_loan_repayments."""

    def test_no_lender_returns_empty(self):
        """If no NonBankLender agent exists, returns empty events."""
        system = System()
        cb = CentralBank(id="CB01", name="CB", kind="central_bank")
        firm = Firm(id="F01", name="Firm", kind="firm")
        system.bootstrap_cb(cb)
        system.add_agent(firm)
        system.mint_cash("F01", 500)

        events = run_lending_phase(system, current_day=0)
        assert events == []

    def test_no_shortfalls_returns_empty(self):
        """If borrowers have enough cash, no loans are made."""
        system = _build_lending_system(
            lender_cash=10000,
            firm_cash=5000,       # Way more than the payable
            firm_payable_amount=100,
            payable_due_day=2,
        )

        config = LendingConfig(horizon=3)
        events = run_lending_phase(system, current_day=0, lending_config=config)
        assert events == []

    def test_shortfall_creates_loan(self):
        """With a firm in shortfall, the lender creates a loan."""
        system = _build_lending_system(
            lender_cash=10000,
            firm_cash=200,
            firm_payable_amount=1000,
            payable_due_day=2,
        )

        config = LendingConfig(horizon=3, min_shortfall=1)
        events = run_lending_phase(system, current_day=0, lending_config=config)

        assert len(events) >= 1
        evt = events[0]
        assert evt["kind"] == "NonBankLoanCreated"
        assert evt["borrower_id"] == "F01"
        assert evt["lender_id"] == "NBL01"
        assert evt["amount"] > 0

    def test_respects_max_default_prob_filter(self):
        """Borrowers with high default probability are excluded."""
        system = _build_lending_system(
            lender_cash=10000,
            firm_cash=200,
            firm_payable_amount=1000,
            payable_due_day=2,
        )

        # Use very restrictive max_default_prob so all borrowers are excluded.
        # The default prob heuristic will produce at least 0.05, so set threshold below that.
        config = LendingConfig(
            horizon=3,
            min_shortfall=1,
            max_default_prob=Decimal("0.01"),
        )
        events = run_lending_phase(system, current_day=0, lending_config=config)
        assert events == []

    def test_respects_exposure_limit(self):
        """Loan amount is capped by max_single_exposure."""
        system = _build_lending_system(
            lender_cash=10000,
            firm_cash=0,
            firm_payable_amount=5000,
            payable_due_day=2,
        )

        # max_single_exposure = 0.15 of initial_capital (10000) = 1500
        config = LendingConfig(
            horizon=3,
            min_shortfall=1,
            max_single_exposure=Decimal("0.15"),
        )
        events = run_lending_phase(system, current_day=0, lending_config=config)

        if events:
            # The loan amount should not exceed 1500
            assert events[0]["amount"] <= 1500

    def test_loan_repayments_processes_due_loans(self):
        """run_loan_repayments processes loans due on the given day."""
        system = _build_lending_system(lender_cash=10000, firm_cash=0)

        # Create a loan maturing on day 2
        system.nonbank_lend_cash(
            lender_id="NBL01",
            borrower_id="F01",
            amount=1000,
            rate=Decimal("0.05"),
            day=0,
            maturity_days=2,
        )

        # Give borrower enough to repay (1050)
        system.mint_cash("F01", 100)  # Already has 1000 from loan

        events = run_loan_repayments(system, current_day=2)
        assert len(events) == 1
        # Borrower has 1000 + 100 = 1100 but needs 1050 -> should repay
        assert events[0]["kind"] == "NonBankLoanRepaid"
        assert events[0]["repaid"] is True

    def test_loan_repayments_handles_default(self):
        """run_loan_repayments records default when borrower cannot pay."""
        system = _build_lending_system(lender_cash=10000, firm_cash=0)

        # Create a loan. Borrower gets exactly 1000 cash from the loan.
        # Repayment is 1050 (1000 * 1.05), borrower only has 1000.
        system.nonbank_lend_cash(
            lender_id="NBL01",
            borrower_id="F01",
            amount=1000,
            rate=Decimal("0.05"),
            day=0,
            maturity_days=2,
        )

        events = run_loan_repayments(system, current_day=2)
        assert len(events) == 1
        assert events[0]["kind"] == "NonBankLoanDefaulted"
        assert events[0]["repaid"] is False

    def test_loan_repayments_no_loans_due(self):
        """run_loan_repayments returns empty when no loans are due."""
        system = _build_lending_system(lender_cash=10000, firm_cash=0)

        system.nonbank_lend_cash(
            lender_id="NBL01",
            borrower_id="F01",
            amount=1000,
            rate=Decimal("0.05"),
            day=0,
            maturity_days=5,
        )

        events = run_loan_repayments(system, current_day=2)
        assert events == []

    def test_lender_defaulted_skipped(self):
        """If the lender agent is defaulted, run_lending_phase skips it."""
        system = _build_lending_system(lender_cash=10000, firm_cash=200, firm_payable_amount=1000)
        system.state.agents["NBL01"].defaulted = True

        events = run_lending_phase(system, current_day=0)
        assert events == []

    def test_defaulted_borrower_skipped(self):
        """Defaulted borrowers are not considered for loans."""
        system = _build_lending_system(lender_cash=10000, firm_cash=200, firm_payable_amount=1000)
        system.state.agents["F01"].defaulted = True

        config = LendingConfig(horizon=3, min_shortfall=1)
        events = run_lending_phase(system, current_day=0, lending_config=config)
        assert events == []


# ═══════════════════════════════════════════════════════════════════════
# 7. Simulation Integration Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSimulationIntegration:
    """Tests for run_day with enable_lender flag."""

    def test_run_day_with_lender_enabled(self):
        """run_day with enable_lender=True executes the lending phase."""
        from bilancio.engines.simulation import run_day

        system = _build_lending_system(
            lender_cash=10000,
            firm_cash=200,
            firm_payable_amount=1000,
            payable_due_day=2,
        )
        system.state.lender_config = LendingConfig(horizon=3, min_shortfall=1)

        run_day(system, enable_lender=True)

        # Check that SubphaseB_Lending was logged
        lending_phases = [
            e for e in system.state.events if e.get("kind") == "SubphaseB_Lending"
        ]
        assert len(lending_phases) == 1

        # Should have lending events in the event list
        loan_events = [
            e for e in system.state.events if e.get("kind") == "NonBankLoanCreated"
        ]
        assert len(loan_events) >= 1

    def test_run_day_without_lender_skips_lending(self):
        """run_day with enable_lender=False does not run lending phase."""
        from bilancio.engines.simulation import run_day

        system = _build_lending_system(
            lender_cash=10000,
            firm_cash=200,
            firm_payable_amount=1000,
            payable_due_day=2,
        )
        system.state.lender_config = LendingConfig()

        run_day(system, enable_lender=False)

        lending_phases = [
            e for e in system.state.events if e.get("kind") == "SubphaseB_Lending"
        ]
        assert len(lending_phases) == 0

    def test_run_day_with_lender_true_but_no_config_skips(self):
        """enable_lender=True but lender_config=None skips lending phase."""
        from bilancio.engines.simulation import run_day

        system = _build_lending_system(lender_cash=10000, firm_cash=200)
        assert system.state.lender_config is None

        run_day(system, enable_lender=True)

        lending_phases = [
            e for e in system.state.events if e.get("kind") == "SubphaseB_Lending"
        ]
        assert len(lending_phases) == 0

    def test_run_day_processes_loan_repayments(self):
        """run_day processes loan repayments at the appropriate day."""
        from bilancio.engines.simulation import run_day

        system = _build_lending_system(
            lender_cash=5000,
            firm_cash=0,
            firm_payable_amount=0,
        )

        # Create a loan maturing on day 2
        system.nonbank_lend_cash(
            lender_id="NBL01",
            borrower_id="F01",
            amount=1000,
            rate=Decimal("0.05"),
            day=0,
            maturity_days=2,
        )

        # Give firm enough to repay
        system.mint_cash("F01", 100)  # Total: 1100, repayment: 1050

        # Day 0 -> no repayment (not due yet)
        run_day(system, enable_lender=False)
        assert system.state.day == 1

        # Day 1 -> still no repayment (maturity_day=2)
        run_day(system, enable_lender=False)
        assert system.state.day == 2

        # Day 2 -> loan is due, should be repaid
        run_day(system, enable_lender=False)
        assert system.state.day == 3

        repaid_events = [
            e for e in system.state.events if e.get("kind") == "NonBankLoanRepaid"
        ]
        assert len(repaid_events) == 1


# ═══════════════════════════════════════════════════════════════════════
# 8. Scenario Generation Tests
# ═══════════════════════════════════════════════════════════════════════


class TestScenarioGenerationLender:
    """Tests for compile_ring_explorer_balanced with mode='lender'."""

    def _make_config(self) -> "RingExplorerGeneratorConfig":
        from bilancio.config.models import (
            RingExplorerGeneratorConfig,
            RingExplorerParamsModel,
            GeneratorCompileConfig,
            RingExplorerLiquidityConfig,
        )

        params = RingExplorerParamsModel(
            n_agents=5,
            seed=42,
            kappa=Decimal("1"),
            liquidity=RingExplorerLiquidityConfig(total=Decimal("5000")),
        )
        config = RingExplorerGeneratorConfig(
            version=1,
            generator="ring_explorer_v1",
            name_prefix="test",
            params=params,
            compile=GeneratorCompileConfig(emit_yaml=False),
        )
        return config

    def test_lender_mode_includes_lender_agent(self):
        """compile_ring_explorer_balanced with mode='lender' includes a non_bank_lender agent."""
        from bilancio.scenarios.ring_explorer import compile_ring_explorer_balanced

        config = self._make_config()
        scenario = compile_ring_explorer_balanced(config, mode="lender")

        agent_kinds = {a["kind"] for a in scenario["agents"]}
        assert "non_bank_lender" in agent_kinds

        lender_agents = [a for a in scenario["agents"] if a["kind"] == "non_bank_lender"]
        assert len(lender_agents) == 1
        assert lender_agents[0]["id"] == "lender"

    def test_lender_gets_cash_allocation(self):
        """Lender agent gets cash via mint_cash action."""
        from bilancio.scenarios.ring_explorer import compile_ring_explorer_balanced

        config = self._make_config()
        scenario = compile_ring_explorer_balanced(
            config, mode="lender", lender_share=Decimal("0.10"),
        )

        mint_actions = [
            a for a in scenario["initial_actions"]
            if "mint_cash" in a and a["mint_cash"].get("to") == "lender"
        ]
        assert len(mint_actions) == 1
        assert mint_actions[0]["mint_cash"]["amount"] > 0

    def test_non_lender_mode_has_no_lender_agent(self):
        """compile_ring_explorer_balanced with mode='active' does NOT include a lender."""
        from bilancio.scenarios.ring_explorer import compile_ring_explorer_balanced

        config = self._make_config()
        scenario = compile_ring_explorer_balanced(config, mode="active")

        agent_kinds = {a["kind"] for a in scenario["agents"]}
        assert "non_bank_lender" not in agent_kinds

    def test_balanced_config_stores_lender_share(self):
        """The _balanced_config metadata includes lender_share."""
        from bilancio.scenarios.ring_explorer import compile_ring_explorer_balanced

        config = self._make_config()
        scenario = compile_ring_explorer_balanced(
            config, mode="lender", lender_share=Decimal("0.15"),
        )

        balanced_cfg = scenario.get("_balanced_config", {})
        assert balanced_cfg.get("lender_share") == 0.15
        assert balanced_cfg.get("mode") == "lender"


# ═══════════════════════════════════════════════════════════════════════
# 9. LenderScenarioConfig Pydantic Model Tests
# ═══════════════════════════════════════════════════════════════════════


class TestLenderScenarioConfig:
    """Tests for the LenderScenarioConfig Pydantic model."""

    def test_defaults(self):
        cfg = LenderScenarioConfig()
        assert cfg.enabled is False
        assert cfg.base_rate == Decimal("0.05")
        assert cfg.risk_premium_scale == Decimal("0.20")
        assert cfg.max_single_exposure == Decimal("0.15")
        assert cfg.max_total_exposure == Decimal("0.80")
        assert cfg.maturity_days == 2
        assert cfg.horizon == 3

    def test_custom_values(self):
        cfg = LenderScenarioConfig(
            enabled=True,
            base_rate=Decimal("0.10"),
            maturity_days=5,
            horizon=7,
        )
        assert cfg.enabled is True
        assert cfg.base_rate == Decimal("0.10")
        assert cfg.maturity_days == 5
        assert cfg.horizon == 7

    def test_scenario_config_has_lender_field(self):
        """ScenarioConfig accepts an optional lender field."""
        cfg = ScenarioConfig(
            name="test",
            agents=[
                {"id": "CB01", "kind": "central_bank", "name": "CB"},
            ],
            lender=LenderScenarioConfig(enabled=True),
        )
        assert cfg.lender is not None
        assert cfg.lender.enabled is True

    def test_scenario_config_lender_defaults_none(self):
        """ScenarioConfig.lender defaults to None."""
        cfg = ScenarioConfig(
            name="test",
            agents=[
                {"id": "CB01", "kind": "central_bank", "name": "CB"},
            ],
        )
        assert cfg.lender is None


# ── Helper used across test classes ──────────────────────────────────


def _agent_cash(system: System, agent_id: str) -> int:
    """Sum all cash held by an agent."""
    total = 0
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is not None and contract.kind == InstrumentKind.CASH:
            total += contract.amount
    return total
