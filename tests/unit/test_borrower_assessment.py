"""Unit tests for _assess_borrower in bank_lending.py.

Tests the borrower balance-sheet assessment that banks perform before
granting a loan (Plan 042).  The coverage ratio measures the borrower's
ability to repay:

    coverage = (liquid - obligations + quality_receivables) / loan_repayment
"""

from decimal import Decimal

import pytest

from bilancio.domain.agent import Agent, AgentKind
from bilancio.domain.instruments.bank_loan import BankLoan
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.means_of_payment import BankDeposit, Cash
from bilancio.engines.bank_lending import _assess_borrower
from bilancio.engines.system import System


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_system() -> System:
    """Create a minimal System with a CB agent (needed as liability issuer for cash)."""
    system = System()
    # CB agent needed as liability_issuer for Cash instruments
    cb = Agent(id="cb", name="Central Bank", kind=AgentKind.CENTRAL_BANK)
    system.state.agents["cb"] = cb
    return system


def _add_firm(system: System, firm_id: str) -> Agent:
    """Add a firm agent and return it."""
    agent = Agent(id=firm_id, name=firm_id, kind=AgentKind.FIRM)
    system.state.agents[firm_id] = agent
    return agent


def _add_cash(system: System, agent: Agent, amount: int, cash_id: str = "cash_1") -> None:
    """Give an agent cash."""
    cash = Cash(
        id=cash_id,
        kind=InstrumentKind.CASH,
        amount=amount,
        denom="USD",
        asset_holder_id=agent.id,
        liability_issuer_id="cb",
    )
    system.state.contracts[cash_id] = cash
    agent.asset_ids.append(cash_id)


def _add_deposit(
    system: System, agent: Agent, amount: int, bank_id: str = "bank_1", dep_id: str = "dep_1",
) -> None:
    """Give an agent a bank deposit."""
    # Ensure a bank agent exists as liability issuer
    if bank_id not in system.state.agents:
        bank = Agent(id=bank_id, name=bank_id, kind=AgentKind.BANK)
        system.state.agents[bank_id] = bank
    deposit = BankDeposit(
        id=dep_id,
        kind=InstrumentKind.BANK_DEPOSIT,
        amount=amount,
        denom="USD",
        asset_holder_id=agent.id,
        liability_issuer_id=bank_id,
    )
    system.state.contracts[dep_id] = deposit
    agent.asset_ids.append(dep_id)


def _add_payable_liability(
    system: System, agent: Agent, amount: int, due_day: int,
    creditor_id: str = "creditor_1", pay_id: str = "pay_l1",
) -> None:
    """Add a payable that the agent owes (liability)."""
    if creditor_id not in system.state.agents:
        creditor = Agent(id=creditor_id, name=creditor_id, kind=AgentKind.FIRM)
        system.state.agents[creditor_id] = creditor
    payable = Payable(
        id=pay_id,
        kind=InstrumentKind.PAYABLE,
        amount=amount,
        denom="USD",
        asset_holder_id=creditor_id,
        liability_issuer_id=agent.id,
        due_day=due_day,
    )
    system.state.contracts[pay_id] = payable
    agent.liability_ids.append(pay_id)


def _add_payable_receivable(
    system: System, agent: Agent, amount: int, due_day: int,
    obligor_id: str = "obligor_1", pay_id: str = "pay_r1",
) -> None:
    """Add a payable that the agent is owed (asset / receivable)."""
    if obligor_id not in system.state.agents:
        obligor = Agent(id=obligor_id, name=obligor_id, kind=AgentKind.FIRM)
        system.state.agents[obligor_id] = obligor
    payable = Payable(
        id=pay_id,
        kind=InstrumentKind.PAYABLE,
        amount=amount,
        denom="USD",
        asset_holder_id=agent.id,
        liability_issuer_id=obligor_id,
        due_day=due_day,
    )
    system.state.contracts[pay_id] = payable
    agent.asset_ids.append(pay_id)


def _add_bank_loan_liability(
    system: System, agent: Agent, principal: int, rate: Decimal,
    issuance_day: int, maturity_days: int,
    bank_id: str = "bank_1", loan_id: str = "bl_1",
) -> None:
    """Add a BankLoan instrument as a liability of the agent."""
    if bank_id not in system.state.agents:
        bank = Agent(id=bank_id, name=bank_id, kind=AgentKind.BANK)
        system.state.agents[bank_id] = bank
    loan = BankLoan(
        id=loan_id,
        kind=InstrumentKind.BANK_LOAN,
        amount=principal,
        denom="USD",
        asset_holder_id=bank_id,
        liability_issuer_id=agent.id,
        rate=rate,
        issuance_day=issuance_day,
        maturity_days=maturity_days,
    )
    system.state.contracts[loan_id] = loan
    agent.liability_ids.append(loan_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthyBorrower:
    """Test basic coverage computation when borrower has liquid > obligations."""

    def test_healthy_borrower_coverage_above_one(self):
        """Borrower with plenty of cash and no obligations has high coverage."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 1000)

        # Borrow 100 at 10% rate, maturity 5 days, current day 0
        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        # loan_repayment = int(100 * 1.10) = 110
        # net_resources = 1000 (cash) - 0 (obligations) + 0 (receivables) = 1000
        # coverage = 1000 / 110 ≈ 9.09
        expected = Decimal(1000) / Decimal(110)
        assert coverage == expected

    def test_healthy_borrower_with_deposits(self):
        """Deposits count as liquid assets alongside cash."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 300)
        _add_deposit(system, firm, 200)

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.05"),
            loan_maturity=5, current_day=0,
        )
        # liquid = 300 + 200 = 500
        # loan_repayment = int(100 * 1.05) = 105
        # coverage = 500 / 105
        expected = Decimal(500) / Decimal(105)
        assert coverage == expected

    def test_obligations_reduce_coverage(self):
        """Payable liabilities due within window reduce coverage."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 500)
        _add_payable_liability(system, firm, amount=200, due_day=3)

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        # liquid = 500, obligations = 200, receivables = 0
        # net_resources = 500 - 200 = 300
        # loan_repayment = 110
        # coverage = 300 / 110
        expected = Decimal(300) / Decimal(110)
        assert coverage == expected

    def test_receivables_increase_coverage(self):
        """Quality receivables arriving within window increase coverage."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 200)
        _add_payable_liability(system, firm, amount=150, due_day=3)
        _add_payable_receivable(system, firm, amount=100, due_day=2)

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        # liquid = 200, obligations = 150, receivables = 100
        # net_resources = 200 - 150 + 100 = 150
        # loan_repayment = 110
        # coverage = 150 / 110
        expected = Decimal(150) / Decimal(110)
        assert coverage == expected


class TestNoReceivables:
    """Test coverage when borrower has no receivables -- just cash vs obligations."""

    def test_cash_only_covers_loan(self):
        """Borrower with cash exactly equal to repayment has coverage ~1."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 110)

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        # net_resources = 110, loan_repayment = 110 → coverage = 1
        assert coverage == Decimal(1)

    def test_cash_below_repayment(self):
        """Borrower with less cash than loan repayment has coverage < 1."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 55)

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        # net_resources = 55, loan_repayment = 110 → coverage = 0.5
        assert coverage == Decimal(55) / Decimal(110)

    def test_bank_loan_obligations_counted(self):
        """Existing BankLoan liabilities due within window reduce coverage."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 500)
        # Existing bank loan: principal=100, rate=5%, issuance_day=0, maturity=3 days
        # maturity_day=3, repayment_amount = int(100 * 1.05) = 105
        _add_bank_loan_liability(
            system, firm, principal=100, rate=Decimal("0.05"),
            issuance_day=0, maturity_days=3,
        )

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        # liquid = 500, obligations = 105 (bank loan repayment)
        # net_resources = 500 - 105 = 395
        # loan_repayment = 110
        # coverage = 395 / 110
        expected = Decimal(395) / Decimal(110)
        assert coverage == expected


class TestDefaultedCounterpartyExclusion:
    """Receivables from defaulted counterparties should be excluded (worth zero)."""

    def test_defaulted_obligor_receivable_excluded(self):
        """Receivable from a defaulted agent is not counted."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 200)
        # Receivable from "obligor_bad" who has defaulted
        _add_payable_receivable(
            system, firm, amount=300, due_day=3,
            obligor_id="obligor_bad", pay_id="pay_bad",
        )
        system.state.defaulted_agent_ids.add("obligor_bad")

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        # liquid = 200, obligations = 0, quality_receivables = 0 (excluded)
        # net_resources = 200
        # loan_repayment = 110
        expected = Decimal(200) / Decimal(110)
        assert coverage == expected

    def test_mix_of_good_and_defaulted_receivables(self):
        """Only non-defaulted receivables contribute to coverage."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 100)

        # Good receivable
        _add_payable_receivable(
            system, firm, amount=200, due_day=3,
            obligor_id="obligor_good", pay_id="pay_good",
        )
        # Bad receivable (defaulted)
        _add_payable_receivable(
            system, firm, amount=500, due_day=4,
            obligor_id="obligor_bad", pay_id="pay_bad",
        )
        system.state.defaulted_agent_ids.add("obligor_bad")

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        # liquid = 100, obligations = 0
        # quality_receivables = 200 (good only; 500 from bad excluded)
        # net_resources = 100 + 200 = 300
        # loan_repayment = 110
        expected = Decimal(300) / Decimal(110)
        assert coverage == expected

    def test_all_receivables_defaulted(self):
        """When all receivables are from defaulted agents, only liquid matters."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 50)
        _add_payable_receivable(
            system, firm, amount=1000, due_day=2,
            obligor_id="bad_1", pay_id="pay_1",
        )
        _add_payable_receivable(
            system, firm, amount=2000, due_day=3,
            obligor_id="bad_2", pay_id="pay_2",
        )
        system.state.defaulted_agent_ids.add("bad_1")
        system.state.defaulted_agent_ids.add("bad_2")

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        # net_resources = 50 - 0 + 0 = 50
        # loan_repayment = 110
        expected = Decimal(50) / Decimal(110)
        assert coverage == expected


class TestStructuralInsolvency:
    """Coverage < 0 when obligations exceed liquid + quality receivables."""

    def test_negative_coverage_insolvent(self):
        """Borrower with large obligations and little cash gets negative coverage."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 100)
        _add_payable_liability(system, firm, amount=500, due_day=2)

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        # liquid = 100, obligations = 500
        # net_resources = 100 - 500 = -400
        # loan_repayment = 110
        # coverage = -400 / 110
        expected = Decimal(-400) / Decimal(110)
        assert coverage == expected
        assert coverage < 0

    def test_deeply_negative_with_defaulted_receivables(self):
        """Insolvent borrower with only defaulted receivables."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 50)
        _add_payable_liability(system, firm, amount=800, due_day=3)
        _add_payable_receivable(
            system, firm, amount=1000, due_day=2,
            obligor_id="defaulted_co", pay_id="pay_r_bad",
        )
        system.state.defaulted_agent_ids.add("defaulted_co")

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        # liquid = 50, obligations = 800, quality_receivables = 0
        # net_resources = 50 - 800 = -750
        # loan_repayment = 110
        expected = Decimal(-750) / Decimal(110)
        assert coverage == expected
        assert coverage < 0

    def test_nonexistent_borrower_returns_negative(self):
        """Non-existent borrower_id returns -1."""
        system = _make_system()
        coverage = _assess_borrower(
            system, "nonexistent", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        assert coverage == Decimal("-1")


class TestMaturityWindowFiltering:
    """Receivables and obligations beyond the loan maturity window are excluded."""

    def test_receivable_beyond_window_excluded(self):
        """Receivable due after maturity_day is not counted."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 100)

        # current_day=0, loan_maturity=5 → maturity_day=5
        # This receivable is due on day 6, beyond the window
        _add_payable_receivable(
            system, firm, amount=500, due_day=6,
            obligor_id="obligor_late", pay_id="pay_late",
        )

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        # quality_receivables = 0 (day 6 > maturity_day 5)
        # net_resources = 100
        # loan_repayment = 110
        expected = Decimal(100) / Decimal(110)
        assert coverage == expected

    def test_receivable_on_maturity_day_included(self):
        """Receivable due exactly on maturity_day is counted."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 100)

        # maturity_day = 0 + 5 = 5; receivable due on day 5 → included
        _add_payable_receivable(
            system, firm, amount=200, due_day=5,
            obligor_id="obligor_ontime", pay_id="pay_ontime",
        )

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        # quality_receivables = 200
        # net_resources = 100 + 200 = 300
        # loan_repayment = 110
        expected = Decimal(300) / Decimal(110)
        assert coverage == expected

    def test_receivable_before_current_day_excluded(self):
        """Receivable due before current_day is excluded (already past)."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 100)

        # current_day=5, receivable due on day 3 → excluded (past)
        _add_payable_receivable(
            system, firm, amount=500, due_day=3,
            obligor_id="obligor_past", pay_id="pay_past",
        )

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=5,
        )
        # quality_receivables = 0 (day 3 < current_day 5)
        # net_resources = 100
        # loan_repayment = 110
        expected = Decimal(100) / Decimal(110)
        assert coverage == expected

    def test_obligation_beyond_window_excluded(self):
        """Payable liability due after maturity_day does not reduce coverage."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 100)

        # current_day=0, loan_maturity=3 → maturity_day=3
        # Obligation due on day 5 → beyond window
        _add_payable_liability(system, firm, amount=999, due_day=5)

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=3, current_day=0,
        )
        # obligations = 0 (day 5 > maturity_day 3)
        # net_resources = 100
        # loan_repayment = 110
        expected = Decimal(100) / Decimal(110)
        assert coverage == expected

    def test_mixed_in_and_out_of_window(self):
        """Some receivables/obligations in window, some out."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 300)

        # current_day=2, loan_maturity=5 → maturity_day=7
        # In-window receivable (day 4)
        _add_payable_receivable(
            system, firm, amount=100, due_day=4,
            obligor_id="obligor_a", pay_id="pay_r_in",
        )
        # Out-of-window receivable (day 10)
        _add_payable_receivable(
            system, firm, amount=999, due_day=10,
            obligor_id="obligor_b", pay_id="pay_r_out",
        )
        # In-window obligation (day 5)
        _add_payable_liability(
            system, firm, amount=150, due_day=5,
            creditor_id="cred_a", pay_id="pay_l_in",
        )
        # Out-of-window obligation (day 9)
        _add_payable_liability(
            system, firm, amount=888, due_day=9,
            creditor_id="cred_b", pay_id="pay_l_out",
        )

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0.10"),
            loan_maturity=5, current_day=2,
        )
        # liquid = 300
        # obligations = 150 (only day 5; day 9 excluded)
        # quality_receivables = 100 (only day 4; day 10 excluded)
        # net_resources = 300 - 150 + 100 = 250
        # loan_repayment = 110
        expected = Decimal(250) / Decimal(110)
        assert coverage == expected


class TestZeroLoanAmount:
    """Edge case: zero loan amount returns high coverage (Decimal 999)."""

    def test_zero_amount_returns_999(self):
        """Zero-cost loan (amount=0) returns coverage of 999."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 100)

        coverage = _assess_borrower(
            system, "firm_1", amount=0, rate=Decimal("0.10"),
            loan_maturity=5, current_day=0,
        )
        # loan_repayment = int(0 * 1.10) = 0, which is <= 0
        # Function returns Decimal("999")
        assert coverage == Decimal("999")

    def test_zero_rate_nonzero_amount(self):
        """Zero rate but nonzero amount still computes normally."""
        system = _make_system()
        firm = _add_firm(system, "firm_1")
        _add_cash(system, firm, 200)

        coverage = _assess_borrower(
            system, "firm_1", amount=100, rate=Decimal("0"),
            loan_maturity=5, current_day=0,
        )
        # loan_repayment = int(100 * 1.0) = 100
        # net_resources = 200
        # coverage = 200 / 100 = 2
        assert coverage == Decimal(2)
