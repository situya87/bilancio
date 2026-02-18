"""Tests for deposit interest accrual (bilancio.engines.bank_interest)."""

from decimal import Decimal

import pytest

from bilancio.banking.types import Quote
from bilancio.decision.profiles import BankProfile
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.bank_interest import accrue_deposit_interest
from bilancio.engines.banking_subsystem import (
    BankingSubsystem,
    BankTreynorState,
    _get_deposit_at_bank,
    initialize_banking_subsystem,
)
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash


def _make_banking_system() -> System:
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


def _make_banking_subsystem(
    system: System, interest_period: int = 2
) -> BankingSubsystem:
    """Initialize a BankingSubsystem with configurable interest_period."""
    profile = BankProfile(interest_period=interest_period)
    banking = initialize_banking_subsystem(
        system=system,
        bank_profile=profile,
        kappa=Decimal("1.0"),
        maturity_days=10,
    )
    return banking


class TestAccrueDepositInterest:
    """Tests for the accrue_deposit_interest function."""

    def test_no_accrual_on_day_0(self) -> None:
        """Day 0 is always skipped, even if it is a period boundary."""
        system = _make_banking_system()
        banking = _make_banking_subsystem(system, interest_period=1)

        # Set a positive deposit rate so interest would accrue if day were valid
        for bank_state in banking.banks.values():
            bank_state.current_quote = Quote(
                deposit_rate=Decimal("0.05"),
                loan_rate=Decimal("0.10"),
                day=0,
            )

        events = accrue_deposit_interest(system, current_day=0, banking=banking)

        assert events == []

    def test_no_accrual_on_non_period_day(self) -> None:
        """With interest_period=2, odd days are not period boundaries."""
        system = _make_banking_system()
        banking = _make_banking_subsystem(system, interest_period=2)

        for bank_state in banking.banks.values():
            bank_state.current_quote = Quote(
                deposit_rate=Decimal("0.05"),
                loan_rate=Decimal("0.10"),
                day=1,
            )

        events = accrue_deposit_interest(system, current_day=1, banking=banking)

        assert events == []

    def test_accrual_on_period_boundary(self) -> None:
        """Interest accrues on period boundary day (day 2 with period=2)."""
        system = _make_banking_system()
        banking = _make_banking_subsystem(system, interest_period=2)

        current_day = 2
        for bank_state in banking.banks.values():
            bank_state.current_quote = Quote(
                deposit_rate=Decimal("0.05"),
                loan_rate=Decimal("0.10"),
                day=current_day,
            )

        # Before accrual, each firm has a deposit of 1000
        assert _get_deposit_at_bank(system, "H_1", "bank_1") == 1000
        assert _get_deposit_at_bank(system, "H_2", "bank_2") == 1000

        events = accrue_deposit_interest(system, current_day=current_day, banking=banking)

        # Interest = floor(1000 * 0.05) = 50 per depositor
        assert len(events) == 2

        # Verify deposit balances increased
        assert _get_deposit_at_bank(system, "H_1", "bank_1") == 1050
        assert _get_deposit_at_bank(system, "H_2", "bank_2") == 1050

        # Verify event fields
        for event in events:
            assert event["kind"] == "DepositInterest"
            assert event["day"] == current_day
            assert event["interest"] == 50
            assert event["rate"] == "0.05"
            assert event["deposit_balance"] == 1050
            assert event["agent"] in ("H_1", "H_2")
            assert event["bank"] in ("bank_1", "bank_2")

        # Verify agent-bank mapping in events
        event_map = {e["agent"]: e for e in events}
        assert event_map["H_1"]["bank"] == "bank_1"
        assert event_map["H_2"]["bank"] == "bank_2"

    def test_no_accrual_when_r_d_zero(self) -> None:
        """Zero deposit rate produces no interest events."""
        system = _make_banking_system()
        banking = _make_banking_subsystem(system, interest_period=2)

        current_day = 2
        for bank_state in banking.banks.values():
            bank_state.current_quote = Quote(
                deposit_rate=Decimal("0"),
                loan_rate=Decimal("0.10"),
                day=current_day,
            )

        events = accrue_deposit_interest(system, current_day=current_day, banking=banking)

        assert events == []
        # Balances unchanged
        assert _get_deposit_at_bank(system, "H_1", "bank_1") == 1000
        assert _get_deposit_at_bank(system, "H_2", "bank_2") == 1000

    def test_no_accrual_when_r_d_negative(self) -> None:
        """Negative deposit rate does not cause negative interest."""
        system = _make_banking_system()
        banking = _make_banking_subsystem(system, interest_period=2)

        current_day = 2
        for bank_state in banking.banks.values():
            bank_state.current_quote = Quote(
                deposit_rate=Decimal("-0.01"),
                loan_rate=Decimal("0.10"),
                day=current_day,
            )

        events = accrue_deposit_interest(system, current_day=current_day, banking=banking)

        assert events == []
        # Balances unchanged
        assert _get_deposit_at_bank(system, "H_1", "bank_1") == 1000
        assert _get_deposit_at_bank(system, "H_2", "bank_2") == 1000

    def test_interest_amount_calculation(self) -> None:
        """Verify interest = floor(deposit * rate) with a larger deposit."""
        system = System(policy=PolicyEngine.default())
        cb = CentralBank(id="cb", name="Central Bank", kind="central_bank")
        bank = Bank(id="bank_1", name="Bank 1", kind="bank")
        firm = Firm(id="H_1", name="Firm 1", kind="firm")
        system.add_agent(cb)
        system.add_agent(bank)
        system.add_agent(firm)
        system.mint_cash(to_agent_id="H_1", amount=10000)
        deposit_cash(system, "H_1", "bank_1", 10000)
        system.mint_reserves(to_bank_id="bank_1", amount=50000)

        profile = BankProfile(interest_period=2)
        banking = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=Decimal("1.0"),
            maturity_days=10,
        )

        current_day = 2
        bank_state = banking.banks["bank_1"]
        bank_state.current_quote = Quote(
            deposit_rate=Decimal("0.03"),
            loan_rate=Decimal("0.10"),
            day=current_day,
        )

        events = accrue_deposit_interest(system, current_day=current_day, banking=banking)

        # Interest = floor(10000 * 0.03) = 300
        assert len(events) == 1
        event = events[0]
        assert event["interest"] == 300
        assert event["agent"] == "H_1"
        assert event["bank"] == "bank_1"
        assert event["deposit_balance"] == 10300
        assert _get_deposit_at_bank(system, "H_1", "bank_1") == 10300
