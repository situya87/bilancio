"""Tests for Plan 041: Paper-Faithful Bank Pricing.

Tests cover:
1-3.  burn_bank_cash: removes cash, preserves reserves, noop when no cash
4-6.  refresh_quote with 10-day projection: cash_tightness, inventory at t+2, scheduled legs
7-9.  CB end-of-day backstop: tops up reserves, noop above target, skipped when frozen
10.   CB freeze at stability
11.   Full sim bank cash burned after setup
12.   Wind-down CB repayment
"""

from decimal import Decimal

from bilancio.decision.profiles import BankProfile
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.banking_subsystem import (
    BankingSubsystem,
    BankLoanRecord,
    _get_bank_reserves,
    initialize_banking_subsystem,
)
from bilancio.engines.simulation import _run_cb_backstop, run_until_stable
from bilancio.engines.system import System
from bilancio.ops.banking import burn_bank_cash, deposit_cash

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_banking_system() -> System:
    """Create a minimal system with CB + 1 bank + 1 firm."""
    system = System(policy=PolicyEngine.default())
    cb = CentralBank(id="cb", name="Central Bank", kind="central_bank")
    bank1 = Bank(id="bank_1", name="Bank 1", kind="bank")
    firm1 = Firm(id="H_1", name="Firm 1", kind="firm")
    system.add_agent(cb)
    system.add_agent(bank1)
    system.add_agent(firm1)
    system.mint_cash(to_agent_id="H_1", amount=1000)
    deposit_cash(system, "H_1", "bank_1", 1000)
    system.mint_reserves(to_bank_id="bank_1", amount=5000)
    return system


def _count_instruments(system: System, agent_id: str, kind: InstrumentKind) -> int:
    """Count instruments of a given kind held by an agent as assets."""
    agent = system.state.agents[agent_id]
    count = 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract and contract.kind == kind:
            count += 1
    return count


def _total_amount(system: System, agent_id: str, kind: InstrumentKind) -> int:
    """Sum amounts of instruments of a given kind held by an agent as assets."""
    agent = system.state.agents[agent_id]
    total = 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract and contract.kind == kind:
            total += contract.amount
    return total


def _event_kinds(system: System) -> list[str]:
    """Extract all event kinds from the system event log."""
    return [e["kind"] for e in system.state.events]


def _make_initialized_banking_subsystem(
    system: System,
    kappa: Decimal = Decimal("1.0"),
    maturity_days: int = 10,
) -> BankingSubsystem:
    """Initialize a BankingSubsystem for the given system."""
    profile = BankProfile()
    return initialize_banking_subsystem(
        system=system,
        bank_profile=profile,
        kappa=kappa,
        maturity_days=maturity_days,
    )


# ===========================================================================
# Change 1: burn_bank_cash
# ===========================================================================


class TestBurnBankCash:
    """Tests for burn_bank_cash (ops/banking.py)."""

    def test_burn_bank_cash_removes_all_cash(self):
        """After deposit_cash, bank has CASH. burn removes it and returns amount."""
        system = _make_banking_system()

        # Verify bank has CASH before burning
        assert _count_instruments(system, "bank_1", InstrumentKind.CASH) > 0
        cash_amount = _total_amount(system, "bank_1", InstrumentKind.CASH)
        assert cash_amount == 1000

        # Burn
        burned = burn_bank_cash(system, "bank_1")

        # Verify
        assert burned == 1000
        assert _count_instruments(system, "bank_1", InstrumentKind.CASH) == 0
        assert "BankCashBurned" in _event_kinds(system)

    def test_burn_bank_cash_preserves_reserves(self):
        """After burning cash, RESERVE_DEPOSIT instruments are still present."""
        system = _make_banking_system()

        reserves_before = _total_amount(system, "bank_1", InstrumentKind.RESERVE_DEPOSIT)
        assert reserves_before == 5000

        burn_bank_cash(system, "bank_1")

        reserves_after = _total_amount(system, "bank_1", InstrumentKind.RESERVE_DEPOSIT)
        assert reserves_after == reserves_before

    def test_burn_bank_cash_noop_no_cash(self):
        """Calling burn on a bank with no cash returns 0, no error."""
        system = _make_banking_system()

        # First burn removes all cash
        burn_bank_cash(system, "bank_1")
        assert _count_instruments(system, "bank_1", InstrumentKind.CASH) == 0

        # Second burn is a noop
        result = burn_bank_cash(system, "bank_1")
        assert result == 0


# ===========================================================================
# Change 2: 10-day reserve projection in refresh_quote
# ===========================================================================


class TestRefreshQuoteProjection:
    """Tests for BankTreynorState.refresh_quote with 10-day projection."""

    def test_refresh_quote_uses_projection(self):
        """When reserves < reserve_floor, loan rate should be higher than when reserves are ample."""
        # System with low reserves
        system_low = _make_banking_system()
        burn_bank_cash(system_low, "bank_1")
        # Start with minimal reserves so the bank is below floor
        # Remove the 5000 reserves and add just 50
        # We can't easily remove reserves, so create two separate systems
        system_high = _make_banking_system()
        burn_bank_cash(system_high, "bank_1")

        # Add more reserves to the high system
        system_high.mint_reserves(to_bank_id="bank_1", amount=50000)

        profile = BankProfile()
        kappa = Decimal("1.0")

        sub_low = initialize_banking_subsystem(
            system=system_low, bank_profile=profile, kappa=kappa, maturity_days=10,
        )
        sub_high = initialize_banking_subsystem(
            system=system_high, bank_profile=profile, kappa=kappa, maturity_days=10,
        )

        quote_low = sub_low.banks["bank_1"].current_quote
        quote_high = sub_high.banks["bank_1"].current_quote

        assert quote_low is not None
        assert quote_high is not None

        # Bank with fewer reserves should have equal or higher loan rate
        assert quote_low.loan_rate >= quote_high.loan_rate, (
            f"Low-reserves loan_rate ({quote_low.loan_rate}) should be >= "
            f"high-reserves loan_rate ({quote_high.loan_rate})"
        )

    def test_refresh_quote_inventory_uses_t2(self):
        """A BankLoanRecord maturing at t+2 should affect inventory via projection path."""
        system = _make_banking_system()
        burn_bank_cash(system, "bank_1")
        profile = BankProfile()
        kappa = Decimal("1.0")

        sub = initialize_banking_subsystem(
            system=system, bank_profile=profile, kappa=kappa, maturity_days=10,
        )

        current_day = system.state.day
        bank_state = sub.banks["bank_1"]

        # Get baseline quote (no outstanding loans)
        quote_no_inflow = bank_state.refresh_quote(system, current_day)

        # Add a BankLoanRecord with maturity at current_day + 2 (inflow at t+2)
        bank_state.outstanding_loans["loan_001"] = BankLoanRecord(
            loan_id="loan_001",
            bank_id="bank_1",
            borrower_id="H_1",
            principal=2000,
            rate=Decimal("0.05"),
            issuance_day=current_day,
            maturity_day=current_day + 2,
        )

        # Refresh with the inflow at t+2
        quote_with_inflow = bank_state.refresh_quote(system, current_day)

        assert quote_no_inflow is not None
        assert quote_with_inflow is not None

        # With an inflow at t+2, projected reserves at t+2 increase,
        # so inventory increases, driving the loan rate down (or equal).
        assert quote_with_inflow.loan_rate <= quote_no_inflow.loan_rate, (
            f"Inflow at t+2 should lower loan_rate: "
            f"with={quote_with_inflow.loan_rate} vs without={quote_no_inflow.loan_rate}"
        )

    def test_refresh_quote_scheduled_legs(self):
        """Adding a BankLoanRecord inflow and a CB loan outflow should produce a valid quote."""
        system = _make_banking_system()
        burn_bank_cash(system, "bank_1")
        profile = BankProfile()
        kappa = Decimal("1.0")

        sub = initialize_banking_subsystem(
            system=system, bank_profile=profile, kappa=kappa, maturity_days=10,
        )

        current_day = system.state.day
        bank_state = sub.banks["bank_1"]

        # Baseline: no scheduled legs
        quote_baseline = bank_state.refresh_quote(system, current_day)

        # Add a BankLoanRecord inflow at t+1
        bank_state.outstanding_loans["loan_100"] = BankLoanRecord(
            loan_id="loan_100",
            bank_id="bank_1",
            borrower_id="H_1",
            principal=1000,
            rate=Decimal("0.05"),
            issuance_day=current_day,
            maturity_day=current_day + 1,
        )

        # Add a CB loan outflow at t+3
        system.cb_lend_reserves("bank_1", 800, current_day)
        # CB loan matures at current_day + 2 (2-day term)

        # Refresh with scheduled legs
        quote_with_legs = bank_state.refresh_quote(system, current_day)

        assert quote_baseline is not None
        assert quote_with_legs is not None
        # The quote should differ from baseline because of the scheduled legs
        # (either the rate or the diagnostic fields should change)
        # At minimum, the quote must be computed successfully (not None)
        assert quote_with_legs.loan_rate is not None
        assert quote_with_legs.deposit_rate is not None


# ===========================================================================
# Change 3: CB end-of-day backstop
# ===========================================================================


class TestCBBackstop:
    """Tests for _run_cb_backstop (simulation.py)."""

    def test_cb_backstop_tops_up_reserves(self):
        """Bank below reserve target gets topped up by CB backstop."""
        system = _make_banking_system()
        burn_bank_cash(system, "bank_1")
        profile = BankProfile()
        kappa = Decimal("1.0")

        sub = initialize_banking_subsystem(
            system=system, bank_profile=profile, kappa=kappa, maturity_days=10,
        )

        bank_state = sub.banks["bank_1"]
        target = bank_state.pricing_params.reserve_target

        # Drain reserves to below target: transfer reserves away
        current_reserves = _get_bank_reserves(system, "bank_1")

        if current_reserves > target:
            # We need to drain reserves. Add a second bank to transfer to.
            bank2 = Bank(id="bank_2", name="Bank 2", kind="bank")
            system.add_agent(bank2)
            system.mint_reserves(to_bank_id="bank_2", amount=1)  # Need at least 1 for bank_2

            drain_amount = current_reserves - target + 100  # Go 100 below target
            if drain_amount > current_reserves:
                drain_amount = current_reserves - 1
            system.transfer_reserves("bank_1", "bank_2", drain_amount)

        reserves_before = _get_bank_reserves(system, "bank_1")
        assert reserves_before < target, (
            f"Expected reserves ({reserves_before}) < target ({target})"
        )

        current_day = system.state.day
        _run_cb_backstop(system, sub, current_day)

        reserves_after = _get_bank_reserves(system, "bank_1")
        assert reserves_after >= target, (
            f"Expected reserves ({reserves_after}) >= target ({target}) after backstop"
        )

        # Verify CBBackstopLoan event was logged
        assert "CBBackstopLoan" in _event_kinds(system)

    def test_cb_backstop_noop_above_target(self):
        """Bank above target: backstop does nothing."""
        system = _make_banking_system()
        burn_bank_cash(system, "bank_1")
        # Add extra reserves to be well above target
        system.mint_reserves(to_bank_id="bank_1", amount=50000)

        profile = BankProfile()
        kappa = Decimal("1.0")

        sub = initialize_banking_subsystem(
            system=system, bank_profile=profile, kappa=kappa, maturity_days=10,
        )

        events_before = len(system.state.events)
        current_day = system.state.day
        _run_cb_backstop(system, sub, current_day)

        # No CBBackstopLoan events should have been added
        new_events = system.state.events[events_before:]
        backstop_events = [e for e in new_events if e["kind"] == "CBBackstopLoan"]
        assert len(backstop_events) == 0

    def test_cb_backstop_skipped_when_frozen(self):
        """When CB lending is frozen, backstop does nothing."""
        system = _make_banking_system()
        burn_bank_cash(system, "bank_1")
        profile = BankProfile()
        kappa = Decimal("1.0")

        sub = initialize_banking_subsystem(
            system=system, bank_profile=profile, kappa=kappa, maturity_days=10,
        )

        # Freeze CB lending
        system.state.cb_lending_frozen = True

        events_before = len(system.state.events)
        current_day = system.state.day
        _run_cb_backstop(system, sub, current_day)

        # No events should have been added
        new_events = system.state.events[events_before:]
        backstop_events = [e for e in new_events if e["kind"] == "CBBackstopLoan"]
        assert len(backstop_events) == 0


# ===========================================================================
# Change 5: CB freeze at stability
# ===========================================================================


class TestCBFreezeAtStability:
    """Tests for CB lending freeze at stability in run_until_stable."""

    def test_cb_freeze_at_stability(self):
        """After run_until_stable with banking, CB lending should be frozen."""
        system = System(policy=PolicyEngine.default())
        cb = CentralBank(id="cb", name="Central Bank", kind="central_bank")
        bank1 = Bank(id="bank_1", name="Bank 1", kind="bank")
        firm1 = Firm(id="H_1", name="Firm 1", kind="firm")
        firm2 = Firm(id="H_2", name="Firm 2", kind="firm")

        system.add_agent(cb)
        system.add_agent(bank1)
        system.add_agent(firm1)
        system.add_agent(firm2)

        # Give firm H_1 cash, deposit it, mint reserves
        system.mint_cash(to_agent_id="H_1", amount=2000)
        system.mint_cash(to_agent_id="H_2", amount=2000)
        deposit_cash(system, "H_1", "bank_1", 2000)
        deposit_cash(system, "H_2", "bank_1", 2000)
        burn_bank_cash(system, "bank_1")
        system.mint_reserves(to_bank_id="bank_1", amount=10000)

        # Initialize banking subsystem
        profile = BankProfile()
        kappa = Decimal("1.0")
        sub = initialize_banking_subsystem(
            system=system, bank_profile=profile, kappa=kappa, maturity_days=10,
        )
        system.state.banking_subsystem = sub

        # Create a simple obligation that will settle quickly
        # (no payables means stability is reached immediately)
        assert not system.state.cb_lending_frozen

        run_until_stable(
            system,
            max_days=20,
            quiet_days=2,
            enable_banking=True,
            enable_bank_lending=True,
        )

        # After stability, CB lending should be frozen
        assert system.state.cb_lending_frozen is True
        assert "CBLendingFreezeStability" in _event_kinds(system)


# ===========================================================================
# Integration tests
# ===========================================================================


class TestIntegration:
    """Integration tests for paper-faithful bank pricing changes."""

    def test_full_sim_bank_cash_burned(self):
        """After burn_bank_cash, banks hold 0 CASH instruments."""
        system = _make_banking_system()

        # Verify bank has cash from deposit
        assert _count_instruments(system, "bank_1", InstrumentKind.CASH) > 0

        # Apply burn_bank_cash (as would happen in setup actions)
        burned = burn_bank_cash(system, "bank_1")
        assert burned > 0

        # Verify bank has 0 CASH instruments
        assert _count_instruments(system, "bank_1", InstrumentKind.CASH) == 0

        # Verify bank still has reserves and deposit liabilities
        assert _total_amount(system, "bank_1", InstrumentKind.RESERVE_DEPOSIT) == 5000

        # Verify the firm's deposit at the bank is unchanged
        dep_ids = system.deposit_ids("H_1", "bank_1")
        total_deposit = sum(
            system.state.contracts[did].amount for did in dep_ids
        )
        assert total_deposit == 1000

    def test_wind_down_cb_repayment(self):
        """Bank with CB loan: wind-down either repays or defaults."""
        system = System(policy=PolicyEngine.default())
        cb = CentralBank(id="cb", name="Central Bank", kind="central_bank")
        bank1 = Bank(id="bank_1", name="Bank 1", kind="bank")
        firm1 = Firm(id="H_1", name="Firm 1", kind="firm")

        system.add_agent(cb)
        system.add_agent(bank1)
        system.add_agent(firm1)

        system.mint_cash(to_agent_id="H_1", amount=1000)
        deposit_cash(system, "H_1", "bank_1", 1000)
        burn_bank_cash(system, "bank_1")

        # Mint enough reserves to handle the CB loan repayment (principal + interest)
        system.mint_reserves(to_bank_id="bank_1", amount=10000)

        # Create a CB loan (matures at day+2)
        current_day = system.state.day
        system.cb_lend_reserves("bank_1", 500, current_day)

        # Verify CB loan exists
        cb_loans = [
            cid for cid, c in system.state.contracts.items()
            if c.kind == InstrumentKind.CB_LOAN
        ]
        assert len(cb_loans) == 1

        # Initialize banking subsystem for wind-down
        profile = BankProfile()
        kappa = Decimal("1.0")
        sub = initialize_banking_subsystem(
            system=system, bank_profile=profile, kappa=kappa, maturity_days=10,
        )
        system.state.banking_subsystem = sub

        # Run the simulation which includes CB loan repayment during run_day
        # Advance to the loan's maturity day
        run_until_stable(
            system,
            max_days=10,
            quiet_days=2,
            enable_banking=True,
        )

        # After the simulation, either the loan was repaid or the bank defaulted
        remaining_cb_loans = [
            cid for cid, c in system.state.contracts.items()
            if c.kind == InstrumentKind.CB_LOAN
        ]

        event_kinds = _event_kinds(system)
        loan_repaid = "CBLoanRepaid" in event_kinds
        loan_written_off = "CBFinalSettlementWrittenOff" in event_kinds
        bank_defaulted = "BankDefaultWinddown" in event_kinds or "BankDefaultCBFreeze" in event_kinds

        # Either the loan was repaid successfully, written off in final settlement, or bank defaulted
        assert loan_repaid or loan_written_off or bank_defaulted or len(remaining_cb_loans) == 0, (
            f"Expected CB loan to be resolved. "
            f"Remaining CB loans: {remaining_cb_loans}, events: {event_kinds}"
        )
