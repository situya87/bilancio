"""Tests that total system cash is conserved across simulation phases.

Cash is a zero-sum instrument within the system: it can only increase via
explicit ``mint_cash`` calls (CB issuance) and decrease via ``retire_cash``.
Every other operation — settlement, transfer, lending, dealer sync — merely
redistributes existing cash among agents.  These tests verify that invariant
by measuring total system cash before and after each phase and asserting
equality.
"""

from decimal import Decimal

from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.agents.household import Household
from bilancio.domain.agents.non_bank_lender import NonBankLender
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.dealer_integration import (
    initialize_dealer_subsystem,
    run_dealer_trading_phase,
    sync_dealer_to_system,
)
from bilancio.engines.lending import LendingConfig, run_lending_phase
from bilancio.engines.simulation import run_day
from bilancio.engines.system import System
from tests.conftest import create_dealer_config, create_test_system_with_payables

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def total_system_cash(system: System) -> Decimal:
    """Sum all CASH instruments across every agent in the system."""
    total = Decimal(0)
    for agent in system.state.agents.values():
        for cid in agent.asset_ids:
            contract = system.state.contracts.get(cid)
            if contract and contract.kind == InstrumentKind.CASH:
                total += Decimal(contract.amount)
    return total


def create_lending_system() -> System:
    """Create a system suitable for lending-phase tests.

    Topology:
        - NonBankLender (NBL01) holds 10 000 in cash.
        - Firm F01 holds 500 in cash but owes F02 1 000 due in 2 days
          (shortfall = 500).
        - Bank B1 and CentralBank CB1 are present for policy compliance.

    Total system cash = 10 500.
    """
    system = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    bank = Bank(id="B1", name="Bank 1", kind="bank")
    lender = NonBankLender(id="NBL01", name="Non-Bank Lender")
    firm1 = Firm(id="F01", name="Firm 1", kind="firm")
    firm2 = Firm(id="F02", name="Firm 2", kind="firm")
    system.bootstrap_cb(cb)
    system.add_agent(bank)
    system.add_agent(lender)
    system.add_agent(firm1)
    system.add_agent(firm2)
    system.mint_cash("NBL01", 10000)
    system.mint_cash("F01", 500)

    # F01 owes F02 1000 due in 2 days (shortfall = 500)
    p = Payable(
        id=system.new_contract_id("PAY"),
        kind=InstrumentKind.PAYABLE,
        amount=1000,
        denom="X",
        asset_holder_id="F02",
        liability_issuer_id="F01",
        due_day=2,
    )
    system.add_contract(p)

    return system


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCashConservation:
    """Verify that total system cash is unchanged by redistribution phases."""

    def test_total_cash_preserved_across_dealer_phase(self):
        """Dealer trading phase + sync must not alter total cash.

        When the dealer subsystem is disabled (no trades executed),
        running ``run_dealer_trading_phase`` followed by
        ``sync_dealer_to_system`` must leave total cash unchanged.
        """
        sys = create_test_system_with_payables()
        config = create_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

        # Disable the subsystem so no trades happen
        subsystem.enabled = False

        cash_before = total_system_cash(sys)

        events = run_dealer_trading_phase(subsystem, sys, current_day=0)
        sync_dealer_to_system(subsystem, sys)

        cash_after = total_system_cash(sys)

        assert events == [], "Disabled subsystem should produce no trade events"
        assert cash_after == cash_before, (
            f"Total cash changed across disabled dealer phase: "
            f"before={cash_before}, after={cash_after}"
        )

    def test_total_cash_preserved_across_lending_phase(self):
        """Lending redistributes cash (lender -> borrower) but conserves total.

        The lending engine transfers existing cash from the lender to the
        borrower and creates a NonBankLoan instrument.  No new cash is
        minted; total system cash must remain unchanged.
        """
        system = create_lending_system()
        cash_before = total_system_cash(system)

        lending_config = LendingConfig(
            base_rate=Decimal("0.05"),
            risk_premium_scale=Decimal("0.20"),
            max_single_exposure=Decimal("0.15"),
            max_total_exposure=Decimal("0.80"),
            maturity_days=2,
            horizon=3,
            min_shortfall=1,
            max_default_prob=Decimal("0.50"),
        )

        events = run_lending_phase(system, current_day=0, lending_config=lending_config)
        cash_after = total_system_cash(system)

        assert cash_after == cash_before, (
            f"Total cash changed across lending phase: "
            f"before={cash_before}, after={cash_after}, "
            f"lending_events={len(events)}"
        )

    def test_total_cash_preserved_across_settlement(self):
        """Settlement transfers cash from debtor to creditor; total is unchanged.

        When a debtor has sufficient cash to settle a payable due today,
        ``run_day`` moves cash between agents but must not create or
        destroy any.
        """
        sys = System()
        cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.add_agent(h2)
        sys.mint_cash("H1", 100)
        sys.mint_cash("H2", 100)

        # H1 owes H2 50 due TODAY (day 0)
        p = Payable(
            id=sys.new_contract_id("P"),
            kind=InstrumentKind.PAYABLE,
            amount=50,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=0,
        )
        sys.add_contract(p)

        cash_before = total_system_cash(sys)

        # Run one full day — settlement happens in SubphaseB2
        run_day(sys, enable_dealer=False, enable_lender=False)

        cash_after = total_system_cash(sys)

        assert cash_after == cash_before, (
            f"Total cash changed across settlement day: before={cash_before}, after={cash_after}"
        )

    def test_cash_conservation_full_day(self):
        """Full run_day with dealer enabled (but disabled subsystem) conserves cash.

        This is an integration-level check: run_day orchestrates scheduled
        actions, dealer trading, settlement, and clearing.  With no
        defaults and no minting, total cash must be preserved end-to-end.
        """
        sys = System()
        cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
        bank = Bank(id="B1", name="Bank 1", kind="bank")
        h1 = Household(id="H1", name="Household 1", kind="household")
        h2 = Household(id="H2", name="Household 2", kind="household")
        h3 = Household(id="H3", name="Household 3", kind="household")
        sys.add_agent(cb)
        sys.add_agent(bank)
        sys.add_agent(h1)
        sys.add_agent(h2)
        sys.add_agent(h3)

        # Give each household enough cash to cover their obligations
        sys.mint_cash("H1", 200)
        sys.mint_cash("H2", 200)
        sys.mint_cash("H3", 200)

        # Payable due TODAY so it settles this day — H1 owes H2 50
        p = Payable(
            id=sys.new_contract_id("P"),
            kind=InstrumentKind.PAYABLE,
            amount=50,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=0,
        )
        sys.add_contract(p)

        # Initialize and attach dealer subsystem (disabled — no trades)
        config = create_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
        subsystem.enabled = False
        sys.state.dealer_subsystem = subsystem

        cash_before = total_system_cash(sys)

        # Run a full day with dealer enabled (subsystem is disabled so
        # the dealer phase runs but produces no trades)
        run_day(sys, enable_dealer=True, enable_lender=False)

        cash_after = total_system_cash(sys)

        assert cash_after == cash_before, (
            f"Total cash changed across full day with dealer: "
            f"before={cash_before}, after={cash_after}"
        )
