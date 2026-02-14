"""Integration tests for the lending -> dealer -> settlement pipeline.

These tests verify that the non-bank lending phase and the dealer sync phase
do not interfere destructively. In particular, they guard against the critical
bug where the dealer sync phase (when the dealer subsystem is disabled) would
read stale ``trader.cash`` values and burn cash that had been injected by the
lending phase.

The fix (moving ``_sync_trader_cash_from_system`` BEFORE the early-return in
``run_dealer_trading_phase``) is in place. These tests exist to prevent
regression.

Key interaction tested:
    1. Lending phase gives a firm/household cash via ``nonbank_lend_cash``.
    2. Dealer trading phase runs (possibly with ``subsystem.enabled = False``).
    3. ``sync_dealer_to_system`` must not reverse the lending cash injection.

References:
    - Fix location: ``dealer_integration.py:543`` (sync before disabled check)
    - MEMORY.md: "moved _sync_trader_cash_from_system() BEFORE the
      if not subsystem.enabled: return [] check"
"""

import pytest
from decimal import Decimal

from bilancio.engines.system import System
from bilancio.domain.agents import Bank, Household, CentralBank
from bilancio.domain.agents.non_bank_lender import NonBankLender
from bilancio.domain.agents.firm import Firm
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.simulation import run_day
from bilancio.engines.dealer_integration import (
    initialize_dealer_subsystem,
    _get_agent_cash,
)
from bilancio.engines.lending import LendingConfig
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.dealer.models import DEFAULT_BUCKETS


def _make_dealer_config() -> DealerRingConfig:
    """Standard dealer configuration for tests."""
    return DealerRingConfig(
        ticket_size=Decimal(1),
        buckets=list(DEFAULT_BUCKETS),
        dealer_share=Decimal("0.25"),
        vbt_share=Decimal("0.50"),
        vbt_anchors={
            "short": (Decimal("1.0"), Decimal("0.20")),
            "mid": (Decimal("1.0"), Decimal("0.30")),
            "long": (Decimal("1.0"), Decimal("0.40")),
        },
        phi_M=Decimal("0.1"),
        phi_O=Decimal("0.1"),
        clip_nonneg_B=True,
        seed=42,
    )


def _total_agent_cash(system: System, agent_id: str) -> Decimal:
    """Return total cash for an agent using the dealer_integration helper."""
    return _get_agent_cash(system, agent_id)


def _total_system_cash(system: System) -> Decimal:
    """Sum all CASH instruments across all agents in the system."""
    total = Decimal(0)
    for contract in system.state.contracts.values():
        if contract.kind == InstrumentKind.CASH:
            total += Decimal(contract.amount)
    return total


def test_lending_then_disabled_dealer_preserves_cash():
    """Cash from the lending phase must survive dealer sync when dealer is disabled.

    Scenario:
        - CentralBank, Bank, NonBankLender (cash=10000), Firm1 (cash=200,
          owes 1000 to Firm2 due in 2 days), Firm2 (cash=100).
        - Several Household agents with inter-household payables so the dealer
          subsystem can create traders from them.
        - Dealer subsystem initialized but disabled (subsystem.enabled=False).
        - LendingConfig attached to system.
        - run_day with enable_dealer=True, enable_lender=True.

    Assertions:
        - Firm1's cash after run_day >= 200 (lending should add, not remove).
        - Total system cash is conserved (lending is a transfer, not creation).
        - If a NonBankLoanCreated event was emitted, Firm1 must have more cash
          than before (minus any settlements it may have paid).
    """
    sys = System()

    # Central bank (required for minting)
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    sys.bootstrap_cb(cb)

    # Bank (required for system structure)
    bank = Bank(id="B1", name="Bank 1", kind="bank")
    sys.add_agent(bank)

    # Non-bank lender with ample cash
    nbl = NonBankLender(id="NBL01", name="Non-Bank Lender")
    sys.add_agent(nbl)
    sys.mint_cash("NBL01", 10000)

    # Firm1: has limited cash, owes a large amount soon
    f1 = Firm(id="F01", name="Firm 1", kind="firm")
    sys.add_agent(f1)
    sys.mint_cash("F01", 200)

    # Firm2: receives the payable
    f2 = Firm(id="F02", name="Firm 2", kind="firm")
    sys.add_agent(f2)
    sys.mint_cash("F02", 100)

    # Create a large payable: F01 owes F02 1000, due in 2 days
    p_id = sys.new_contract_id("P")
    payable = Payable(
        id=p_id,
        kind=InstrumentKind.PAYABLE,
        amount=1000,
        denom="X",
        asset_holder_id="F02",
        liability_issuer_id="F01",
        due_day=sys.state.day + 2,
    )
    sys.add_contract(payable)

    # Households with inter-household payables (needed for dealer subsystem traders)
    h1 = Household(id="H01", name="Household 1", kind="household")
    h2 = Household(id="H02", name="Household 2", kind="household")
    h3 = Household(id="H03", name="Household 3", kind="household")
    sys.add_agent(h1)
    sys.add_agent(h2)
    sys.add_agent(h3)
    sys.mint_cash("H01", 50)
    sys.mint_cash("H02", 50)
    sys.mint_cash("H03", 50)

    # Create household payables so the dealer has traders with tickets
    for (src, dst, due) in [("H01", "H02", 3), ("H02", "H03", 5), ("H03", "H01", 8)]:
        pid = sys.new_contract_id("P")
        p = Payable(
            id=pid,
            kind=InstrumentKind.PAYABLE,
            amount=30,
            denom="X",
            asset_holder_id=dst,
            liability_issuer_id=src,
            due_day=sys.state.day + due,
        )
        sys.add_contract(p)

    # Set up lending config
    sys.state.lender_config = LendingConfig()

    # Initialize dealer subsystem (disabled)
    config = _make_dealer_config()
    subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
    sys.state.dealer_subsystem = subsystem
    subsystem.enabled = False

    # Record baseline
    f01_cash_before = _total_agent_cash(sys, "F01")
    nbl_cash_before = _total_agent_cash(sys, "NBL01")
    total_cash_before = _total_system_cash(sys)

    assert f01_cash_before == Decimal(200), "F01 should start with 200 cash"

    # Run one day
    run_day(sys, enable_dealer=True, enable_lender=True)

    # Record after state
    f01_cash_after = _total_agent_cash(sys, "F01")
    nbl_cash_after = _total_agent_cash(sys, "NBL01")
    total_cash_after = _total_system_cash(sys)

    # Check for lending events
    loan_events = [
        e for e in sys.state.events if e.get("kind") == "NonBankLoanCreated"
    ]

    if loan_events:
        # Lending happened: F01 must have gained cash (before any settlement)
        # The loan amount was added; F01 may have also paid settlements on day 0
        # but the payable is due on day 2, so no settlement should occur yet.
        assert f01_cash_after > f01_cash_before, (
            f"F01 received a loan but cash did not increase: "
            f"before={f01_cash_before}, after={f01_cash_after}"
        )
        # Lender should have less cash (transferred to borrower)
        assert nbl_cash_after < nbl_cash_before, (
            "NonBankLender cash should decrease after lending"
        )
    else:
        # Lending did not happen (possibly no shortfall detected).
        # Key assertion: cash must NOT have decreased due to dealer sync bug.
        assert f01_cash_after >= f01_cash_before, (
            f"F01 cash decreased without lending or settlement: "
            f"before={f01_cash_before}, after={f01_cash_after}. "
            "This suggests the dealer sync bug is reintroduced."
        )

    # Total system cash must be conserved.
    # Lending transfers existing cash (no new creation beyond CB operations).
    # The only source of new cash is CB minting during run_day (e.g., reserve interest).
    # Allow for small CB corridor adjustments.
    assert total_cash_after == total_cash_before, (
        f"Total system cash changed unexpectedly: "
        f"before={total_cash_before}, after={total_cash_after}"
    )


def test_lending_effect_is_nonzero_in_nbfi_mode():
    """At least one loan must be created in a 5-day simulation with NBFI lending.

    Scenario:
        - 5 Household agents in a ring, each owing ~100 to the next.
        - Each household has only 30-50 cash (kappa < 1).
        - A NonBankLender with 2000 cash provides liquidity.
        - Dealer subsystem is initialized but disabled (NBFI-only mode).
        - Run 5 days with enable_lender=True, enable_dealer=True.

    Assertions:
        - At least one NonBankLoanCreated event is emitted.
        - System invariants hold at the end.
    """
    sys = System()

    # Central bank
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    sys.bootstrap_cb(cb)

    # Bank (required for system structure)
    bank = Bank(id="B1", name="Bank 1", kind="bank")
    sys.add_agent(bank)

    # Non-bank lender with substantial cash
    nbl = NonBankLender(id="NBL01", name="Non-Bank Lender")
    sys.add_agent(nbl)
    sys.mint_cash("NBL01", 2000)

    # 5 Households in a ring with limited cash
    household_ids = [f"H{i:02d}" for i in range(1, 6)]
    cash_amounts = [30, 40, 35, 50, 45]  # All well below the 100 obligation

    for hid, cash_amt in zip(household_ids, cash_amounts):
        h = Household(id=hid, name=f"Household {hid}", kind="household")
        sys.add_agent(h)
        sys.mint_cash(hid, cash_amt)

    # Ring payables: H01->H02->H03->H04->H05->H01
    # Due days spread across 5 days to ensure lending opportunities
    for i in range(5):
        src = household_ids[i]
        dst = household_ids[(i + 1) % 5]
        due = i + 1  # Due on day 1, 2, 3, 4, 5

        pid = sys.new_contract_id("P")
        payable = Payable(
            id=pid,
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id=dst,
            liability_issuer_id=src,
            due_day=due,
        )
        sys.add_contract(payable)

    # Set up lending config with a short horizon so shortfalls are detected early
    sys.state.lender_config = LendingConfig(
        horizon=3,
        min_shortfall=1,
        maturity_days=2,
    )

    # Initialize dealer subsystem (disabled -- NBFI-only mode)
    config = _make_dealer_config()
    subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
    sys.state.dealer_subsystem = subsystem
    subsystem.enabled = False

    # Record initial total cash
    total_cash_before = _total_system_cash(sys)

    # Run 5 days
    for _ in range(5):
        run_day(sys, enable_dealer=True, enable_lender=True)

    # Check for at least one lending event
    loan_events = [
        e for e in sys.state.events if e.get("kind") == "NonBankLoanCreated"
    ]
    assert len(loan_events) >= 1, (
        "Expected at least one NonBankLoanCreated event in 5 days of "
        "simulation with cash-constrained agents and an available lender. "
        f"Events: {[e.get('kind') for e in sys.state.events]}"
    )

    # Verify system invariants hold
    sys.assert_invariants()

    # Verify total cash is conserved (lending is a transfer, not creation).
    # CB corridor may add/remove reserves, but for this test no CB lending
    # is triggered. Allow for any CB-level adjustments.
    total_cash_after = _total_system_cash(sys)
    assert total_cash_after == total_cash_before, (
        f"Total system cash changed: before={total_cash_before}, "
        f"after={total_cash_after}. Lending should be a transfer, "
        "not creation/destruction."
    )
