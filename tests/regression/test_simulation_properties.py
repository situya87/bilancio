"""End-to-end simulation regression tests that verify key properties.

These tests run full simulations and check structural properties:
- Lending creates loans when shortfalls exist
- Trading effect is bounded
- Defaults increase with lower liquidity (kappa)
- System invariants hold after simulation

All tests use fixed seeds for reproducibility and small rings for speed.
"""

from decimal import Decimal

import pytest

from bilancio.dealer.models import DEFAULT_BUCKETS
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.domain.agents import Bank, CentralBank, Household
from bilancio.domain.agents.non_bank_lender import NonBankLender
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.dealer_integration import initialize_dealer_subsystem
from bilancio.engines.lending import LendingConfig
from bilancio.engines.simulation import run_until_stable
from bilancio.engines.system import System

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_ring_system(
    n_agents=20,
    cash_per_agent=50,
    payable_amount=100,
    maturity_days=5,
    seed=42,
    default_mode="expel-agent",
):
    """Build a ring of n Household agents with payables.

    Topology: H1->H2->...->HN->H1 (each agent owes the next).
    Each agent starts with ``cash_per_agent`` units of cash and owes
    ``payable_amount`` to the next agent in the ring.

    The ``default_mode`` defaults to ``"expel-agent"`` so that defaults
    are handled gracefully (agents are expelled) rather than raising
    ``DefaultError``.
    """
    system = System(default_mode=default_mode)
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    bank = Bank(id="B1", name="Bank 1", kind="bank")
    system.add_agent(cb)
    system.add_agent(bank)

    for i in range(1, n_agents + 1):
        h = Household(id=f"H{i}", name=f"Household {i}", kind="household")
        system.add_agent(h)
        system.mint_cash(f"H{i}", cash_per_agent)

    for i in range(1, n_agents + 1):
        from_id = f"H{i}"
        to_id = f"H{(i % n_agents) + 1}"
        due_day = 1 + ((i - 1) % maturity_days)
        p = Payable(
            id=system.new_contract_id("P"),
            kind=InstrumentKind.PAYABLE,
            amount=payable_amount,
            denom="X",
            asset_holder_id=to_id,
            liability_issuer_id=from_id,
            due_day=due_day,
        )
        system.add_contract(p)

    return system


def count_defaults(system):
    """Count distinct agents that defaulted during the simulation."""
    return len(system.state.defaulted_agent_ids)


def count_default_events(system):
    """Count ObligationDefaulted / ObligationWrittenOff / AgentDefaulted events."""
    default_kinds = {"ObligationDefaulted", "ObligationWrittenOff", "AgentDefaulted"}
    return sum(1 for e in system.state.events if e.get("kind") in default_kinds)


def count_events(system, kind):
    """Count events of a specific kind."""
    return sum(1 for e in system.state.events if e.get("kind") == kind)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_lending_effect_is_nonzero():
    """NBFI lending must create at least one loan when agents have shortfalls.

    Setup: kappa < 1 (cash=80 < payable=100) so agents have shortfalls.
    A NonBankLender with ample cash is added. After running the simulation
    with lending enabled, at least one NonBankLoanCreated event must exist.
    """
    sys = build_ring_system(
        n_agents=20,
        cash_per_agent=80,
        payable_amount=100,
        maturity_days=5,
        seed=42,
    )

    # Add a well-capitalised non-bank lender
    lender = NonBankLender(id="NBFI1", name="Non-Bank Lender")
    sys.add_agent(lender)
    sys.mint_cash("NBFI1", 5000)

    # Configure lending
    sys.state.lender_config = LendingConfig(
        base_rate=Decimal("0.05"),
        risk_premium_scale=Decimal("0.20"),
        max_single_exposure=Decimal("0.15"),
        max_total_exposure=Decimal("0.80"),
        maturity_days=2,
        horizon=3,
        min_shortfall=1,
    )

    run_until_stable(sys, max_days=10, enable_lender=True)

    loan_events = [e for e in sys.state.events if e.get("kind") == "NonBankLoanCreated"]
    assert len(loan_events) >= 1, (
        f"Expected at least 1 NonBankLoanCreated event, got {len(loan_events)}. "
        f"Event kinds: {sorted({e.get('kind') for e in sys.state.events})}"
    )


@pytest.mark.regression
def test_trading_effect_is_reasonable():
    """Active vs passive: trading effect must be bounded in [-0.5, +0.5].

    Runs two identical rings (same seed) -- one passive (no dealer), one
    active (dealer enabled). The difference in the fraction of defaulted
    agents must be within a reasonable bound.
    """
    n = 20
    maturity = 5

    # --- Passive run (no dealer) ---
    sys_passive = build_ring_system(
        n_agents=n,
        cash_per_agent=50,
        payable_amount=100,
        maturity_days=maturity,
        seed=42,
    )
    run_until_stable(sys_passive, max_days=15)
    defaults_passive = count_defaults(sys_passive)

    # --- Active run (dealer enabled) ---
    sys_active = build_ring_system(
        n_agents=n,
        cash_per_agent=50,
        payable_amount=100,
        maturity_days=maturity,
        seed=42,
    )
    dealer_config = DealerRingConfig(
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
    subsystem = initialize_dealer_subsystem(sys_active, dealer_config, current_day=0)
    sys_active.state.dealer_subsystem = subsystem
    run_until_stable(sys_active, max_days=15, enable_dealer=True)
    defaults_active = count_defaults(sys_active)

    # Trading effect: difference in default fractions
    trading_effect = (defaults_passive - defaults_active) / n
    assert -0.5 <= trading_effect <= 0.5, (
        f"Trading effect {trading_effect:.3f} out of bounds. "
        f"Passive defaults={defaults_passive}, active defaults={defaults_active}"
    )


@pytest.mark.regression
def test_defaults_increase_with_lower_kappa():
    """Lower kappa (liquidity ratio) must produce >= defaults than higher kappa.

    kappa = cash_per_agent / payable_amount.
    - kappa=0.3 (cash=30, payable=100): high stress
    - kappa=2.0 (cash=200, payable=100): low stress

    The number of defaulted agents under high stress must be >= the number
    under low stress.
    """
    payable = 100

    # High stress: kappa = 0.3
    sys_low_kappa = build_ring_system(
        n_agents=20,
        cash_per_agent=30,
        payable_amount=payable,
        maturity_days=5,
        seed=42,
    )
    run_until_stable(sys_low_kappa, max_days=15)
    defaults_low_kappa = count_defaults(sys_low_kappa)

    # Low stress: kappa = 2.0
    sys_high_kappa = build_ring_system(
        n_agents=20,
        cash_per_agent=200,
        payable_amount=payable,
        maturity_days=5,
        seed=42,
    )
    run_until_stable(sys_high_kappa, max_days=15)
    defaults_high_kappa = count_defaults(sys_high_kappa)

    assert defaults_low_kappa >= defaults_high_kappa, (
        f"Monotonicity violated: kappa=0.3 had {defaults_low_kappa} defaults "
        f"but kappa=2.0 had {defaults_high_kappa} defaults"
    )
    # Also verify high kappa actually has zero or very few defaults
    assert defaults_high_kappa <= 2, (
        f"kappa=2.0 should have very few defaults, got {defaults_high_kappa}"
    )


@pytest.mark.regression
def test_system_invariants_after_simulation():
    """After a full simulation run, system invariants must hold.

    Checks:
    - system.assert_invariants() passes (double-entry, no negative balances, etc.)
    - No agent has negative cash
    """
    sys = build_ring_system(
        n_agents=20,
        cash_per_agent=100,
        payable_amount=100,
        maturity_days=5,
        seed=42,
    )
    run_until_stable(sys, max_days=15)

    # System invariants must hold
    sys.assert_invariants()

    # No agent should have negative cash
    for agent_id, agent in sys.state.agents.items():
        total_cash = sum(
            sys.state.contracts[cid].amount
            for cid in agent.asset_ids
            if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH
        )
        assert total_cash >= 0, f"Agent {agent_id} has negative cash: {total_cash}"


@pytest.mark.regression
def test_nbfi_creates_loans_when_shortfalls_exist():
    """NBFI scenario with severe shortfalls must produce loans.

    Uses a more stressed scenario (n=10, cash=20, payable=100) than
    test_lending_effect_is_nonzero to verify lending under extreme stress.
    """
    sys = build_ring_system(
        n_agents=10,
        cash_per_agent=20,
        payable_amount=100,
        maturity_days=5,
        seed=42,
    )

    lender = NonBankLender(id="NBFI1", name="Non-Bank Lender")
    sys.add_agent(lender)
    sys.mint_cash("NBFI1", 5000)

    sys.state.lender_config = LendingConfig(
        base_rate=Decimal("0.05"),
        risk_premium_scale=Decimal("0.20"),
        max_single_exposure=Decimal("0.20"),
        max_total_exposure=Decimal("0.90"),
        maturity_days=3,
        horizon=5,
        min_shortfall=1,
    )

    run_until_stable(sys, max_days=10, enable_lender=True)

    loan_events = [e for e in sys.state.events if e.get("kind") == "NonBankLoanCreated"]
    assert len(loan_events) >= 1, (
        f"Expected at least 1 NonBankLoanCreated event under extreme stress, "
        f"got {len(loan_events)}. "
        f"Event kinds: {sorted({e.get('kind') for e in sys.state.events})}"
    )
