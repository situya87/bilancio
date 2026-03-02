"""End-to-end simulation regression tests that verify key properties.

These tests run full simulations and check structural properties:
- Lending creates loans when shortfalls exist
- Trading effect is bounded
- Defaults increase with lower liquidity (kappa)
- System invariants hold after simulation
- Banking arms restrict lending at low kappa (Plan 045)
- CB backstop stays bounded (Plan 045)
- Settlement forecasts detect cross-bank outflows (Plan 045)

All tests use fixed seeds for reproducibility and small rings for speed.
"""

from decimal import Decimal

import pytest

from bilancio.dealer.models import DEFAULT_BUCKETS
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.decision.profiles import BankProfile
from bilancio.domain.agents import Bank, CentralBank, Household
from bilancio.domain.agents.non_bank_lender import NonBankLender
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.banking_subsystem import initialize_banking_subsystem
from bilancio.engines.dealer_integration import initialize_dealer_subsystem
from bilancio.engines.lending import LendingConfig
from bilancio.engines.simulation import run_until_stable
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash

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


def build_banking_ring_system(
    n_agents=20,
    cash_per_agent=50,
    payable_amount=100,
    maturity_days=5,
    seed=42,
    n_banks=2,
    kappa=Decimal("0.5"),
    credit_risk_loading=Decimal("0.5"),
    max_borrower_risk=Decimal("0.4"),
    reserve_multiplier=10,
    default_mode="expel-agent",
):
    """Build a ring with N banks and households deposited round-robin.

    Creates the full banking subsystem: banks hold reserves, households
    deposit cash into assigned banks, and the BankingSubsystem is
    initialized with settlement-aware reserve projection.

    ``reserve_multiplier`` controls bank capitalization: reserves =
    multiplier * total_deposits.  Use low values (1-2) to stress banks.
    """
    system = System(default_mode=default_mode)

    # Central bank with escalation config
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    cb.rate_escalation_slope = Decimal("0.05")
    cb.max_outstanding_ratio = Decimal("2.0")
    cb.escalation_base_amount = n_agents * payable_amount
    system.add_agent(cb)

    # Create banks
    bank_ids = []
    for b in range(1, n_banks + 1):
        bank = Bank(id=f"B{b}", name=f"Bank {b}", kind="bank")
        system.add_agent(bank)
        bank_ids.append(f"B{b}")

    # Create households in ring, deposit cash into banks round-robin
    trader_banks: dict[str, list[str]] = {}
    for i in range(1, n_agents + 1):
        h = Household(id=f"H{i}", name=f"Household {i}", kind="household")
        system.add_agent(h)
        system.mint_cash(f"H{i}", cash_per_agent)

        assigned_bank = bank_ids[(i - 1) % n_banks]
        trader_banks[f"H{i}"] = [assigned_bank]
        deposit_cash(system, f"H{i}", assigned_bank, cash_per_agent)

    # Mint reserves per bank: reserve_multiplier * total deposits at that bank
    for bank_id in bank_ids:
        # Count deposits at this bank
        total_deposits = 0
        for cid in system.state.agents[bank_id].liability_ids:
            c = system.state.contracts.get(cid)
            if c and c.kind == InstrumentKind.BANK_DEPOSIT:
                total_deposits += c.amount
        reserves = reserve_multiplier * total_deposits
        system.mint_reserves(bank_id, reserves)

    # Create payable ring: H1->H2->...->HN->H1
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

    # Initialize banking subsystem
    profile = BankProfile(
        credit_risk_loading=credit_risk_loading,
        max_borrower_risk=max_borrower_risk,
    )
    subsystem = initialize_banking_subsystem(
        system,
        bank_profile=profile,
        kappa=kappa,
        maturity_days=maturity_days,
        trader_banks=trader_banks,
    )
    system.state.banking_subsystem = subsystem

    return system


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


# ---------------------------------------------------------------------------
# Banking regression tests (Plan 045 invariants)
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_bank_lending_restricted_at_low_kappa():
    """At low kappa with tight bank reserves, banks must restrict lending.

    Plan 045 fixed a bug where banks lent freely on day 0 ignoring
    settlement drain.  With settlement-aware reserve projection, banks
    must restrict lending when reserves are insufficient to cover both
    loans and upcoming settlement outflows.

    Uses reserve_multiplier=1 so bank reserves are tight enough that
    the projected settlement drain actually constrains lending.
    """
    n = 20
    sys = build_banking_ring_system(
        n_agents=n,
        cash_per_agent=30,
        payable_amount=100,
        maturity_days=5,
        kappa=Decimal("0.3"),
        reserve_multiplier=1,
    )

    run_until_stable(sys, max_days=15, enable_banking=True, enable_bank_lending=True)

    loan_count = count_events(sys, "BankLoanIssued")
    assert loan_count < n, (
        f"Bank should restrict lending at kappa=0.3 with tight reserves, "
        f"but issued {loan_count} loans to {n} agents (expected fewer than {n})"
    )

    # Low kappa means some agents must default
    defaults = count_defaults(sys)
    assert defaults >= 1, (
        f"Expected at least 1 default at kappa=0.3, got {defaults}"
    )

    # System invariants must still hold
    sys.assert_invariants()


@pytest.mark.regression
def test_cb_backstop_bounded_at_low_kappa():
    """CB backstop usage must stay bounded even under stress.

    Pre-Plan-045, the CB issued a backstop loan for nearly every
    settlement failure because banks depleted reserves on day-0 lending.
    With the fix, CB loan count must stay proportional to system size,
    not explode.
    """
    n = 20
    sys = build_banking_ring_system(
        n_agents=n,
        cash_per_agent=30,
        payable_amount=100,
        maturity_days=5,
        kappa=Decimal("0.3"),
        reserve_multiplier=1,
    )

    run_until_stable(sys, max_days=15, enable_banking=True, enable_bank_lending=True)

    cb_loan_count = sys.state.cb_loans_created_count
    assert cb_loan_count <= n, (
        f"CB backstop created {cb_loan_count} loans for {n} agents — "
        f"should be bounded by system size (max {n})"
    )


@pytest.mark.regression
def test_settlement_forecast_nonzero_with_cross_bank_payables():
    """Settlement forecasts must detect non-zero outflows when payables
    cross bank boundaries.

    Uses 15 agents with 2 banks so that the number of cross-bank
    payables per due-day is odd (3), breaking the symmetry that would
    otherwise make net flows cancel to zero.
    """
    sys = build_banking_ring_system(
        n_agents=15,
        cash_per_agent=50,
        payable_amount=100,
        maturity_days=5,
        kappa=Decimal("0.5"),
    )

    subsystem = sys.state.banking_subsystem

    # Day 1: some payables are due — forecast should show cross-bank flows
    forecasts = subsystem.compute_settlement_forecasts(sys, current_day=1)
    has_nonzero = any(abs(v) > 0 for v in forecasts.values())
    assert has_nonzero, (
        f"Settlement forecast returned all zeros with 2 banks and "
        f"cross-bank payables: {forecasts}"
    )

    # After refreshing quotes, at least one bank should have projected
    # reserves below its initial level (settlement drain reduces the path)
    initial_reserves = {}
    for bank_id, bank_state in subsystem.banks.items():
        agent = sys.state.agents[bank_id]
        initial_reserves[bank_id] = sum(
            sys.state.contracts[cid].amount
            for cid in agent.asset_ids
            if cid in sys.state.contracts
            and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
        )

    subsystem.refresh_all_quotes(sys, current_day=1)

    has_lower_projection = any(
        subsystem.banks[bid].min_projected_reserves < initial_reserves[bid]
        for bid in subsystem.banks
    )
    assert has_lower_projection, (
        f"No bank has min_projected_reserves < initial reserves after "
        f"settlement forecast. Projections: "
        f"{({bid: subsystem.banks[bid].min_projected_reserves for bid in subsystem.banks})}, "
        f"Initial: {initial_reserves}"
    )


@pytest.mark.regression
def test_banking_differs_from_passive():
    """Banking arms must produce measurably different results from passive.

    Runs two identical rings (same seed) — one passive, one with banking.
    The banking run must issue at least one bank loan, and the outcomes
    (defaults or CB usage) must differ.
    """
    n = 20
    cash = 50
    payable = 100
    maturity = 5

    # --- Passive run ---
    sys_passive = build_ring_system(
        n_agents=n,
        cash_per_agent=cash,
        payable_amount=payable,
        maturity_days=maturity,
        seed=42,
    )
    run_until_stable(sys_passive, max_days=15)
    defaults_passive = count_defaults(sys_passive)

    # --- Banking run ---
    sys_banking = build_banking_ring_system(
        n_agents=n,
        cash_per_agent=cash,
        payable_amount=payable,
        maturity_days=maturity,
        kappa=Decimal("0.5"),
    )
    run_until_stable(
        sys_banking, max_days=15, enable_banking=True, enable_bank_lending=True
    )
    defaults_banking = count_defaults(sys_banking)

    # Banking run must have at least 1 loan issued
    bank_loans = count_events(sys_banking, "BankLoanIssued")
    assert bank_loans >= 1, (
        f"Banking run produced zero BankLoanIssued events — banks did nothing. "
        f"Event kinds: {sorted({e.get('kind') for e in sys_banking.state.events})}"
    )

    # Outcomes must differ: either defaults differ or CB usage differs
    cb_loans_passive = count_events(sys_passive, "CBLoanCreated")
    cb_loans_banking = sys_banking.state.cb_loans_created_count
    outcomes_differ = (defaults_passive != defaults_banking) or (
        cb_loans_passive != cb_loans_banking
    )
    assert outcomes_differ, (
        f"Banking arm produced identical outcomes to passive: "
        f"defaults={defaults_passive}, CB loans passive={cb_loans_passive}, "
        f"CB loans banking={cb_loans_banking}"
    )


@pytest.mark.regression
def test_stylized_fact_liquidity_shortfall_produces_defaults():
    """Stylized fact: severe liquidity shortfall should trigger defaults.

    In ring payment systems with fixed obligations, kappa << 1 means agents
    cannot settle all obligations from cash buffers, so default incidence
    should be non-zero.
    """
    sys = build_ring_system(
        n_agents=12,
        cash_per_agent=20,
        payable_amount=100,
        maturity_days=5,
        seed=42,
    )
    run_until_stable(sys, max_days=10)

    defaults = count_defaults(sys)
    assert defaults >= 1, f"Expected defaults under stylized low-kappa stress, got {defaults}"


@pytest.mark.regression
def test_stylized_fact_nbfi_credit_supply_emits_loans():
    """Stylized fact: NBFI credit supply should activate under shortfalls."""
    sys = build_ring_system(
        n_agents=12,
        cash_per_agent=20,
        payable_amount=100,
        maturity_days=5,
        seed=42,
    )

    lender = NonBankLender(id="NBFI1", name="Non-Bank Lender")
    sys.add_agent(lender)
    sys.mint_cash("NBFI1", 3000)

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

    loan_count = count_events(sys, "NonBankLoanCreated")
    assert loan_count >= 1, f"Expected NBFI loans under stylized shortfall scenario, got {loan_count}"
