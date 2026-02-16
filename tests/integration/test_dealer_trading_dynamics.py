"""Integration tests for dealer trading dynamics.

Tests verify that the dealer trading system produces realistic two-sided
market behavior when running a full balanced scenario with active dealer
trading enabled. Each test builds a complete ring scenario, runs the
simulation, and inspects the trade records produced by the dealer subsystem.
"""

from decimal import Decimal

import pytest

from bilancio.config.apply import apply_to_system
from bilancio.config.models import (
    RingExplorerGeneratorConfig,
    ScenarioConfig,
)
from bilancio.dealer.metrics import TradeRecord
from bilancio.engines.dealer_integration import DealerSubsystem
from bilancio.engines.dealer_sync import _update_vbt_credit_mids
from bilancio.engines.simulation import run_day
from bilancio.engines.system import System
from bilancio.scenarios import compile_ring_explorer_balanced

# ---------------------------------------------------------------------------
# Helper: build and run an active balanced scenario
# ---------------------------------------------------------------------------


def _run_active_scenario(
    kappa: Decimal = Decimal("0.3"),
    n_agents: int = 50,
    maturity_days: int = 10,
    seed: int = 100,
    max_days: int | None = None,
) -> tuple[System, DealerSubsystem, list[TradeRecord]]:
    """Build a balanced scenario with dealer trading, run it, return trade data.

    This follows the same code path as the balanced comparison sweep
    (``RingSweepRunner._prepare_run`` with ``balanced_mode=True``):
      1. Compile a ring explorer balanced scenario dict.
      2. Inject dealer + balanced_dealer config sections.
      3. Parse into ``ScenarioConfig`` and apply to a fresh ``System``.
      4. Run the simulation day-by-day with dealer enabled.
      5. Return the system, dealer subsystem, and trade records.

    Risk assessment is disabled to allow free-flowing trades.  When
    ``outside_mid_ratio=0.75`` the VBT mid sits at ~0.64, so dealer
    bids are structurally below the 0.85 no-history expected-value
    threshold, which would reject almost all sells if risk assessment
    were enabled. Tests that specifically exercise risk-adjusted pricing
    should use a dedicated setup (see ``TestVBTCreditUpdate``).

    Args:
        kappa: Debt-to-liquidity ratio (lower = more stressed).
        n_agents: Number of firms in the ring.
        maturity_days: Payment horizon in days.
        seed: PRNG seed for reproducibility.
        max_days: Simulation length; defaults to ``3 * maturity_days``.

    Returns:
        Tuple of (system, dealer_subsystem, trade_records).
    """
    if max_days is None:
        max_days = 3 * maturity_days

    # Step 1: Compile the balanced scenario dict
    generator_data = {
        "version": 1,
        "generator": "ring_explorer_v1",
        "name_prefix": "test-dynamics",
        "params": {
            "n_agents": n_agents,
            "seed": seed,
            "kappa": str(kappa),
            "Q_total": str(Decimal(n_agents) * Decimal("100")),
            "inequality": {
                "scheme": "dirichlet",
                "concentration": "1",
                "monotonicity": "0",
            },
            "maturity": {
                "days": maturity_days,
                "mode": "lead_lag",
                "mu": "0.5",
            },
            "liquidity": {
                "allocation": {"mode": "uniform"},
            },
        },
        "compile": {"emit_yaml": False},
    }
    generator_config = RingExplorerGeneratorConfig.model_validate(generator_data)

    face_value = Decimal("20")
    outside_mid_ratio = Decimal("0.75")
    vbt_share = Decimal("0.25")
    dealer_share = Decimal("0.125")

    scenario = compile_ring_explorer_balanced(
        generator_config,
        face_value=face_value,
        outside_mid_ratio=outside_mid_ratio,
        vbt_share_per_bucket=vbt_share,
        dealer_share_per_bucket=dealer_share,
        mode="active",
        rollover_enabled=True,
        source_path=None,
    )

    # Step 2: Inject dealer and balanced_dealer sections
    scenario["dealer"] = {
        "enabled": True,
        "ticket_size": 1,
        "dealer_share": "0.25",
        "vbt_share": "0.50",
        "risk_assessment": {"enabled": False},
    }
    scenario["balanced_dealer"] = {
        "enabled": True,
        "face_value": str(face_value),
        "outside_mid_ratio": str(outside_mid_ratio),
        "vbt_share_per_bucket": str(vbt_share),
        "dealer_share_per_bucket": str(dealer_share),
        "mode": "active",
        "rollover_enabled": True,
    }

    # Use expel-agent default handling so defaults do not crash the simulation
    scenario.setdefault("run", {})["default_handling"] = "expel-agent"

    # Step 3: Parse and apply
    config = ScenarioConfig(**scenario)
    system = System(default_mode="expel-agent")
    apply_to_system(config, system)
    system.state.rollover_enabled = True

    subsystem = system.state.dealer_subsystem
    assert subsystem is not None, "Dealer subsystem must be initialized"

    # Step 4: Run simulation
    for _ in range(max_days):
        run_day(system, enable_dealer=True)

    return system, subsystem, list(subsystem.metrics.trades)


# ---------------------------------------------------------------------------
# Module-scoped fixture: shared scenario for tests 1-4
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def active_scenario():
    """Run a single active scenario shared by all trading-dynamics tests.

    Parameters chosen to reliably produce diverse trading activity:
    - kappa=0.3: severely stressed (many agents need liquidity)
    - n_agents=50: enough participants for two-sided flow
    - seed=100: deterministic, verified to produce 100+ trades across 7 days
    """
    system, subsystem, trades = _run_active_scenario(
        kappa=Decimal("0.3"),
        n_agents=50,
        seed=100,
        maturity_days=10,
        max_days=30,
    )
    return system, subsystem, trades


# ---------------------------------------------------------------------------
# Test 1: Trading happens on multiple days
# ---------------------------------------------------------------------------


def test_trading_happens_on_multiple_days(active_scenario):
    """Trades should occur on at least 3 different days, not a single burst."""
    _, _, trades = active_scenario

    assert len(trades) > 0, "There should be at least some trades"

    unique_days = {t.day for t in trades}
    assert len(unique_days) >= 3, (
        f"Trades should span at least 3 different days, but only found days {sorted(unique_days)}"
    )


# ---------------------------------------------------------------------------
# Test 2: Both buy and sell trades occur
# ---------------------------------------------------------------------------


def test_both_buy_and_sell_trades_occur(active_scenario):
    """The dealer market should produce both BUY and SELL trades."""
    _, _, trades = active_scenario

    sell_trades = [t for t in trades if t.side == "SELL"]
    buy_trades = [t for t in trades if t.side == "BUY"]

    assert len(sell_trades) > 0, "There should be at least one SELL trade"
    assert len(buy_trades) >= 1, (
        f"There should be at least one BUY trade, "
        f"but found {len(buy_trades)} buys out of {len(trades)} total trades"
    )


# ---------------------------------------------------------------------------
# Test 3: Dealer is not always passthrough
# ---------------------------------------------------------------------------


def test_dealer_not_always_passthrough(active_scenario):
    """At least some trades should be interior (dealer market-making)."""
    _, _, trades = active_scenario

    assert len(trades) > 0, "There should be at least some trades"

    interior_trades = [t for t in trades if not t.is_passthrough]
    passthrough_trades = [t for t in trades if t.is_passthrough]

    assert len(interior_trades) > 0, (
        f"At least some trades should be interior (dealer market-making), "
        f"but all {len(trades)} trades were passthrough. "
        f"Passthrough count: {len(passthrough_trades)}"
    )


# ---------------------------------------------------------------------------
# Test 4: Trades span multiple maturity buckets
# ---------------------------------------------------------------------------


def test_trades_span_multiple_buckets(active_scenario):
    """Trades should span at least 2 different maturity buckets."""
    _, _, trades = active_scenario

    assert len(trades) > 0, "There should be at least some trades"

    unique_buckets = {t.bucket for t in trades}
    assert len(unique_buckets) >= 2, (
        f"Trades should span at least 2 maturity buckets, but only found buckets: {unique_buckets}"
    )


# ---------------------------------------------------------------------------
# Test 5: VBT mid updates with credit risk
# ---------------------------------------------------------------------------


def test_vbt_mid_updates_with_credit_risk():
    """VBT M should change when the risk assessor learns from settlement history.

    Steps:
      1. Set up a dealer subsystem with risk assessment enabled.
      2. Record initial VBT M (should be credit-adjusted < 1.0).
      3. Feed several successful settlements into the risk assessor.
      4. Call _update_vbt_credit_mids.
      5. Assert M changed (should increase with fewer defaults).
    """
    agents_data = [
        {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
        {"id": "T1", "kind": "household", "name": "Trader 1"},
        {"id": "T2", "kind": "household", "name": "Trader 2"},
        {"id": "vbt_short", "kind": "household", "name": "VBT (short)"},
        {"id": "dealer_short", "kind": "household", "name": "Dealer (short)"},
    ]

    initial_actions = [
        {"mint_cash": {"to": "T1", "amount": 100}},
        {"mint_cash": {"to": "T2", "amount": 100}},
        {"mint_cash": {"to": "vbt_short", "amount": 50}},
        {"mint_cash": {"to": "dealer_short", "amount": 30}},
        {"create_payable": {"from": "T1", "to": "T2", "amount": 20, "due_day": 3}},
        {"create_payable": {"from": "T2", "to": "T1", "amount": 20, "due_day": 3}},
        {"create_payable": {"from": "T1", "to": "vbt_short", "amount": 20, "due_day": 2}},
        {"create_payable": {"from": "T2", "to": "dealer_short", "amount": 20, "due_day": 2}},
    ]

    scenario_data = {
        "name": "vbt-credit-test",
        "agents": agents_data,
        "initial_actions": initial_actions,
        "dealer": {
            "enabled": True,
            "ticket_size": 1,
            "dealer_share": "0.25",
            "vbt_share": "0.50",
            "risk_assessment": {"enabled": True},
        },
        "balanced_dealer": {
            "enabled": True,
            "face_value": "20",
            "outside_mid_ratio": "0.75",
            "mode": "active",
        },
    }

    config = ScenarioConfig(**scenario_data)
    system = System()
    apply_to_system(config, system)

    subsystem = system.state.dealer_subsystem
    assert subsystem is not None, "Dealer subsystem must be initialized"
    assert subsystem.risk_assessor is not None, "Risk assessor must be initialized"

    # Record initial VBT M values
    initial_mids = {bid: vbt.M for bid, vbt in subsystem.vbts.items()}

    # Verify initial M < 1.0 (credit-adjusted with default prior)
    for bucket_id, m in initial_mids.items():
        assert m < Decimal("1.0"), (
            f"Initial VBT M for '{bucket_id}' should be < 1.0 (credit-adjusted), got {m}"
        )

    # Simulate several successful settlements (no defaults)
    for day in range(10):
        subsystem.risk_assessor.update_history(day, "T1", defaulted=False)
        subsystem.risk_assessor.update_history(day, "T2", defaulted=False)

    # Update VBT mids based on new credit information
    _update_vbt_credit_mids(subsystem, current_day=10)

    # Verify M changed (should increase since observed default rate < prior)
    for bucket_id, vbt in subsystem.vbts.items():
        new_m = vbt.M
        old_m = initial_mids[bucket_id]
        assert new_m != old_m, (
            f"VBT M for '{bucket_id}' should have changed after learning "
            f"from settlement history, but stayed at {old_m}"
        )
        assert new_m > old_m, (
            f"VBT M for '{bucket_id}' should increase when observed default "
            f"rate is lower than prior: old={old_m}, new={new_m}"
        )
