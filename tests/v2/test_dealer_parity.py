"""Parity for the dealer subsystem (slice 5): marker, passive, and active.

The active dealer reuses the shared trading machinery (DealerSubsystem,
TradeExecutor, matching engine) on both engines, so parity hinges on the
surrounding state reconciliation: ticket ingestion/maturity, payable
ownership sync, and the cash reconciliation (CashMinted/CashRetired)
points. Market makers are capitalized via initial mints so trading rounds
actually execute.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bilancio.config.models import ScenarioConfig
from bilancio_v2 import run_scenario
from tests.v2.golden_cases import load_golden_case, v2_case_snapshot


def assert_v2_self_consistent(config, banking_config=None):
    """Run on v2 and assert the run completes with all invariants holding.

    Ledger conservation invariants are enforced daily inside the engine;
    this adds the exported-balance double-entry check. Engine refusals
    (insufficient funds under fail-fast economics) are legitimate outcomes.
    """
    from bilancio.core.errors import DefaultError
    from bilancio_v2 import run_scenario
    from bilancio_v2.balance_invariants import assert_balance_invariants
    from bilancio_v2.views import balance_rows

    try:
        result = run_scenario(config, banking_config=banking_config)
    except (DefaultError, ValueError):
        return
    assert_balance_invariants(balance_rows(result))


BUCKETS = {
    "short": {"tau_min": 1, "tau_max": 3},
    "mid": {"tau_min": 4, "tau_max": 8},
    "long": {"tau_min": 9, "tau_max": 999},
}


def dealer_scenario(
    mode: str | None,
    *,
    n_agents: int = 6,
    cash: int = 120,
    amount: int = 200,
    capitalize_market_makers: bool = False,
    risk_enabled: bool = False,
    trading_rounds: int = 10,
) -> ScenarioConfig:
    agents = [{"id": "CB", "kind": "central_bank", "name": "CB"}] + [
        {"id": f"H{i}", "kind": "household", "name": f"H{i}"} for i in range(n_agents)
    ]
    actions: list[dict] = []
    if capitalize_market_makers:
        for bucket in BUCKETS:
            actions.append({"mint_cash": {"to": f"dealer_{bucket}", "amount": 800}})
            actions.append({"mint_cash": {"to": f"vbt_{bucket}", "amount": 1500}})
    actions += [{"mint_cash": {"to": f"H{i}", "amount": cash}} for i in range(n_agents)]
    for i in range(n_agents):
        actions.append(
            {
                "create_payable": {
                    "from": f"H{i}",
                    "to": f"H{(i + 1) % n_agents}",
                    "amount": amount,
                    "due_day": 2 + (i % 4),
                    "maturity_distance": 2 + (i % 4),
                }
            }
        )
    config: dict = {
        "version": 1,
        "name": f"dealer-{mode}",
        "agents": agents,
        "initial_actions": actions,
        "dealer": {
            "enabled": True,
            "ticket_size": "100",
            "buckets": BUCKETS,
            "risk_assessment": {"enabled": risk_enabled},
        },
        "run": {"max_days": 12, "quiet_days": 2, "default_handling": "expel-agent"},
    }
    if mode is not None:
        config["balanced_dealer"] = {
            "enabled": True,
            "mode": mode,
            "face_value": "100",
            "kappa": "0.5",
            "trading_rounds": trading_rounds,
        }
    return ScenarioConfig.model_validate(config)


@pytest.mark.parametrize(
    ("mode", "kwargs", "golden_name"),
    [
        (None, {}, "dealer_marker"),  # marker mode: daily snapshots, no trading
        (None, {"risk_enabled": True}, "dealer_marker_risk"),
        ("passive", {}, "dealer_passive"),
        ("active", {"capitalize_market_makers": True}, "dealer_active"),
        (
            "active",
            {"capitalize_market_makers": True, "risk_enabled": True, "cash": 60},
            "dealer_active_risk",
        ),
    ],
)
def test_dealer_modes_match_golden(mode: str | None, kwargs: dict, golden_name: str) -> None:
    golden = load_golden_case(golden_name)
    snapshot = v2_case_snapshot(dealer_scenario(mode, **kwargs), None)
    assert snapshot["balances"] == golden["balances"], golden_name
    assert snapshot["events"] == golden["events"], golden_name


def test_active_dealer_actually_trades() -> None:
    """Guard against vacuous parity: capitalized market makers must trade."""
    result = run_scenario(dealer_scenario("active", capitalize_market_makers=True, cash=60))
    kinds = {event["kind"] for event in result.events}
    assert "ClaimTransferredDealer" in kinds or "CashRetired" in kinds


@st.composite
def dealer_cases(draw: st.DrawFn) -> ScenarioConfig:
    n_agents = draw(st.integers(min_value=4, max_value=8))
    mode = draw(st.sampled_from(["active", "active", "passive", None]))
    agents = [{"id": "CB", "kind": "central_bank", "name": "CB"}] + [
        {"id": f"H{i}", "kind": "household", "name": f"H{i}"} for i in range(n_agents)
    ]
    actions: list[dict] = []
    if mode == "active":
        for bucket in BUCKETS:
            actions.append({"mint_cash": {"to": f"dealer_{bucket}", "amount": draw(st.integers(200, 1500))}})
            actions.append({"mint_cash": {"to": f"vbt_{bucket}", "amount": draw(st.integers(500, 3000))}})
    for i in range(n_agents):
        actions.append({"mint_cash": {"to": f"H{i}", "amount": draw(st.integers(20, 150))}})
    for i in range(n_agents):
        actions.append(
            {
                "create_payable": {
                    "from": f"H{i}",
                    "to": f"H{(i + 1) % n_agents}",
                    "amount": draw(st.integers(100, 400)),
                    "due_day": draw(st.integers(2, 7)),
                }
            }
        )
    config: dict = {
        "version": 1,
        "name": "dealer-property",
        "agents": agents,
        "initial_actions": actions,
        "dealer": {
            "enabled": True,
            "ticket_size": "100",
            "buckets": BUCKETS,
            "risk_assessment": {"enabled": draw(st.booleans())},
        },
        "run": {"max_days": 14, "quiet_days": 2, "default_handling": "expel-agent"},
    }
    if mode is not None:
        config["balanced_dealer"] = {
            "enabled": True,
            "mode": mode,
            "face_value": "100",
            "kappa": draw(st.sampled_from(["0.3", "0.5", "1"])),
            "trading_rounds": draw(st.sampled_from([5, 20, 100])),
            "issuer_specific_pricing": draw(st.booleans()),
        }
    return ScenarioConfig.model_validate(config)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(config=dealer_cases())
def test_dealer_scenarios_self_consistent(config: ScenarioConfig) -> None:
    assert_v2_self_consistent(config)
