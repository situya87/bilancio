"""Top-level shared fixtures and factories for the bilancio test suite.

Consolidates duplicated test setup that was previously copy-pasted across
multiple test files (test_cash_conservation, test_dealer_sync_cash,
test_dealer_integration, test_metrics, test_banking_dealer_sync).
"""

from decimal import Decimal

import pytest

from bilancio.config.apply import apply_to_system
from bilancio.config.loaders import preprocess_config
from bilancio.config.models import RingExplorerGeneratorConfig, ScenarioConfig
from bilancio.dealer.models import DEFAULT_BUCKETS
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.system import System
from bilancio.scenarios.ring_explorer import compile_ring_explorer

# ---------------------------------------------------------------------------
# Shared factory functions (importable by any test module)
# ---------------------------------------------------------------------------


def create_test_system_with_payables() -> System:
    """Create a minimal system with three households, cash, and a payable ring.

    Topology:
        H1 owes H2  50  (due in 2 days)
        H2 owes H3  30  (due in 5 days)
        H3 owes H1  20  (due in 10 days)

    Each household starts with 100 in cash (total 300).
    Includes CB1 and B1 for policy compliance.
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
    sys.mint_cash("H1", 100)
    sys.mint_cash("H2", 100)
    sys.mint_cash("H3", 100)

    p1 = Payable(
        id=sys.new_contract_id("P"),
        kind=InstrumentKind.PAYABLE,
        amount=50,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="H1",
        due_day=sys.state.day + 2,
    )
    sys.add_contract(p1)

    p2 = Payable(
        id=sys.new_contract_id("P"),
        kind=InstrumentKind.PAYABLE,
        amount=30,
        denom="X",
        asset_holder_id="H3",
        liability_issuer_id="H2",
        due_day=sys.state.day + 5,
    )
    sys.add_contract(p2)

    p3 = Payable(
        id=sys.new_contract_id("P"),
        kind=InstrumentKind.PAYABLE,
        amount=20,
        denom="X",
        asset_holder_id="H1",
        liability_issuer_id="H3",
        due_day=sys.state.day + 10,
    )
    sys.add_contract(p3)

    return sys


def create_dealer_config() -> DealerRingConfig:
    """Standard dealer configuration for testing.

    Uses DEFAULT_BUCKETS (short/mid/long), with standard anchors,
    25% dealer share, 50% VBT share, seed=42.
    """
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


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ring_system_10():
    """Create a System with 10 agents in a ring, ready for simulation.

    Returns a (system, config) tuple so tests can inspect the scenario config
    if needed.  Originally from tests/benchmark/conftest.py.
    """
    gen_config = RingExplorerGeneratorConfig.model_validate(
        {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "benchmark",
            "params": {
                "n_agents": 10,
                "kappa": "1",
                "seed": 42,
                "Q_total": "1000",
                "inequality": {"scheme": "dirichlet", "concentration": "1"},
                "maturity": {"days": 5, "mode": "lead_lag", "mu": "0.5"},
                "liquidity": {"allocation": {"mode": "uniform"}},
            },
            "compile": {"emit_yaml": False},
        }
    )
    scenario_dict = compile_ring_explorer(gen_config, source_path=None)
    scenario_dict = preprocess_config(scenario_dict)
    config = ScenarioConfig(**scenario_dict)

    system = System()
    apply_to_system(config, system)
    return system, config
