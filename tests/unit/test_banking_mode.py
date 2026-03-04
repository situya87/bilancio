"""Unit tests for Plan 038 banking infrastructure.

Verifies that:
- PolicyEngine allows NonBankLender to hold BankDeposit
- _get_agent_cash functions count both CASH and BANK_DEPOSIT
- compile_ring_explorer_balanced produces correct scenarios with n_banks=0 and n_banks>0
- System._find_agent_bank() finds the right bank
"""

from decimal import Decimal

from bilancio.config.models import RingExplorerGeneratorConfig
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.agents.household import Household
from bilancio.domain.agents.non_bank_lender import NonBankLender
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.means_of_payment import BankDeposit
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.system import System
from bilancio.scenarios.ring.compiler import compile_ring_explorer_balanced

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generator_config(n_agents: int = 4, kappa: str = "2") -> RingExplorerGeneratorConfig:
    """Create a minimal RingExplorerGeneratorConfig for testing."""
    return RingExplorerGeneratorConfig.model_validate(
        {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "Banking Test",
            "params": {
                "n_agents": n_agents,
                "seed": 42,
                "kappa": kappa,
                "Q_total": "400",
                "inequality": {"scheme": "dirichlet", "concentration": "1"},
                "maturity": {"days": 5, "mode": "lead_lag", "mu": "0.5"},
            },
            "compile": {"emit_yaml": False},
        }
    )


def _create_system_with_deposits() -> System:
    """Create a system where households have bank deposits instead of just cash."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    bank = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    nbl = NonBankLender(id="NBL01", name="Non-Bank Lender")
    sys.add_agent(cb)
    sys.add_agent(bank)
    sys.add_agent(h1)
    sys.add_agent(h2)
    sys.add_agent(nbl)

    # Give H1 both cash and a deposit
    sys.mint_cash("H1", 50)
    dep1 = BankDeposit(
        id="DEP_H1",
        kind=InstrumentKind.BANK_DEPOSIT,
        amount=100,
        denom="X",
        asset_holder_id="H1",
        liability_issuer_id="B1",
    )
    sys.add_contract(dep1)

    # Give H2 only a deposit (no cash)
    dep2 = BankDeposit(
        id="DEP_H2",
        kind=InstrumentKind.BANK_DEPOSIT,
        amount=200,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="B1",
    )
    sys.add_contract(dep2)

    # Give NBL a deposit
    dep_nbl = BankDeposit(
        id="DEP_NBL",
        kind=InstrumentKind.BANK_DEPOSIT,
        amount=500,
        denom="X",
        asset_holder_id="NBL01",
        liability_issuer_id="B1",
    )
    sys.add_contract(dep_nbl)

    return sys


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------


class TestPolicyBankDeposit:
    """Verify PolicyEngine allows NonBankLender to hold BankDeposit."""

    def test_policy_nonbank_lender_can_hold_deposit(self):
        """NonBankLender must be allowed to hold BankDeposit instruments."""
        policy = PolicyEngine.default()
        nbl = NonBankLender(id="NBL01", name="Non-Bank Lender")
        Bank(id="B1", name="Bank 1", kind="bank")

        deposit = BankDeposit(
            id="DEP_TEST",
            kind=InstrumentKind.BANK_DEPOSIT,
            amount=100,
            denom="X",
            asset_holder_id="NBL01",
            liability_issuer_id="B1",
        )

        assert policy.can_hold(nbl, deposit), (
            "PolicyEngine.default() must allow NonBankLender to hold BankDeposit"
        )

    def test_policy_household_can_hold_deposit(self):
        """Household must be allowed to hold BankDeposit instruments."""
        policy = PolicyEngine.default()
        h = Household(id="H1", name="H1", kind="household")
        Bank(id="B1", name="Bank 1", kind="bank")

        deposit = BankDeposit(
            id="DEP_TEST",
            kind=InstrumentKind.BANK_DEPOSIT,
            amount=100,
            denom="X",
            asset_holder_id="H1",
            liability_issuer_id="B1",
        )

        assert policy.can_hold(h, deposit), (
            "PolicyEngine.default() must allow Household to hold BankDeposit"
        )

    def test_policy_firm_can_hold_deposit(self):
        """Firm must be allowed to hold BankDeposit instruments."""
        policy = PolicyEngine.default()
        f = Firm(id="F1", name="Firm 1", kind="firm")
        Bank(id="B1", name="Bank 1", kind="bank")

        deposit = BankDeposit(
            id="DEP_TEST",
            kind=InstrumentKind.BANK_DEPOSIT,
            amount=100,
            denom="X",
            asset_holder_id="F1",
            liability_issuer_id="B1",
        )

        assert policy.can_hold(f, deposit), (
            "PolicyEngine.default() must allow Firm to hold BankDeposit"
        )


# ---------------------------------------------------------------------------
# _get_agent_cash tests
# ---------------------------------------------------------------------------


class TestGetAgentCash:
    """Verify _get_agent_cash includes both CASH and BANK_DEPOSIT."""

    def test_dealer_integration_get_agent_cash_includes_deposits(self):
        """dealer_integration._get_agent_cash must count BANK_DEPOSIT as cash."""
        from bilancio.engines.dealer_integration import _get_agent_cash

        sys = _create_system_with_deposits()

        # H1 has 50 cash + 100 deposit = 150
        assert _get_agent_cash(sys, "H1") == Decimal(150), (
            "H1 should have 50 cash + 100 deposit = 150"
        )

        # H2 has only 200 deposit
        assert _get_agent_cash(sys, "H2") == Decimal(200), (
            "H2 should have 200 deposit (no cash)"
        )

        # NBL has 500 deposit
        assert _get_agent_cash(sys, "NBL01") == Decimal(500), (
            "NBL01 should have 500 deposit"
        )

    def test_lending_get_agent_cash_includes_deposits(self):
        """lending._get_agent_cash must count BANK_DEPOSIT as cash."""
        from bilancio.engines.lending import _get_agent_cash

        sys = _create_system_with_deposits()

        # H1 has 50 cash + 100 deposit = 150
        assert _get_agent_cash(sys, "H1") == 150, (
            "H1 should have 50 cash + 100 deposit = 150"
        )

        # H2 has only 200 deposit
        assert _get_agent_cash(sys, "H2") == 200, (
            "H2 should have 200 deposit (no cash)"
        )


# ---------------------------------------------------------------------------
# compile_ring_explorer_balanced tests
# ---------------------------------------------------------------------------


class TestCompilerNoBanks:
    """Verify n_banks=0 produces no bank agents and no deposit actions."""

    def test_no_banks_scenario_unchanged(self):
        """compile_ring_explorer_balanced(n_banks=0) must not produce bank agents."""
        config = _make_generator_config()
        scenario = compile_ring_explorer_balanced(
            config,
            n_banks=0,
            mode="passive",
        )

        agent_kinds = [a["kind"] for a in scenario["agents"]]
        assert "bank" not in agent_kinds, (
            "With n_banks=0, no bank agents should be created"
        )

        # No deposit_cash actions
        deposit_actions = [a for a in scenario["initial_actions"] if "deposit_cash" in a]
        assert len(deposit_actions) == 0, (
            "With n_banks=0, no deposit_cash actions should exist"
        )

        # No mint_reserves actions (other than potential CB reserves)
        reserve_actions = [a for a in scenario["initial_actions"] if "mint_reserves" in a]
        assert len(reserve_actions) == 0, (
            "With n_banks=0, no mint_reserves actions should exist"
        )


class TestCompilerWithBanks:
    """Verify n_banks>0 produces correct banking infrastructure."""

    def test_banks_scenario_has_bank_agents(self):
        """compile_ring_explorer_balanced(n_banks=2) must produce bank_1, bank_2."""
        config = _make_generator_config()
        scenario = compile_ring_explorer_balanced(
            config,
            n_banks=2,
            mode="passive",
        )

        agent_ids = [a["id"] for a in scenario["agents"]]
        assert "bank_1" in agent_ids, "bank_1 must be in agents"
        assert "bank_2" in agent_ids, "bank_2 must be in agents"

        bank_agents = [a for a in scenario["agents"] if a["kind"] == "bank"]
        assert len(bank_agents) == 2, f"Expected 2 bank agents, got {len(bank_agents)}"

    def test_banks_scenario_has_deposit_actions(self):
        """n_banks>0 must generate deposit_cash actions for non-bank agents."""
        config = _make_generator_config()
        scenario = compile_ring_explorer_balanced(
            config,
            n_banks=2,
            mode="passive",
        )

        deposit_actions = [a for a in scenario["initial_actions"] if "deposit_cash" in a]
        assert len(deposit_actions) > 0, (
            "With n_banks=2, deposit_cash actions must be generated"
        )

        # Each deposit action must reference a valid bank
        for action in deposit_actions:
            bank_id = action["deposit_cash"]["bank"]
            assert bank_id in ("bank_1", "bank_2"), (
                f"deposit_cash action references unknown bank: {bank_id}"
            )

    def test_banks_scenario_has_reserve_minting(self):
        """n_banks>0 must generate mint_reserves actions for each bank."""
        config = _make_generator_config()
        scenario = compile_ring_explorer_balanced(
            config,
            n_banks=2,
            mode="passive",
        )

        reserve_actions = [a for a in scenario["initial_actions"] if "mint_reserves" in a]
        reserve_banks = {a["mint_reserves"]["to"] for a in reserve_actions}

        assert "bank_1" in reserve_banks, "bank_1 must receive reserve minting"
        assert "bank_2" in reserve_banks, "bank_2 must receive reserve minting"

        # Reserves should be substantial (reserve_multiplier * total_deposited / n_banks)
        for action in reserve_actions:
            assert action["mint_reserves"]["amount"] > 0, (
                "Reserve amount must be positive"
            )

    def test_banks_scenario_policy_overrides(self):
        """n_banks>0 must set mop_rank overrides for household/firm/non_bank_lender."""
        config = _make_generator_config()
        scenario = compile_ring_explorer_balanced(
            config,
            n_banks=2,
            mode="passive",
        )

        overrides = scenario.get("policy_overrides", {})
        assert "mop_rank" in overrides, "policy_overrides must include mop_rank"

        mop_rank = overrides["mop_rank"]
        assert "household" in mop_rank, "mop_rank must include household"
        assert "firm" in mop_rank, "mop_rank must include firm"
        assert "non_bank_lender" in mop_rank, "mop_rank must include non_bank_lender"

        # All should prefer bank_deposit
        assert mop_rank["household"] == ["bank_deposit"], (
            "Household mop_rank must be ['bank_deposit']"
        )
        assert mop_rank["firm"] == ["bank_deposit"], (
            "Firm mop_rank must be ['bank_deposit']"
        )
        assert mop_rank["non_bank_lender"] == ["bank_deposit"], (
            "NonBankLender mop_rank must be ['bank_deposit']"
        )

    def test_banks_scenario_balanced_config_stored(self):
        """n_banks value must be stored in the _balanced_config section."""
        config = _make_generator_config()
        scenario = compile_ring_explorer_balanced(
            config,
            n_banks=2,
            mode="passive",
        )

        balanced_config = scenario.get("_balanced_config", {})
        assert balanced_config.get("n_banks") == 2, (
            "_balanced_config must store n_banks=2"
        )
        assert "bank_assignments" in balanced_config, (
            "_balanced_config must include bank_assignments"
        )
        assert len(balanced_config["bank_assignments"]) > 0, (
            "bank_assignments must not be empty when n_banks>0"
        )

    def test_no_banks_balanced_config_zero(self):
        """n_banks=0 must be stored in _balanced_config."""
        config = _make_generator_config()
        scenario = compile_ring_explorer_balanced(
            config,
            n_banks=0,
            mode="passive",
        )

        balanced_config = scenario.get("_balanced_config", {})
        assert balanced_config.get("n_banks") == 0, (
            "_balanced_config must store n_banks=0"
        )


# ---------------------------------------------------------------------------
# System._find_agent_bank tests
# ---------------------------------------------------------------------------


class TestFindAgentBank:
    """Verify System._find_agent_bank() finds the correct bank."""

    def test_find_agent_bank_with_deposit(self):
        """_find_agent_bank returns the bank where the agent holds a deposit."""
        sys = _create_system_with_deposits()

        # H1 has a deposit at B1
        assert sys._find_agent_bank("H1") == "B1", (
            "H1 holds a deposit at B1, so _find_agent_bank should return B1"
        )

        # H2 has a deposit at B1
        assert sys._find_agent_bank("H2") == "B1", (
            "H2 holds a deposit at B1, so _find_agent_bank should return B1"
        )

        # NBL01 has a deposit at B1
        assert sys._find_agent_bank("NBL01") == "B1", (
            "NBL01 holds a deposit at B1, so _find_agent_bank should return B1"
        )

    def test_find_agent_bank_no_deposit(self):
        """_find_agent_bank returns None when agent has no bank deposit."""
        sys = System()
        cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
        h1 = Household(id="H1", name="Household 1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.mint_cash("H1", 100)

        assert sys._find_agent_bank("H1") is None, (
            "H1 has only cash (no deposit), so _find_agent_bank should return None"
        )

    def test_find_agent_bank_nonexistent_agent(self):
        """_find_agent_bank returns None for a nonexistent agent."""
        sys = System()
        assert sys._find_agent_bank("DOES_NOT_EXIST") is None

    def test_find_agent_bank_multiple_banks(self):
        """_find_agent_bank returns the first bank found in the deposit list."""
        sys = System()
        cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
        b1 = Bank(id="B1", name="Bank 1", kind="bank")
        b2 = Bank(id="B2", name="Bank 2", kind="bank")
        h1 = Household(id="H1", name="Household 1", kind="household")
        sys.add_agent(cb)
        sys.add_agent(b1)
        sys.add_agent(b2)
        sys.add_agent(h1)

        # Add deposit at B1
        dep1 = BankDeposit(
            id="DEP_H1_B1",
            kind=InstrumentKind.BANK_DEPOSIT,
            amount=100,
            denom="X",
            asset_holder_id="H1",
            liability_issuer_id="B1",
        )
        sys.add_contract(dep1)

        result = sys._find_agent_bank("H1")
        assert result == "B1", (
            f"Expected B1 as the bank for H1's deposit, got {result}"
        )
