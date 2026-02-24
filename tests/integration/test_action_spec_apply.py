"""Integration tests for action spec application to the system."""

import pytest
from decimal import Decimal

from bilancio.config.models import ScenarioConfig
from bilancio.config.apply import apply_to_system
from bilancio.engines.system import System


def _build_minimal_scenario_with_action_specs() -> dict:
    """Build a minimal scenario dict that uses action_specs."""
    return {
        "version": 1,
        "name": "Action Spec Test",
        "agents": [
            {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
            {"id": "H1", "kind": "household", "name": "Agent 1"},
            {"id": "H2", "kind": "household", "name": "Agent 2"},
            {"id": "H3", "kind": "household", "name": "Agent 3"},
        ],
        "action_specs": [
            {
                "kind": "household",
                "actions": [
                    {"action": "settle", "phase": "B2_Settlement"},
                ],
                "profile_type": "trader",
                "profile_params": {
                    "risk_aversion": "0.5",
                    "planning_horizon": 10,
                },
                "information": "omniscient",
            },
        ],
        "initial_actions": [
            {"mint_cash": {"to": "H1", "amount": 500}},
            {"mint_cash": {"to": "H2", "amount": 300}},
            {"mint_cash": {"to": "H3", "amount": 200}},
            {"create_payable": {"from": "H1", "to": "H2", "amount": 200, "due_day": 3}},
            {"create_payable": {"from": "H2", "to": "H3", "amount": 150, "due_day": 3}},
            {"create_payable": {"from": "H3", "to": "H1", "amount": 100, "due_day": 3}},
        ],
        "run": {
            "mode": "until_stable",
            "max_days": 10,
        },
    }


class TestActionSpecApply:
    """Tests for applying action_specs to a system."""

    def test_action_specs_parsed_from_dict(self):
        """ScenarioConfig accepts action_specs field."""
        scenario = _build_minimal_scenario_with_action_specs()
        config = ScenarioConfig(**scenario)
        assert config.action_specs is not None
        assert len(config.action_specs) == 1
        assert config.action_specs[0].kind == "household"
        assert config.action_specs[0].profile_type == "trader"

    def test_action_specs_applied_to_system(self):
        """Scenario with action_specs can be applied without errors."""
        scenario = _build_minimal_scenario_with_action_specs()
        config = ScenarioConfig(**scenario)
        system = System()
        apply_to_system(config, system)

        # Verify agents were created
        assert "H1" in system.state.agents
        assert "H2" in system.state.agents
        assert "H3" in system.state.agents

    def test_old_format_still_works(self):
        """Scenario without action_specs still works (legacy path)."""
        scenario = {
            "version": 1,
            "name": "Legacy Test",
            "agents": [
                {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
                {"id": "H1", "kind": "household", "name": "Agent 1"},
                {"id": "H2", "kind": "household", "name": "Agent 2"},
            ],
            "initial_actions": [
                {"mint_cash": {"to": "H1", "amount": 500}},
                {"mint_cash": {"to": "H2", "amount": 300}},
                {"create_payable": {"from": "H1", "to": "H2", "amount": 200, "due_day": 3}},
            ],
            "run": {
                "mode": "until_stable",
                "max_days": 10,
            },
        }
        config = ScenarioConfig(**scenario)
        assert config.action_specs is None
        system = System()
        apply_to_system(config, system)
        assert "H1" in system.state.agents

    def test_action_specs_with_lending(self):
        """action_specs with B_Lending phase initializes lending config."""
        scenario = {
            "version": 1,
            "name": "Lending Action Spec Test",
            "agents": [
                {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
                {"id": "H1", "kind": "household", "name": "Agent 1"},
                {"id": "H2", "kind": "household", "name": "Agent 2"},
                {"id": "lender", "kind": "non_bank_lender", "name": "Lender"},
            ],
            "action_specs": [
                {
                    "kind": "household",
                    "actions": [
                        {"action": "settle", "phase": "B2_Settlement"},
                        {"action": "borrow", "phase": "B_Lending"},
                    ],
                },
                {
                    "kind": "non_bank_lender",
                    "actions": [
                        {"action": "lend", "phase": "B_Lending", "strategy": "linear_pricer"},
                    ],
                    "profile_type": "lender",
                    "profile_params": {
                        "kappa": "1.0",
                        "risk_aversion": "0.3",
                        "planning_horizon": 5,
                        "profit_target": "0.05",
                        "max_loan_maturity": 10,
                    },
                    "information": "omniscient",
                },
            ],
            "initial_actions": [
                {"mint_cash": {"to": "H1", "amount": 500}},
                {"mint_cash": {"to": "H2", "amount": 300}},
                {"mint_cash": {"to": "lender", "amount": 1000}},
                {"create_payable": {"from": "H1", "to": "H2", "amount": 200, "due_day": 3}},
            ],
            "run": {
                "mode": "until_stable",
                "max_days": 10,
            },
        }
        config = ScenarioConfig(**scenario)
        system = System()
        apply_to_system(config, system)

        # Lending config should be set
        assert system.state.lender_config is not None

    def test_action_specs_with_rating(self):
        """action_specs with B_Rating phase initializes rating config."""
        scenario = {
            "version": 1,
            "name": "Rating Action Spec Test",
            "agents": [
                {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
                {"id": "H1", "kind": "household", "name": "Agent 1"},
                {"id": "RA", "kind": "rating_agency", "name": "Rating Agency"},
            ],
            "action_specs": [
                {
                    "kind": "household",
                    "actions": [
                        {"action": "settle", "phase": "B2_Settlement"},
                    ],
                },
                {
                    "kind": "rating_agency",
                    "actions": [
                        {"action": "rate", "phase": "B_Rating"},
                    ],
                    "profile_type": "rating",
                    "profile_params": {
                        "lookback_window": 7,
                        "balance_sheet_weight": "0.5",
                        "history_weight": "0.5",
                    },
                    "information": "realistic",
                },
            ],
            "initial_actions": [
                {"mint_cash": {"to": "H1", "amount": 500}},
            ],
            "run": {
                "mode": "until_stable",
                "max_days": 10,
            },
        }
        config = ScenarioConfig(**scenario)
        system = System()
        apply_to_system(config, system)

        # Rating config should be set
        assert system.state.rating_config is not None


class TestRingCompilerActionSpecs:
    """Tests for ring compiler action_specs generation."""

    def test_build_action_specs_passive(self):
        from bilancio.scenarios.ring.compiler import _build_action_specs

        specs = _build_action_specs(mode="passive")
        assert len(specs) == 1
        household_spec = specs[0]
        assert household_spec["kind"] == "household"
        actions = [a["action"] for a in household_spec["actions"]]
        assert "settle" in actions
        assert "sell_ticket" not in actions
        assert "buy_ticket" not in actions

    def test_build_action_specs_active(self):
        from bilancio.scenarios.ring.compiler import _build_action_specs

        specs = _build_action_specs(mode="active")
        household_spec = specs[0]
        actions = [a["action"] for a in household_spec["actions"]]
        assert "settle" in actions
        assert "sell_ticket" in actions
        assert "buy_ticket" in actions

    def test_build_action_specs_nbfi(self):
        from bilancio.scenarios.ring.compiler import _build_action_specs

        specs = _build_action_specs(mode="nbfi")
        # Should have household + lender
        assert len(specs) == 2
        kinds = [s["kind"] for s in specs]
        assert "household" in kinds
        assert "non_bank_lender" in kinds

        # Household should have borrow
        household = [s for s in specs if s["kind"] == "household"][0]
        actions = [a["action"] for a in household["actions"]]
        assert "borrow" in actions
        # No dealer trading in pure nbfi
        assert "sell_ticket" not in actions

    def test_build_action_specs_nbfi_dealer(self):
        from bilancio.scenarios.ring.compiler import _build_action_specs

        specs = _build_action_specs(mode="nbfi_dealer")
        household = [s for s in specs if s["kind"] == "household"][0]
        actions = [a["action"] for a in household["actions"]]
        assert "sell_ticket" in actions
        assert "borrow" in actions

    def test_build_action_specs_with_trader_params(self):
        from bilancio.scenarios.ring.compiler import _build_action_specs

        specs = _build_action_specs(
            mode="active",
            trader_params={"risk_aversion": "0.5"},
        )
        household = specs[0]
        assert household["profile_type"] == "trader"
        assert household["profile_params"]["risk_aversion"] == "0.5"

    def test_emit_action_specs_flag(self):
        from bilancio.config.models import RingExplorerGeneratorConfig
        from bilancio.scenarios.ring.compiler import compile_ring_explorer_balanced

        config = RingExplorerGeneratorConfig(
            name_prefix="test",
            params={"n_agents": 5, "kappa": "1.0", "seed": 42, "Q_total": "1000"},
            compile={"emit_yaml": False},
        )
        # Without flag
        scenario = compile_ring_explorer_balanced(config)
        assert "action_specs" not in scenario

        # With flag
        scenario = compile_ring_explorer_balanced(config, emit_action_specs=True)
        assert "action_specs" in scenario
        assert len(scenario["action_specs"]) >= 1
