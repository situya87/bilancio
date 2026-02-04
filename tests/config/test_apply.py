"""Tests for applying configuration to system."""

import pytest
from decimal import Decimal

from bilancio.engines.system import System
from bilancio.config.models import ScenarioConfig, AgentSpec
from bilancio.config.apply import apply_to_system, create_agent, apply_action
from bilancio.domain.agents import Bank, Household, CentralBank, Firm


class TestCreateAgent:
    """Test agent creation from specification."""
    
    def test_create_central_bank(self):
        """Test creating a central bank agent."""
        spec = AgentSpec(id="CB", kind="central_bank", name="Central Bank")
        agent = create_agent(spec)
        assert isinstance(agent, CentralBank)
        assert agent.id == "CB"
        assert agent.name == "Central Bank"
        assert agent.kind == "central_bank"
    
    def test_create_bank(self):
        """Test creating a bank agent."""
        spec = AgentSpec(id="B1", kind="bank", name="First Bank")
        agent = create_agent(spec)
        assert isinstance(agent, Bank)
        assert agent.id == "B1"
        assert agent.name == "First Bank"
        assert agent.kind == "bank"
    
    def test_create_household(self):
        """Test creating a household agent."""
        spec = AgentSpec(id="H1", kind="household", name="Smith Family")
        agent = create_agent(spec)
        assert isinstance(agent, Household)
        assert agent.id == "H1"
        assert agent.name == "Smith Family"
        assert agent.kind == "household"
    
    def test_create_firm(self):
        """Test creating a firm agent."""
        spec = AgentSpec(id="F1", kind="firm", name="ABC Corp")
        agent = create_agent(spec)
        assert isinstance(agent, Firm)
        assert agent.id == "F1"
        assert agent.name == "ABC Corp"
        assert agent.kind == "firm"


class TestApplyToSystem:
    """Test applying configuration to system."""
    
    def test_apply_minimal_config(self):
        """Test applying minimal configuration."""
        config = ScenarioConfig(
            name="Minimal",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
                {"id": "B1", "kind": "bank", "name": "Bank"}
            ]
        )
        
        system = System()
        apply_to_system(config, system)
        
        # Check agents were added
        assert "CB" in system.state.agents
        assert "B1" in system.state.agents
        assert system.state.agents["CB"].kind == "central_bank"
        assert system.state.agents["B1"].kind == "bank"
        
        # System should pass invariants
        system.assert_invariants()
    
    def test_apply_with_initial_actions(self):
        """Test applying configuration with initial actions."""
        config = ScenarioConfig(
            name="With Actions",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
                {"id": "B1", "kind": "bank", "name": "Bank"},
                {"id": "H1", "kind": "household", "name": "Household"}
            ],
            initial_actions=[
                {"mint_reserves": {"to": "B1", "amount": 10000}},
                {"mint_cash": {"to": "H1", "amount": 1000}}
            ]
        )
        
        system = System()
        apply_to_system(config, system)
        
        # Check agents
        assert len(system.state.agents) == 3
        
        # Check that actions were executed
        # Bank should have reserves
        bank = system.state.agents["B1"]
        assert len(bank.asset_ids) > 0  # Should have reserve deposit
        
        # Household should have cash
        household = system.state.agents["H1"]
        assert len(household.asset_ids) > 0  # Should have cash
        
        # System should pass invariants
        system.assert_invariants()
    
    def test_apply_with_policy_overrides(self):
        """Test applying configuration with policy overrides."""
        config = ScenarioConfig(
            name="With Policy",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "CB"}
            ],
            policy_overrides={
                "mop_rank": {
                    "household": ["cash", "bank_deposit"],
                    "bank": ["reserve_deposit"]
                }
            }
        )
        
        system = System()
        apply_to_system(config, system)
        
        # Check policy was updated
        assert system.policy.mop_rank["household"] == ["cash", "bank_deposit"]
        assert system.policy.mop_rank["bank"] == ["reserve_deposit"]
    
    def test_apply_complex_scenario(self):
        """Test applying a complex scenario with multiple actions."""
        config = ScenarioConfig(
            name="Complex",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
                {"id": "B1", "kind": "bank", "name": "Bank 1"},
                {"id": "B2", "kind": "bank", "name": "Bank 2"},
                {"id": "H1", "kind": "household", "name": "Household 1"},
                {"id": "H2", "kind": "household", "name": "Household 2"},
                {"id": "F1", "kind": "firm", "name": "Firm 1"}
            ],
            initial_actions=[
                {"mint_reserves": {"to": "B1", "amount": 20000}},
                {"mint_reserves": {"to": "B2", "amount": 15000}},
                {"mint_cash": {"to": "H1", "amount": 5000}},
                {"mint_cash": {"to": "H2", "amount": 3000}},
                {"deposit_cash": {"customer": "H1", "bank": "B1", "amount": 4000}},
                {"deposit_cash": {"customer": "H2", "bank": "B2", "amount": 2000}},
                {"create_stock": {"owner": "F1", "sku": "WIDGET", "quantity": 100, "unit_price": "25"}},
                {"create_payable": {"from": "H1", "to": "H2", "amount": 500, "due_day": 1}}
            ]
        )
        
        system = System()
        apply_to_system(config, system)
        
        # Check all agents were created
        assert len(system.state.agents) == 6
        
        # Check stock was created
        assert len(system.state.stocks) > 0
        stocks = [s for s in system.state.stocks.values() if s.owner_id == "F1"]
        assert len(stocks) == 1
        assert stocks[0].sku == "WIDGET"
        assert stocks[0].quantity == 100
        
        # Check payable was created
        # Payable uses liability_issuer_id for the debtor
        payables = [c for c in system.state.contracts.values() 
                   if hasattr(c, 'liability_issuer_id') and c.liability_issuer_id == "H1"]
        assert len(payables) > 0
        
        # System should pass invariants
        system.assert_invariants()
    
    def test_apply_with_delivery_obligation(self):
        """Test applying configuration with delivery obligations."""
        config = ScenarioConfig(
            name="Delivery",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "F1", "kind": "firm", "name": "Firm"},
                {"id": "H1", "kind": "household", "name": "Household"}
            ],
            initial_actions=[
                {"mint_cash": {"to": "F1", "amount": 1000}},
                {"mint_cash": {"to": "H1", "amount": 1000}},
                {"create_stock": {"owner": "F1", "sku": "ITEM", "quantity": 10, "unit_price": "50"}},
                {"create_delivery_obligation": {
                    "from": "F1", "to": "H1",
                    "sku": "ITEM", "quantity": 5,
                    "unit_price": "50", "due_day": 2
                }}
            ]
        )
        
        system = System()
        apply_to_system(config, system)
        
        # Check delivery obligation was created
        obligations = [c for c in system.state.contracts.values()
                      if hasattr(c, 'sku')]
        assert len(obligations) > 0
        
        # System should pass invariants
        system.assert_invariants()

    def test_apply_transfer_claim(self):
        """Test transfer_claim action updates asset holder."""
        config = ScenarioConfig(
            name="Transfer Claim",
            agents=[
                {"id": "F1", "kind": "firm", "name": "Firm 1"},
                {"id": "F2", "kind": "firm", "name": "Firm 2"},
            ],
            initial_actions=[
                {"create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 1, "alias": "P1"}},
                {"transfer_claim": {"contract_alias": "P1", "to_agent": "F1"}},
            ],
        )

        system = System()
        apply_to_system(config, system)

        contract_id = system.state.aliases["P1"]
        contract = system.state.contracts[contract_id]
        assert contract.asset_holder_id == "F1"
        assert contract_id in system.state.agents["F1"].asset_ids
        assert contract_id not in system.state.agents["F2"].asset_ids

        system.assert_invariants()
