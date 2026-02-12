"""Tests for configuration models."""

import pytest
from decimal import Decimal
from pydantic import ValidationError

from bilancio.config.models import (
    AgentSpec,
    MintCash,
    MintReserves,
    DepositCash,
    CreateStock,
    CreateDeliveryObligation,
    CreatePayable,
    ScenarioConfig,
    RunConfig,
    PolicyOverrides
)


class TestAgentSpec:
    """Test AgentSpec model validation."""
    
    def test_valid_agent_spec(self):
        """Test creating a valid agent specification."""
        agent = AgentSpec(
            id="B1",
            kind="bank",
            name="First Bank"
        )
        assert agent.id == "B1"
        assert agent.kind == "bank"
        assert agent.name == "First Bank"
    
    def test_invalid_agent_kind(self):
        """Test that invalid agent kind raises error."""
        with pytest.raises(ValidationError):
            AgentSpec(
                id="X1",
                kind="invalid_kind",
                name="Invalid Agent"
            )


class TestActions:
    """Test action model validation."""
    
    def test_mint_cash_valid(self):
        """Test valid mint cash action."""
        action = MintCash(to="H1", amount=Decimal("1000"))
        assert action.to == "H1"
        assert action.amount == Decimal("1000")
        assert action.action == "mint_cash"
    
    def test_mint_cash_negative_amount(self):
        """Test that negative amounts are rejected."""
        with pytest.raises(ValidationError):
            MintCash(to="H1", amount=Decimal("-100"))
    
    def test_deposit_cash_valid(self):
        """Test valid deposit cash action."""
        action = DepositCash(
            customer="H1",
            bank="B1",
            amount=Decimal("500")
        )
        assert action.customer == "H1"
        assert action.bank == "B1"
        assert action.amount == Decimal("500")
    
    def test_create_stock_valid(self):
        """Test valid create stock action."""
        action = CreateStock(
            owner="F1",
            sku="WIDGET",
            quantity=100,
            unit_price=Decimal("25.50")
        )
        assert action.owner == "F1"
        assert action.sku == "WIDGET"
        assert action.quantity == 100
        assert action.unit_price == Decimal("25.50")
    
    def test_create_stock_invalid_quantity(self):
        """Test that zero or negative quantities are rejected."""
        with pytest.raises(ValidationError):
            CreateStock(
                owner="F1",
                sku="WIDGET",
                quantity=0,
                unit_price=Decimal("25")
            )
    
    def test_create_delivery_obligation_with_aliases(self):
        """Test delivery obligation with from/to aliases."""
        # Use the aliases 'from' and 'to' instead of from_agent/to_agent
        data = {
            "from": "F1",
            "to": "H1",
            "sku": "WIDGET",
            "quantity": 10,
            "unit_price": Decimal("25"),
            "due_day": 3
        }
        action = CreateDeliveryObligation(**data)
        assert action.from_agent == "F1"
        assert action.to_agent == "H1"
        assert action.due_day == 3
    
    def test_create_payable_valid(self):
        """Test valid create payable action."""
        # Use the aliases 'from' and 'to' instead of from_agent/to_agent
        data = {
            "from": "H1",
            "to": "H2",
            "amount": Decimal("300"),
            "due_day": 1
        }
        action = CreatePayable(**data)
        assert action.from_agent == "H1"
        assert action.to_agent == "H2"
        assert action.amount == Decimal("300")
        assert action.due_day == 1
    
    def test_create_payable_negative_due_day(self):
        """Test that negative due days are rejected."""
        with pytest.raises(ValidationError):
            CreatePayable(
                from_agent="H1",
                to_agent="H2",
                amount=Decimal("300"),
                due_day=-1
            )


class TestScenarioConfig:
    """Test ScenarioConfig model validation."""
    
    def test_minimal_valid_config(self):
        """Test minimal valid scenario configuration."""
        config = ScenarioConfig(
            name="Test Scenario",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
                {"id": "B1", "kind": "bank", "name": "Bank 1"}
            ]
        )
        assert config.name == "Test Scenario"
        assert config.version == 1
        assert len(config.agents) == 2
        assert config.description is None
        assert config.initial_actions == []
    
    def test_config_with_all_fields(self):
        """Test scenario configuration with all fields."""
        config = ScenarioConfig(
            version=1,
            name="Full Scenario",
            description="A complete test scenario",
            policy_overrides={
                "mop_rank": {
                    "household": ["bank_deposit", "cash"]
                }
            },
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "B1", "kind": "bank", "name": "Bank"}
            ],
            initial_actions=[
                {"mint_reserves": {"to": "B1", "amount": 10000}}
            ],
            run={
                "mode": "until_stable",
                "max_days": 90,
                "quiet_days": 2,
                "show": {
                    "balances": ["CB", "B1"],
                    "events": "detailed"
                },
                "export": {
                    "balances_csv": "balances.csv",
                    "events_jsonl": "events.jsonl"
                }
            }
        )
        assert config.name == "Full Scenario"
        assert config.description == "A complete test scenario"
        assert config.policy_overrides.mop_rank["household"] == ["bank_deposit", "cash"]
        assert config.run.mode == "until_stable"
        assert config.run.max_days == 90
        assert config.run.default_handling == "fail-fast"
        assert config.run.export.balances_csv == "balances.csv"
    
    def test_duplicate_agent_ids_rejected(self):
        """Test that duplicate agent IDs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(
                name="Invalid",
                agents=[
                    {"id": "B1", "kind": "bank", "name": "Bank 1"},
                    {"id": "B1", "kind": "bank", "name": "Bank 2"}
                ]
            )
        assert "unique" in str(exc_info.value).lower()
    
    def test_unsupported_version_rejected(self):
        """Test that unsupported versions are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(
                version=2,
                name="Future Version",
                agents=[
                    {"id": "CB", "kind": "central_bank", "name": "CB"}
                ]
            )
        assert "version" in str(exc_info.value).lower()


class TestRunConfig:
    """Test RunConfig model validation."""
    
    def test_default_run_config(self):
        """Test default run configuration values."""
        config = RunConfig()
        assert config.mode == "until_stable"
        assert config.max_days == 90
        assert config.quiet_days == 2
        assert config.default_handling == "fail-fast"
        assert config.show.events == "detailed"
        assert config.show.balances is None
        assert config.export.balances_csv is None
        assert config.export.events_jsonl is None

    def test_custom_run_config(self):
        """Test custom run configuration."""
        config = RunConfig(
            mode="step",
            max_days=30,
            quiet_days=5,
            default_handling="expel-agent",
            show={"balances": ["A1", "A2"], "events": "summary"},
            export={"balances_csv": "out.csv"}
        )
        assert config.mode == "step"
        assert config.max_days == 30
        assert config.quiet_days == 5
        assert config.default_handling == "expel-agent"
        assert config.show.balances == ["A1", "A2"]
        assert config.show.events == "summary"
        assert config.export.balances_csv == "out.csv"
    
    @pytest.mark.parametrize("field,value", [
        ("max_days", -1),
        ("quiet_days", -1),
    ])
    def test_negative_values_rejected(self, field, value):
        """Test that negative values for numeric fields are rejected."""
        with pytest.raises(ValidationError):
            RunConfig(**{field: value})
