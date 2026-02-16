"""Tests for YAML configuration loading."""

import tempfile
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from bilancio.config.loaders import load_yaml, parse_action, preprocess_config
from bilancio.config.models import (
    CreateDeliveryObligation,
    CreatePayable,
    CreateStock,
    DepositCash,
    MintCash,
    MintReserves,
)


class TestParseAction:
    """Test action parsing from dictionaries."""

    def test_parse_mint_cash(self):
        """Test parsing mint_cash action."""
        action = parse_action({"mint_cash": {"to": "H1", "amount": 1000}})
        assert isinstance(action, MintCash)
        assert action.to == "H1"
        assert action.amount == 1000

    def test_parse_mint_reserves(self):
        """Test parsing mint_reserves action."""
        action = parse_action({"mint_reserves": {"to": "B1", "amount": 5000}})
        assert isinstance(action, MintReserves)
        assert action.to == "B1"
        assert action.amount == 5000

    def test_parse_deposit_cash(self):
        """Test parsing deposit_cash action."""
        action = parse_action({"deposit_cash": {"customer": "H1", "bank": "B1", "amount": 500}})
        assert isinstance(action, DepositCash)
        assert action.customer == "H1"
        assert action.bank == "B1"
        assert action.amount == 500

    def test_parse_create_stock(self):
        """Test parsing create_stock action."""
        action = parse_action(
            {
                "create_stock": {
                    "owner": "F1",
                    "sku": "WIDGET",
                    "quantity": 100,
                    "unit_price": "25.50",
                }
            }
        )
        assert isinstance(action, CreateStock)
        assert action.owner == "F1"
        assert action.sku == "WIDGET"
        assert action.quantity == 100
        assert action.unit_price == Decimal("25.50")

    def test_parse_delivery_obligation_with_aliases(self):
        """Test parsing delivery obligation with from/to aliases."""
        action = parse_action(
            {
                "create_delivery_obligation": {
                    "from": "F1",
                    "to": "H1",
                    "sku": "WIDGET",
                    "quantity": 10,
                    "unit_price": "25",
                    "due_day": 3,
                }
            }
        )
        assert isinstance(action, CreateDeliveryObligation)
        assert action.from_agent == "F1"
        assert action.to_agent == "H1"
        assert action.sku == "WIDGET"

    def test_parse_payable_with_aliases(self):
        """Test parsing payable with from/to aliases."""
        action = parse_action(
            {"create_payable": {"from": "H1", "to": "H2", "amount": 300, "due_day": 1}}
        )
        assert isinstance(action, CreatePayable)
        assert action.from_agent == "H1"
        assert action.to_agent == "H2"
        assert action.amount == 300

    def test_parse_unknown_action(self):
        """Test that unknown action types raise error."""
        with pytest.raises(ValueError) as exc_info:
            parse_action({"unknown_action": {"data": "value"}})
        assert "Unknown action type" in str(exc_info.value)


class TestPreprocessConfig:
    """Test configuration preprocessing."""

    def test_convert_string_decimals(self):
        """Test conversion of string decimals."""
        data = {
            "amount": "123.45",
            "nested": {"price": "99.99"},
            "list": [{"value": "10.50"}, {"value": "20.75"}],
        }

        result = preprocess_config(data)

        assert result["amount"] == Decimal("123.45")
        assert result["nested"]["price"] == Decimal("99.99")
        assert result["list"][0]["value"] == Decimal("10.50")
        assert result["list"][1]["value"] == Decimal("20.75")

    def test_preserve_non_numeric_strings(self):
        """Test that non-numeric strings are preserved."""
        data = {
            "name": "Test Name",
            "id": "ABC123",
            "amount": "100.00",
            "description": "Some text with 123 numbers",
        }

        result = preprocess_config(data)

        assert result["name"] == "Test Name"
        assert result["id"] == "ABC123"
        assert result["amount"] == Decimal("100.00")
        assert result["description"] == "Some text with 123 numbers"


class TestLoadYaml:
    """Test YAML file loading."""

    def test_load_valid_yaml(self):
        """Test loading a valid YAML configuration."""
        yaml_content = """
version: 1
name: Test Scenario
description: A test scenario

agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: First Bank

initial_actions:
  - mint_reserves: {to: B1, amount: 10000}
  - mint_cash: {to: H1, amount: 1000}

run:
  mode: until_stable
  max_days: 30
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            config = load_yaml(temp_path)
            assert config.name == "Test Scenario"
            assert config.version == 1
            assert len(config.agents) == 2
            assert config.agents[0].id == "CB"
            assert config.agents[1].id == "B1"
            assert len(config.initial_actions) == 2
            assert config.run.mode == "until_stable"
            assert config.run.max_days == 30
        finally:
            temp_path.unlink()

    def test_load_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_yaml(Path("nonexistent.yaml"))

    def test_load_invalid_yaml_syntax(self):
        """Test loading file with invalid YAML syntax."""
        yaml_content = """
version: 1
name: Bad YAML
  - this is invalid
    : syntax
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(yaml.YAMLError):
                load_yaml(temp_path)
        finally:
            temp_path.unlink()

    def test_load_invalid_config_schema(self):
        """Test loading YAML with invalid configuration schema."""
        yaml_content = """
version: 1
name: Invalid Config
agents:
  - id: CB
    kind: invalid_kind
    name: Invalid Agent
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                load_yaml(temp_path)
            assert "validation failed" in str(exc_info.value).lower()
        finally:
            temp_path.unlink()

    def test_load_example_scenario(self):
        """Test loading one of the example scenarios."""
        example_path = Path("examples/scenarios/simple_bank.yaml")
        if example_path.exists():
            config = load_yaml(example_path)
            assert config.name == "Simple Banking System"
            assert len(config.agents) == 4  # CB, B1, H1, H2
            assert config.run.mode == "until_stable"
