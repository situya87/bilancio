"""Additional coverage tests for config/loaders.py.

Targets: preprocess_config edge cases, parse_action for remaining action types,
load_yaml with non-dict content, and generator path.
"""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from bilancio.config.loaders import load_yaml, parse_action, preprocess_config
from bilancio.config.models import (
    BurnBankCash,
    ClientPayment,
    CreateCBLoan,
    TransferCash,
    TransferClaim,
    TransferReserves,
    TransferStock,
    WithdrawCash,
)


# ============================================================================
# parse_action - additional action types
# ============================================================================


class TestParseActionAdditional:
    """Cover remaining action type branches in parse_action."""

    def test_transfer_cash(self):
        action = parse_action({"transfer_cash": {"from_agent": "H1", "to_agent": "H2", "amount": 500}})
        assert isinstance(action, TransferCash)
        assert action.amount == 500

    def test_transfer_reserves(self):
        action = parse_action(
            {"transfer_reserves": {"from_bank": "B1", "to_bank": "B2", "amount": 3000}}
        )
        assert isinstance(action, TransferReserves)
        assert action.amount == 3000

    def test_withdraw_cash(self):
        action = parse_action(
            {"withdraw_cash": {"customer": "H1", "bank": "B1", "amount": 200}}
        )
        assert isinstance(action, WithdrawCash)
        assert action.customer == "H1"

    def test_client_payment(self):
        action = parse_action(
            {"client_payment": {"payer": "H1", "payee": "H2", "amount": 100}}
        )
        assert isinstance(action, ClientPayment)
        assert action.payer == "H1"

    def test_transfer_stock(self):
        action = parse_action(
            {
                "transfer_stock": {
                    "from_agent": "F1",
                    "to_agent": "H1",
                    "sku": "WIDGET",
                    "quantity": 5,
                }
            }
        )
        assert isinstance(action, TransferStock)
        assert action.sku == "WIDGET"

    def test_create_cb_loan(self):
        action = parse_action(
            {
                "create_cb_loan": {
                    "bank": "B1",
                    "amount": 10000,
                    "rate": "0.05",
                    "maturity_days": 1,
                }
            }
        )
        assert isinstance(action, CreateCBLoan)

    def test_burn_bank_cash(self):
        action = parse_action({"burn_bank_cash": {"bank": "B1", "amount": 500}})
        assert isinstance(action, BurnBankCash)

    def test_transfer_claim(self):
        action = parse_action(
            {
                "transfer_claim": {
                    "contract_id": "c001",
                    "to_agent": "D1",
                }
            }
        )
        assert isinstance(action, TransferClaim)


# ============================================================================
# preprocess_config edge cases
# ============================================================================


class TestPreprocessConfigEdgeCases:
    """Cover edge cases in preprocess_config."""

    def test_infinity_string_not_converted(self):
        data = {"val": "Infinity"}
        result = preprocess_config(data)
        assert result["val"] == "Infinity"

    def test_nan_string_not_converted(self):
        data = {"val": "NaN"}
        result = preprocess_config(data)
        assert result["val"] == "NaN"

    def test_int_passthrough(self):
        data = {"count": 42, "flag": True}
        result = preprocess_config(data)
        assert result["count"] == 42
        assert result["flag"] is True

    def test_nested_list_conversion(self):
        data = {"items": [{"price": "12.50"}, {"price": "7.00"}]}
        result = preprocess_config(data)
        assert result["items"][0]["price"] == Decimal("12.50")
        assert result["items"][1]["price"] == Decimal("7.00")


# ============================================================================
# load_yaml edge cases
# ============================================================================


class TestLoadYamlEdgeCases:
    """Cover edge cases in load_yaml."""

    def test_non_dict_content_raises_valueerror(self, tmp_path):
        """YAML file that contains a list instead of dict."""
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="YAML dictionary"):
            load_yaml(yaml_path)

    def test_generator_config(self, tmp_path):
        """Test YAML with generator key."""
        yaml_content = """
generator: ring_explorer_v1
name_prefix: "Test Gen"
params:
  n_agents: 3
  seed: 42
  kappa: "1.0"
  Q_total: "300"
  maturity:
    days: 2
    mode: lead_lag
    mu: "0"
"""
        yaml_path = tmp_path / "gen.yaml"
        yaml_path.write_text(yaml_content)
        config = load_yaml(yaml_path)
        assert config.name is not None
        assert len(config.agents) >= 3
