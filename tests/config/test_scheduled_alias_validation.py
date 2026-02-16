from pathlib import Path

import pytest

from bilancio.config.apply import validate_scheduled_aliases
from bilancio.config.loaders import load_yaml


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


BASE_AGENTS = """
version: 1
name: "Test"
agents:
  - {id: CB, kind: central_bank, name: CB}
  - {id: F1, kind: firm, name: F1}
  - {id: F2, kind: firm, name: F2}
initial_actions: []
run: {mode: until_stable, max_days: 5}
"""


def test_validate_scheduled_aliases_unknown_alias(tmp_path: Path):
    text = (
        BASE_AGENTS
        + """
scheduled_actions:
  - {day: 1, action: {transfer_claim: {contract_alias: NOT_DEFINED, to_agent: F2}}}
"""
    )
    cfg = load_yaml(_write(tmp_path, "c.yaml", text))
    with pytest.raises(ValueError):
        validate_scheduled_aliases(cfg)


def test_validate_scheduled_aliases_duplicate(tmp_path: Path):
    text = (
        BASE_AGENTS
        + """
initial_actions:
  - {create_payable: {from: F1, to: F2, amount: 1, due_day: 1, alias: A1}}
scheduled_actions:
  - {day: 1, action: {create_delivery_obligation: {from: F2, to: F1, sku: X, quantity: 1, unit_price: "1", due_day: 1, alias: A1}}}
"""
    )
    cfg = load_yaml(_write(tmp_path, "d.yaml", text))
    with pytest.raises(ValueError):
        validate_scheduled_aliases(cfg)
