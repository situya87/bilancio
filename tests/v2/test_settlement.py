"""Behavioral tests for v2 settlement: payment priority, defaults, recovery."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bilancio.config.models import ScenarioConfig
from bilancio.core.errors import DefaultError
from bilancio_v2 import run_scenario


def make_config(payable_amount: int, default_handling: str) -> ScenarioConfig:
    return ScenarioConfig.model_validate(
        {
            "version": 1,
            "name": "settlement-test",
            "agents": [
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "B1", "kind": "bank", "name": "Bank"},
                {"id": "DEBTOR", "kind": "household", "name": "Debtor"},
                {"id": "CRED1", "kind": "household", "name": "Creditor 1"},
                {"id": "CRED2", "kind": "household", "name": "Creditor 2"},
            ],
            "initial_actions": [
                {"mint_reserves": {"to": "B1", "amount": 10_000}},
                {"mint_cash": {"to": "DEBTOR", "amount": 500}},
                {"deposit_cash": {"customer": "DEBTOR", "bank": "B1", "amount": 300}},
                {
                    "create_payable": {
                        "from": "DEBTOR",
                        "to": "CRED1",
                        "amount": payable_amount,
                        "due_day": 1,
                    }
                },
                {"create_payable": {"from": "DEBTOR", "to": "CRED2", "amount": 100, "due_day": 3}},
            ],
            "run": {
                "max_days": 10,
                "quiet_days": 2,
                "default_handling": default_handling,
            },
        }
    )


def test_household_pays_deposits_before_cash() -> None:
    result = run_scenario(make_config(payable_amount=400, default_handling="fail-fast"))
    ledger = result.ledger
    # 300 deposit drained first, the remaining 100 paid in cash.
    assert ledger.deposits[("DEBTOR", "B1")] == Decimal(0)
    assert ledger.deposits[("CRED1", "B1")] == Decimal(300)
    assert ledger.cash["CRED1"] == Decimal(100)
    kinds = [event["kind"] for event in result.events]
    assert "PayableSettled" in kinds
    assert "ObligationDefaulted" not in kinds


def test_fail_fast_shortfall_raises() -> None:
    with pytest.raises(DefaultError):
        run_scenario(make_config(payable_amount=600, default_handling="fail-fast"))


def test_expel_default_recovers_pro_rata_and_writes_off() -> None:
    result = run_scenario(make_config(payable_amount=600, default_handling="expel-agent"))
    ledger = result.ledger
    assert "DEBTOR" in ledger.defaulted_agent_ids

    kinds = [event["kind"] for event in result.events]
    assert "PartialSettlement" in kinds  # 500 of 600 paid before the shortfall
    assert "ObligationDefaulted" in kinds
    assert "AgentDefaulted" in kinds
    assert "ObligationWrittenOff" in kinds  # the day-3 payable to CRED2

    # The debtor is fully drained; everything it had went to creditors.
    assert ledger.cash["DEBTOR"] == Decimal(0)
    assert ledger.deposits[("DEBTOR", "B1")] == Decimal(0)
    assert ledger.cash["CRED1"] + ledger.deposits[("CRED1", "B1")] == Decimal(500)

    # Invariants hold at the end of a default cascade.
    ledger.check_invariants()
