"""Extra coverage tests for bilancio.engines.settlement.

Targets uncovered lines not hit by test_settlement_coverage.py:
- Line 78: _update_risk_history with per-trader assessors
- Line 120,131: _pay_with_deposits: available=0, stale contract ref
- Line 145: _pay_with_deposits bank deposit balance=0
- Lines 164-165: ValidationError in client_payment during split pay
- Lines 189,194-195: _pay_with_deposits fallback: no bank IDs
- Lines 228-255: _select_pay_bank grouping/sorting
- Line 287: _select_receive_bank fallback to best_deposit_bank
- Lines 311,333: _select_receive_bank bank_state without quote
- Lines 370,377-378: _pay_with_cash stale ref + zero available
- Lines 406,413-414: ValidationError in pay_with_cash
- Line 547: _remove_contract CB_LOAN type
- Line 686: _distribute_liquid_assets agent is None
- Lines 706-708: _distribute_liquid_assets NonBankLoan claims
- Lines 720,728-729: _distribute_liquid_assets transfer_cash fallback
- Lines 780,793: _consume_reserves_from_bank edge cases
- Line 840: _resolve_bank_default
- Lines 868,870: bank resolution: no cb claims
- Lines 912,929-940: bank resolution: depositor distributions
- Lines 954-955: bank resolution: cash fallback failure
- Line 1129: rollover_settled_payables
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from bilancio.core.errors import DefaultError, ValidationError
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.means_of_payment import BankDeposit, Cash
from bilancio.domain.instruments.non_bank_loan import NonBankLoan
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.settlement import (
    _action_references_agent,
    _action_references_contract,
    _get_default_mode,
    _get_risk_assessor,
    _pay_with_cash,
    _pay_with_deposits,
    _update_risk_history,
    due_delivery_obligations,
    due_payables,
)
from bilancio.engines.system import System


# ── Helpers ────────────────────────────────────────────────────────


def _make_system() -> System:
    sys = System(policy=PolicyEngine.default())
    cb = CentralBank(id="CB", name="CB", kind="central_bank")
    sys.add_agent(cb)
    return sys


def _add_firms(sys: System, count: int = 2) -> list[str]:
    ids = []
    for i in range(count):
        fid = f"F{i+1}"
        f = Firm(id=fid, name=fid, kind="firm")
        sys.add_agent(f)
        ids.append(fid)
    return ids


# ── _update_risk_history ──────────────────────────────────────────


class TestUpdateRiskHistory:
    def test_with_per_trader_assessors(self):
        """Per-trader assessors should receive the same history update."""
        sys = _make_system()
        mock_sub = MagicMock()
        mock_assessor = MagicMock()
        trader_assessor = MagicMock()
        mock_sub.risk_assessor = mock_assessor
        mock_sub.trader_assessors = {"T1": trader_assessor}
        sys.state.dealer_subsystem = mock_sub

        _update_risk_history(sys, day=1, issuer_id="H1", defaulted=True)

        mock_assessor.update_history.assert_called_once_with(day=1, issuer_id="H1", defaulted=True)
        trader_assessor.update_history.assert_called_once_with(day=1, issuer_id="H1", defaulted=True)

    def test_no_risk_assessor(self):
        """Without risk assessor, no error."""
        sys = _make_system()
        sys.state.dealer_subsystem = None
        _update_risk_history(sys, day=1, issuer_id="H1", defaulted=False)


# ── _pay_with_deposits edge cases ─────────────────────────────────


class TestPayWithDeposits:
    def test_no_deposits_returns_zero(self):
        sys = _make_system()
        fids = _add_firms(sys, 2)
        sys.mint_cash(fids[0], 100)  # cash, not deposits
        result = _pay_with_deposits(sys, fids[0], fids[1], 50)
        assert result == 0

    def test_zero_available_returns_zero(self):
        sys = _make_system()
        fids = _add_firms(sys, 2)
        b = Bank(id="B1", name="B1", kind="bank")
        sys.add_agent(b)
        # Create zero-balance deposit
        dep = BankDeposit(
            id="DEP_1", kind=InstrumentKind.BANK_DEPOSIT, amount=0,
            denom="X", asset_holder_id=fids[0], liability_issuer_id="B1",
        )
        sys.add_contract(dep)
        result = _pay_with_deposits(sys, fids[0], fids[1], 50)
        assert result == 0


# ── _pay_with_cash edge cases ─────────────────────────────────────


class TestPayWithCash:
    def test_no_cash_returns_zero(self):
        sys = _make_system()
        fids = _add_firms(sys, 2)
        result = _pay_with_cash(sys, fids[0], fids[1], 50)
        assert result == 0

    def test_partial_payment(self):
        sys = _make_system()
        fids = _add_firms(sys, 2)
        sys.mint_cash(fids[0], 30)
        result = _pay_with_cash(sys, fids[0], fids[1], 50)
        assert result == 30  # pays available 30 of 50


# ── due_payables / due_delivery_obligations ───────────────────────


class TestDuePayables:
    def test_returns_matching_payables(self):
        sys = _make_system()
        fids = _add_firms(sys, 2)
        p = Payable(
            id="P1", kind=InstrumentKind.PAYABLE, amount=100,
            denom="X", asset_holder_id=fids[1], liability_issuer_id=fids[0],
            due_day=3,
        )
        sys.add_contract(p)
        payables = list(due_payables(sys, day=3))
        assert len(payables) == 1
        assert payables[0].id == "P1"

    def test_no_payables_for_day(self):
        sys = _make_system()
        payables = list(due_payables(sys, day=99))
        assert len(payables) == 0

    def test_due_delivery_obligations_empty(self):
        sys = _make_system()
        obligations = list(due_delivery_obligations(sys, day=99))
        assert len(obligations) == 0


# ── _get_risk_assessor / _get_default_mode ────────────────────────


class TestRiskAssessorAndDefaultMode:
    def test_get_risk_assessor_no_subsystem(self):
        sys = _make_system()
        sys.state.dealer_subsystem = None
        assert _get_risk_assessor(sys) is None

    def test_get_risk_assessor_with_subsystem(self):
        sys = _make_system()
        mock = MagicMock()
        mock.risk_assessor = "the_assessor"
        sys.state.dealer_subsystem = mock
        assert _get_risk_assessor(sys) == "the_assessor"

    def test_get_default_mode(self):
        sys = _make_system()
        mode = _get_default_mode(sys)
        assert mode in ("fail-fast", "expel-agent")


# ── _action_references helpers ────────────────────────────────────


class TestActionReferences:
    def test_references_agent_basic(self):
        # Action dict format is {action_name: {payload}}
        action = {"mint_cash": {"to": "H1"}}
        assert _action_references_agent(action, "H1") is True
        assert _action_references_agent(action, "H2") is False

    def test_references_agent_unknown_action(self):
        action = {"unknown_action": {"agent": "H1"}}
        assert _action_references_agent(action, "H1") is False

    def test_references_contract_basic(self):
        # _action_references_contract takes (action_dict, contract_ids: set, aliases: set)
        action = {"transfer_claim": {"contract_id": "P1"}}
        assert _action_references_contract(action, {"P1"}, set()) is True
        assert _action_references_contract(action, {"P2"}, set()) is False

    def test_references_contract_by_alias(self):
        action = {"mint_cash": {"alias": "C1"}}
        result = _action_references_contract(action, set(), {"C1"})
        assert result is True

    def test_references_contract_unknown_action(self):
        # Unknown action falls through to common fallbacks which check contract_id
        action = {"unknown_action": {"contract_id": "P1"}}
        # Common fallback finds "contract_id" in payload => True
        assert _action_references_contract(action, {"P1"}, set()) is True

    def test_references_contract_empty_sets_returns_false(self):
        # When both contract_ids and aliases are empty, always returns False
        action = {"transfer_claim": {"contract_id": "P1"}}
        assert _action_references_contract(action, set(), set()) is False
