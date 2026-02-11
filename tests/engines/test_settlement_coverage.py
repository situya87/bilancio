"""Additional tests for settlement.py to cover uncovered branches and edge cases.

Focuses on:
- _pay_with_deposits: stale refs, zero available, creditor without deposit
- _pay_with_cash: stale refs, zero available, ValidationError fallback
- _pay_bank_to_bank_with_reserves: same bank, no reserves, zero reserves, ValidationError
- _deliver_stock: no stock, zero quantity, partial delivery, FIFO, ValidationError
- _remove_contract: already removed, Payable with effective_creditor, due_day cleanup, cash/reserve tracking
- _action_references_agent: malformed input, list agent references
- _action_references_contract: malformed input, common fallback keys
- _cancel_scheduled_actions_for_agent: no scheduled actions, contract-based cancellation
- _expel_agent: wrong mode, already defaulted, missing agent, CB default, delivery obligation write-off
- settle_due_delivery_obligations: full settlement, partial, fail-fast, defaulted debtor
- _settle_single_payable: defaulted debtor skip, rollover info
- _handle_payable_default: fail-fast raise, partial with distribution
- settle_due: payable removed mid-iteration, rollover
- rollover_settled_payables: defaulted parties, partial rollover, Model C logic
"""

import pytest
from decimal import Decimal

from bilancio.core.errors import DefaultError, ValidationError
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.delivery import DeliveryObligation
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.settlement import (
    DEFAULT_MODE_EXPEL,
    DEFAULT_MODE_FAIL_FAST,
    _action_references_agent,
    _action_references_contract,
    _cancel_scheduled_actions_for_agent,
    _deliver_stock,
    _expel_agent,
    _handle_payable_default,
    _pay_bank_to_bank_with_reserves,
    _pay_with_cash,
    _pay_with_deposits,
    _remove_contract,
    due_delivery_obligations,
    due_payables,
    rollover_settled_payables,
    settle_due,
    settle_due_delivery_obligations,
)
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system_with_firms(default_mode: str = "fail-fast") -> tuple:
    """Create a basic system with CB + 2 firms."""
    system = System(default_mode=default_mode)
    cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
    d = Firm(id="D1", name="Debtor", kind="firm")
    c = Firm(id="C1", name="Creditor", kind="firm")
    system.add_agent(cb)
    system.add_agent(d)
    system.add_agent(c)
    return system, cb, d, c


def _make_payable(system, debtor, creditor, amount, due_day, maturity_distance=None):
    p = Payable(
        id=system.new_contract_id("PAY"),
        kind=InstrumentKind.PAYABLE,
        amount=amount,
        denom="X",
        asset_holder_id=creditor.id,
        liability_issuer_id=debtor.id,
        due_day=due_day,
        maturity_distance=maturity_distance,
    )
    system.add_contract(p)
    return p


# ===========================================================================
# _pay_with_deposits edge cases
# ===========================================================================


class TestPayWithDeposits:
    def test_stale_asset_reference_skipped(self):
        """If debtor has a stale asset_id (contract removed), it should be skipped."""
        system, cb, debtor, creditor = _system_with_firms()
        # Add a stale reference
        debtor.asset_ids.append("STALE_ID")
        result = _pay_with_deposits(system, debtor.id, creditor.id, 100)
        assert result == 0

    def test_zero_deposit_balance(self):
        """Debtor has a bank deposit with amount=0 -> returns 0."""
        system = System()
        cb = CentralBank(id="CB", name="CB", kind="central_bank")
        bank = Bank(id="B1", name="Bank", kind="bank")
        debtor = Household(id="H1", name="H1", kind="household")
        creditor = Household(id="H2", name="H2", kind="household")
        system.add_agent(cb)
        system.add_agent(bank)
        system.add_agent(debtor)
        system.add_agent(creditor)

        # Mint and deposit, then withdraw all => zero balance deposit
        system.mint_cash("H1", 100)
        deposit_cash(system, "H1", "B1", 100)
        # Now withdraw all the cash back
        from bilancio.ops.banking import withdraw_cash
        withdraw_cash(system, "H1", "B1", 100)

        result = _pay_with_deposits(system, debtor.id, creditor.id, 50)
        assert result == 0

    def test_creditor_no_deposit_uses_debtor_bank(self):
        """When creditor has no deposit, creditor_bank_id falls back to debtor_bank_id."""
        system = System()
        cb = CentralBank(id="CB", name="CB", kind="central_bank")
        bank = Bank(id="B1", name="Bank", kind="bank")
        debtor = Household(id="H1", name="H1", kind="household")
        creditor = Household(id="H2", name="H2", kind="household")
        system.add_agent(cb)
        system.add_agent(bank)
        system.add_agent(debtor)
        system.add_agent(creditor)

        system.mint_cash("H1", 200)
        deposit_cash(system, "H1", "B1", 200)

        # Creditor has NO deposit at any bank
        result = _pay_with_deposits(system, debtor.id, creditor.id, 100)
        assert result == 100
        # Creditor should now have deposit at B1 (debtor's bank)
        assert system.total_deposit("H2", "B1") == 100

    def test_stale_creditor_asset_reference_skipped(self):
        """Stale reference in creditor's asset_ids is skipped gracefully."""
        system = System()
        cb = CentralBank(id="CB", name="CB", kind="central_bank")
        bank = Bank(id="B1", name="Bank", kind="bank")
        debtor = Household(id="H1", name="H1", kind="household")
        creditor = Household(id="H2", name="H2", kind="household")
        system.add_agent(cb)
        system.add_agent(bank)
        system.add_agent(debtor)
        system.add_agent(creditor)

        system.mint_cash("H1", 200)
        deposit_cash(system, "H1", "B1", 200)

        # Add stale ref to creditor
        creditor.asset_ids.append("STALE_REF")

        result = _pay_with_deposits(system, debtor.id, creditor.id, 50)
        assert result == 50


# ===========================================================================
# _pay_with_cash edge cases
# ===========================================================================


class TestPayWithCash:
    def test_no_cash_returns_zero(self):
        """Debtor with no cash instruments returns 0."""
        system, _, debtor, creditor = _system_with_firms()
        result = _pay_with_cash(system, debtor.id, creditor.id, 100)
        assert result == 0

    def test_stale_asset_reference_skipped(self):
        """Stale asset_id in debtor's list is gracefully skipped."""
        system, _, debtor, creditor = _system_with_firms()
        debtor.asset_ids.append("STALE_CASH")
        result = _pay_with_cash(system, debtor.id, creditor.id, 100)
        assert result == 0

    def test_zero_cash_balance(self):
        """Debtor has cash instrument with amount=0 -> returns 0."""
        system, _, debtor, creditor = _system_with_firms()
        system.mint_cash(debtor.id, 100)
        system.retire_cash(debtor.id, 100)
        # Debtor may still have a 0-amount cash instrument
        result = _pay_with_cash(system, debtor.id, creditor.id, 50)
        assert result == 0

    def test_partial_cash_available(self):
        """Debtor has less cash than required."""
        system, _, debtor, creditor = _system_with_firms()
        system.mint_cash(debtor.id, 30)
        result = _pay_with_cash(system, debtor.id, creditor.id, 50)
        assert result == 30


# ===========================================================================
# _pay_bank_to_bank_with_reserves edge cases
# ===========================================================================


class TestPayBankToBank:
    def test_same_bank_returns_zero(self):
        """Same bank -> returns 0."""
        system, _, _, _ = _system_with_firms()
        bank = Bank(id="B1", name="Bank", kind="bank")
        system.add_agent(bank)
        result = _pay_bank_to_bank_with_reserves(system, "B1", "B1", 100)
        assert result == 0

    def test_no_reserves_returns_zero(self):
        """Bank with no reserves -> returns 0."""
        system, _, _, _ = _system_with_firms()
        b1 = Bank(id="B1", name="B1", kind="bank")
        b2 = Bank(id="B2", name="B2", kind="bank")
        system.add_agent(b1)
        system.add_agent(b2)
        result = _pay_bank_to_bank_with_reserves(system, "B1", "B2", 100)
        assert result == 0

    def test_stale_reserve_reference_skipped(self):
        """Stale asset_id in bank's list is gracefully skipped."""
        system, _, _, _ = _system_with_firms()
        b1 = Bank(id="B1", name="B1", kind="bank")
        b2 = Bank(id="B2", name="B2", kind="bank")
        system.add_agent(b1)
        system.add_agent(b2)
        b1.asset_ids.append("STALE_RESERVE")
        result = _pay_bank_to_bank_with_reserves(system, "B1", "B2", 100)
        assert result == 0

    def test_zero_reserve_balance(self):
        """Bank has a reserve with amount=0 -> returns 0."""
        system, _, _, _ = _system_with_firms()
        b1 = Bank(id="B1", name="B1", kind="bank")
        b2 = Bank(id="B2", name="B2", kind="bank")
        system.add_agent(b1)
        system.add_agent(b2)
        system.mint_reserves("B1", 50)
        # Consume reserves to bring them to zero (transfer all to B2 then we start from 0)
        system.transfer_reserves("B1", "B2", 50)
        result = _pay_bank_to_bank_with_reserves(system, "B1", "B2", 100)
        assert result == 0

    def test_partial_reserves_available(self):
        """Bank has fewer reserves than required."""
        system, _, _, _ = _system_with_firms()
        b1 = Bank(id="B1", name="B1", kind="bank")
        b2 = Bank(id="B2", name="B2", kind="bank")
        system.add_agent(b1)
        system.add_agent(b2)
        system.mint_reserves("B1", 30)
        result = _pay_bank_to_bank_with_reserves(system, "B1", "B2", 100)
        assert result == 30


# ===========================================================================
# _deliver_stock edge cases
# ===========================================================================


class TestDeliverStock:
    def test_no_stock_returns_zero(self):
        """Debtor has no stock of the required SKU."""
        system, _, debtor, creditor = _system_with_firms()
        result = _deliver_stock(system, debtor.id, creditor.id, "WIDGET", 10)
        assert result == 0

    def test_zero_quantity_stock(self):
        """Debtor has stock with quantity=0."""
        system, _, debtor, creditor = _system_with_firms()
        system.create_stock(debtor.id, "WIDGET", 0, Decimal("10"))
        result = _deliver_stock(system, debtor.id, creditor.id, "WIDGET", 10)
        assert result == 0

    def test_partial_stock_delivery(self):
        """Debtor has less stock than required."""
        system, _, debtor, creditor = _system_with_firms()
        system.create_stock(debtor.id, "WIDGET", 5, Decimal("10"))
        result = _deliver_stock(system, debtor.id, creditor.id, "WIDGET", 10)
        assert result == 5

    def test_fifo_delivery_across_lots(self):
        """Multiple stock lots delivered in FIFO order."""
        system, _, debtor, creditor = _system_with_firms()
        system.create_stock(debtor.id, "WIDGET", 3, Decimal("10"))
        system.create_stock(debtor.id, "WIDGET", 4, Decimal("10"))
        result = _deliver_stock(system, debtor.id, creditor.id, "WIDGET", 6)
        assert result == 6

    def test_wrong_sku_ignored(self):
        """Stock of different SKU is not delivered."""
        system, _, debtor, creditor = _system_with_firms()
        system.create_stock(debtor.id, "GADGET", 10, Decimal("10"))
        result = _deliver_stock(system, debtor.id, creditor.id, "WIDGET", 5)
        assert result == 0

    def test_exact_delivery(self):
        """Debtor has exactly the required amount."""
        system, _, debtor, creditor = _system_with_firms()
        system.create_stock(debtor.id, "WIDGET", 10, Decimal("10"))
        result = _deliver_stock(system, debtor.id, creditor.id, "WIDGET", 10)
        assert result == 10


# ===========================================================================
# _remove_contract edge cases
# ===========================================================================


class TestRemoveContract:
    def test_already_removed(self):
        """Removing a non-existent contract is a no-op."""
        system, _, _, _ = _system_with_firms()
        _remove_contract(system, "NONEXISTENT")
        # Should not raise

    def test_remove_cash_updates_outstanding(self):
        """Removing a cash contract decrements cb_cash_outstanding."""
        system, _, debtor, _ = _system_with_firms()
        cid = system.mint_cash(debtor.id, 100)
        assert system.state.cb_cash_outstanding == 100
        _remove_contract(system, cid)
        assert system.state.cb_cash_outstanding == 0

    def test_remove_reserve_updates_outstanding(self):
        """Removing a reserve contract decrements cb_reserves_outstanding."""
        system, _, _, _ = _system_with_firms()
        bank = Bank(id="B1", name="Bank", kind="bank")
        system.add_agent(bank)
        cid = system.mint_reserves("B1", 200)
        assert system.state.cb_reserves_outstanding == 200
        _remove_contract(system, cid)
        assert system.state.cb_reserves_outstanding == 0

    def test_remove_payable_with_due_day_index(self):
        """Removing a payable clears the due_day index entry."""
        system, _, debtor, creditor = _system_with_firms()
        p = _make_payable(system, debtor, creditor, 100, due_day=5)
        assert p.id in system.state.contracts_by_due_day.get(5, [])
        _remove_contract(system, p.id)
        # Due day bucket should be empty and removed
        assert 5 not in system.state.contracts_by_due_day

    def test_remove_payable_with_effective_creditor_different(self):
        """Removing a Payable where holder_id != asset_holder_id (secondary market transfer)."""
        system, _, debtor, creditor = _system_with_firms()
        third = Firm(id="T1", name="Third", kind="firm")
        system.add_agent(third)
        p = _make_payable(system, debtor, creditor, 100, due_day=5)
        # Simulate secondary market transfer: holder_id set, asset moved
        p.holder_id = third.id
        creditor.asset_ids.remove(p.id)
        third.asset_ids.append(p.id)
        _remove_contract(system, p.id)
        assert p.id not in system.state.contracts
        assert p.id not in third.asset_ids

    def test_remove_contract_cleans_original_holder_too(self):
        """When effective_creditor differs from asset_holder_id, both are cleaned."""
        system, _, debtor, creditor = _system_with_firms()
        third = Firm(id="T1", name="Third", kind="firm")
        system.add_agent(third)
        p = _make_payable(system, debtor, creditor, 100, due_day=5)
        # Simulate: holder_id set but asset_ids not cleaned (inconsistent state)
        p.holder_id = third.id
        # Don't remove from creditor's asset_ids (simulate inconsistency)
        third.asset_ids.append(p.id)
        _remove_contract(system, p.id)
        assert p.id not in creditor.asset_ids
        assert p.id not in third.asset_ids


# ===========================================================================
# _action_references_agent edge cases
# ===========================================================================


class TestActionReferencesAgent:
    def test_non_dict_returns_false(self):
        assert _action_references_agent("not_a_dict", "A1") is False

    def test_multi_key_dict_returns_false(self):
        assert _action_references_agent({"a": {}, "b": {}}, "A1") is False

    def test_non_dict_payload_returns_false(self):
        assert _action_references_agent({"mint_cash": "string_payload"}, "A1") is False

    def test_matching_agent_in_field(self):
        action = {"mint_cash": {"to": "A1", "amount": 100}}
        assert _action_references_agent(action, "A1") is True

    def test_non_matching_agent(self):
        action = {"mint_cash": {"to": "A2", "amount": 100}}
        assert _action_references_agent(action, "A1") is False

    def test_agent_in_list_field(self):
        """Agent ID found inside a list value for a field."""
        action = {"create_delivery_obligation": {"from": ["A1", "A2"], "to": "A3"}}
        assert _action_references_agent(action, "A1") is True

    def test_agent_not_in_list_field(self):
        action = {"create_delivery_obligation": {"from": ["A2", "A3"], "to": "A4"}}
        assert _action_references_agent(action, "A1") is False

    def test_unknown_action_name(self):
        """Unknown action name has no registered fields -> returns False."""
        action = {"unknown_action": {"agent": "A1"}}
        assert _action_references_agent(action, "A1") is False

    def test_transfer_reserves_fields(self):
        action = {"transfer_reserves": {"from_bank": "B1", "to_bank": "B2"}}
        assert _action_references_agent(action, "B1") is True
        assert _action_references_agent(action, "B2") is True
        assert _action_references_agent(action, "B3") is False


# ===========================================================================
# _action_references_contract edge cases
# ===========================================================================


class TestActionReferencesContract:
    def test_non_dict_returns_false(self):
        assert _action_references_contract("not_a_dict", {"C1"}, set()) is False

    def test_multi_key_dict_returns_false(self):
        assert _action_references_contract({"a": {}, "b": {}}, {"C1"}, set()) is False

    def test_non_dict_payload_returns_false(self):
        assert _action_references_contract({"mint_cash": "payload"}, {"C1"}, set()) is False

    def test_empty_sets_returns_false(self):
        action = {"mint_cash": {"alias": "ALIAS1"}}
        assert _action_references_contract(action, set(), set()) is False

    def test_matching_contract_id_in_registered_field(self):
        action = {"transfer_claim": {"contract_id": "C1", "to_agent": "A1"}}
        assert _action_references_contract(action, {"C1"}, set()) is True

    def test_matching_alias_in_registered_field(self):
        action = {"transfer_claim": {"contract_alias": "PAY1", "to_agent": "A1"}}
        assert _action_references_contract(action, set(), {"PAY1"}) is True

    def test_matching_via_common_fallback_keys(self):
        """Unregistered action name, but common fallback 'alias' key matches."""
        action = {"some_action": {"alias": "PAY1"}}
        assert _action_references_contract(action, set(), {"PAY1"}) is True

    def test_matching_via_fallback_contract_id_key(self):
        action = {"some_action": {"contract_id": "C2"}}
        assert _action_references_contract(action, {"C2"}, set()) is True

    def test_no_match(self):
        action = {"some_action": {"other_field": "value"}}
        assert _action_references_contract(action, {"C1"}, {"PAY1"}) is False


# ===========================================================================
# _cancel_scheduled_actions_for_agent
# ===========================================================================


class TestCancelScheduledActions:
    def test_no_scheduled_actions_noop(self):
        """No scheduled actions means nothing to cancel."""
        system, _, debtor, _ = _system_with_firms(default_mode="expel-agent")
        _cancel_scheduled_actions_for_agent(system, debtor.id)
        # Should not raise

    def test_cancels_by_contract_reference(self):
        """Actions referencing cancelled contracts are also removed."""
        system, _, debtor, creditor = _system_with_firms(default_mode="expel-agent")
        system.state.scheduled_actions_by_day[5] = [
            {"transfer_claim": {"contract_id": "C99", "to_agent": creditor.id}},
            {"mint_cash": {"to": "SOMEONE_ELSE", "amount": 10}},
        ]
        _cancel_scheduled_actions_for_agent(
            system, debtor.id,
            cancelled_contract_ids={"C99"},
        )
        # Only the mint_cash action should remain
        assert len(system.state.scheduled_actions_by_day.get(5, [])) == 1

    def test_removes_empty_day_entries(self):
        """If all actions for a day are cancelled, the day entry is deleted."""
        system, _, debtor, _ = _system_with_firms(default_mode="expel-agent")
        system.state.scheduled_actions_by_day[5] = [
            {"mint_cash": {"to": debtor.id, "amount": 10}},
        ]
        _cancel_scheduled_actions_for_agent(system, debtor.id)
        assert 5 not in system.state.scheduled_actions_by_day

    def test_cancels_by_alias_reference(self):
        """Actions referencing cancelled aliases are removed."""
        system, _, debtor, creditor = _system_with_firms(default_mode="expel-agent")
        system.state.scheduled_actions_by_day[5] = [
            {"transfer_claim": {"contract_alias": "MY_PAY", "to_agent": creditor.id}},
        ]
        _cancel_scheduled_actions_for_agent(
            system, debtor.id,
            cancelled_aliases={"MY_PAY"},
        )
        assert 5 not in system.state.scheduled_actions_by_day


# ===========================================================================
# _expel_agent edge cases
# ===========================================================================


class TestExpelAgent:
    def test_wrong_mode_is_noop(self):
        """fail-fast mode -> _expel_agent does nothing."""
        system, _, debtor, _ = _system_with_firms(default_mode="fail-fast")
        _expel_agent(system, debtor.id)
        assert debtor.id not in system.state.defaulted_agent_ids

    def test_already_defaulted_is_noop(self):
        """Already defaulted agent -> _expel_agent returns early."""
        system, _, debtor, _ = _system_with_firms(default_mode="expel-agent")
        debtor.defaulted = True
        system.state.defaulted_agent_ids.add(debtor.id)
        events_before = len(system.state.events)
        _expel_agent(system, debtor.id)
        assert len(system.state.events) == events_before  # no new events

    def test_missing_agent_is_noop(self):
        """Non-existent agent -> _expel_agent returns early."""
        system, _, _, _ = _system_with_firms(default_mode="expel-agent")
        _expel_agent(system, "NONEXISTENT")
        assert "NONEXISTENT" not in system.state.defaulted_agent_ids

    def test_central_bank_default_raises(self):
        """Central bank cannot default -> raises DefaultError."""
        system, cb, _, _ = _system_with_firms(default_mode="expel-agent")
        with pytest.raises(DefaultError, match="Central bank cannot default"):
            _expel_agent(system, cb.id)

    def test_writes_off_delivery_obligations(self):
        """DeliveryObligation liabilities are written off with SKU in event."""
        system, _, debtor, creditor = _system_with_firms(default_mode="expel-agent")
        # Add a third firm so not all non-CB agents default
        extra = Firm(id="E1", name="Extra", kind="firm")
        system.add_agent(extra)

        # Create delivery obligation: debtor owes creditor
        d_id = system.create_delivery_obligation(
            debtor.id, creditor.id, sku="WIDGET", quantity=10,
            unit_price=Decimal("5"), due_day=3,
        )

        _expel_agent(system, debtor.id, trigger_contract_id="TRIGGER")
        written_off = [e for e in system.state.events if e["kind"] == "ObligationWrittenOff"]
        assert any(e.get("sku") == "WIDGET" for e in written_off)

    def test_all_agents_default_raises(self):
        """If all non-CB agents have defaulted, raises DefaultError."""
        system, _, debtor, creditor = _system_with_firms(default_mode="expel-agent")
        # Default creditor first
        creditor.defaulted = True
        system.state.defaulted_agent_ids.add(creditor.id)
        # Now default debtor -> all non-CB agents defaulted
        with pytest.raises(DefaultError, match="All non-central-bank agents have defaulted"):
            _expel_agent(system, debtor.id)

    def test_skips_trigger_contract_during_writeoff(self):
        """The trigger contract itself is not written off again."""
        system, _, debtor, creditor = _system_with_firms(default_mode="expel-agent")
        extra = Firm(id="E1", name="Extra", kind="firm")
        system.add_agent(extra)

        p1 = _make_payable(system, debtor, creditor, 100, due_day=1)
        p2 = _make_payable(system, debtor, creditor, 50, due_day=2)

        _expel_agent(system, debtor.id, trigger_contract_id=p1.id)
        written_off_ids = [e["contract_id"] for e in system.state.events if e["kind"] == "ObligationWrittenOff"]
        assert p2.id in written_off_ids
        assert p1.id not in written_off_ids  # trigger is skipped

    def test_cancelled_aliases_removed_from_state(self):
        """Pre-existing cancelled aliases are removed from system aliases."""
        system, _, debtor, creditor = _system_with_firms(default_mode="expel-agent")
        extra = Firm(id="E1", name="Extra", kind="firm")
        system.add_agent(extra)

        system.state.aliases["OLD_ALIAS"] = "OLD_CONTRACT"

        _expel_agent(
            system, debtor.id,
            cancelled_aliases={"OLD_ALIAS"},
        )
        assert "OLD_ALIAS" not in system.state.aliases

    def test_obligation_without_due_day(self):
        """Written-off obligations with no due_day omit it from the event."""
        system, _, debtor, creditor = _system_with_firms(default_mode="expel-agent")
        extra = Firm(id="E1", name="Extra", kind="firm")
        system.add_agent(extra)

        # Create a payable with due_day=None (unusual but possible)
        p = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=50,
            denom="X",
            asset_holder_id=creditor.id,
            liability_issuer_id=debtor.id,
            due_day=0,  # must be >= 0 for validation
        )
        system.add_contract(p)
        # Manually set due_day to None after creation to test the branch
        p.due_day = None

        _expel_agent(system, debtor.id)
        written = [e for e in system.state.events if e["kind"] == "ObligationWrittenOff"]
        assert len(written) >= 1
        # The event should not contain due_day key if it was None
        for w in written:
            if w["contract_id"] == p.id:
                assert "due_day" not in w


# ===========================================================================
# settle_due_delivery_obligations
# ===========================================================================


class TestSettleDueDeliveryObligations:
    def test_successful_delivery(self):
        """Full delivery settlement succeeds and logs event."""
        system, _, debtor, creditor = _system_with_firms()
        system.create_stock(debtor.id, "WIDGET", 10, Decimal("5"))
        d_id = system.create_delivery_obligation(
            debtor.id, creditor.id, sku="WIDGET", quantity=10,
            unit_price=Decimal("5"), due_day=1,
        )
        settle_due_delivery_obligations(system, 1)
        assert d_id not in system.state.contracts
        settled_events = [e for e in system.state.events if e["kind"] == "DeliveryObligationSettled"]
        assert len(settled_events) == 1
        assert settled_events[0]["sku"] == "WIDGET"
        assert settled_events[0]["qty"] == 10

    def test_partial_delivery_fail_fast(self):
        """Partial delivery in fail-fast mode raises DefaultError."""
        system, _, debtor, creditor = _system_with_firms()
        system.create_stock(debtor.id, "WIDGET", 5, Decimal("5"))
        system.create_delivery_obligation(
            debtor.id, creditor.id, sku="WIDGET", quantity=10,
            unit_price=Decimal("5"), due_day=1,
        )
        with pytest.raises(DefaultError, match="Insufficient stock"):
            settle_due_delivery_obligations(system, 1)

    def test_partial_delivery_expel_mode(self):
        """Partial delivery in expel mode logs events and expels agent."""
        system, _, debtor, creditor = _system_with_firms(default_mode="expel-agent")
        # Extra agent so not all default
        extra = Firm(id="E1", name="Extra", kind="firm")
        system.add_agent(extra)

        system.create_stock(debtor.id, "WIDGET", 3, Decimal("5"))
        system.create_delivery_obligation(
            debtor.id, creditor.id, sku="WIDGET", quantity=10,
            unit_price=Decimal("5"), due_day=1,
        )
        settle_due_delivery_obligations(system, 1)

        assert debtor.defaulted is True
        default_events = [e for e in system.state.events if e["kind"] == "ObligationDefaulted"]
        assert len(default_events) == 1
        assert default_events[0]["shortfall"] == 7
        assert default_events[0]["sku"] == "WIDGET"

        partial_events = [e for e in system.state.events if e["kind"] == "PartialSettlement"]
        assert len(partial_events) == 1
        assert partial_events[0]["delivered_quantity"] == 3

    def test_no_delivery_expel_mode(self):
        """Zero delivery in expel mode still logs default (no partial event)."""
        system, _, debtor, creditor = _system_with_firms(default_mode="expel-agent")
        extra = Firm(id="E1", name="Extra", kind="firm")
        system.add_agent(extra)

        system.create_delivery_obligation(
            debtor.id, creditor.id, sku="WIDGET", quantity=10,
            unit_price=Decimal("5"), due_day=1,
        )
        settle_due_delivery_obligations(system, 1)

        assert debtor.defaulted is True
        partial_events = [e for e in system.state.events if e["kind"] == "PartialSettlement"]
        assert len(partial_events) == 0  # No partial if zero delivered

    def test_defaulted_debtor_skipped(self):
        """Delivery obligations for already-defaulted debtors are skipped."""
        system, _, debtor, creditor = _system_with_firms(default_mode="expel-agent")
        debtor.defaulted = True
        system.state.defaulted_agent_ids.add(debtor.id)

        d_id = system.create_delivery_obligation(
            debtor.id, creditor.id, sku="WIDGET", quantity=10,
            unit_price=Decimal("5"), due_day=1,
        )
        settle_due_delivery_obligations(system, 1)
        # Obligation should still exist (was skipped)
        assert d_id in system.state.contracts

    def test_removed_obligation_skipped(self):
        """If obligation was removed mid-iteration, it's skipped."""
        system, _, debtor, creditor = _system_with_firms()
        d_id = system.create_delivery_obligation(
            debtor.id, creditor.id, sku="WIDGET", quantity=10,
            unit_price=Decimal("5"), due_day=1,
        )
        # Remove it before settlement
        _remove_contract(system, d_id)
        settle_due_delivery_obligations(system, 1)
        # No events since it was removed


# ===========================================================================
# due_payables / due_delivery_obligations
# ===========================================================================


class TestDueQueries:
    def test_no_payables_due(self):
        """No payables due on given day."""
        system, _, debtor, creditor = _system_with_firms()
        _make_payable(system, debtor, creditor, 100, due_day=5)
        result = list(due_payables(system, 1))
        assert len(result) == 0

    def test_non_payable_in_due_day_index_skipped(self):
        """Non-payable contracts in the due_day index are skipped."""
        system, _, debtor, creditor = _system_with_firms()
        # Create a delivery obligation on the same due_day
        d_id = system.create_delivery_obligation(
            debtor.id, creditor.id, sku="WIDGET", quantity=5,
            unit_price=Decimal("10"), due_day=1,
        )
        result = list(due_payables(system, 1))
        assert len(result) == 0  # delivery obligation is not a payable

    def test_removed_contract_in_due_day_index_skipped(self):
        """Contract removed from system but still in index -> skipped."""
        system, _, debtor, creditor = _system_with_firms()
        p = _make_payable(system, debtor, creditor, 100, due_day=1)
        # Remove from contracts dict but leave in index
        del system.state.contracts[p.id]
        result = list(due_payables(system, 1))
        assert len(result) == 0

    def test_due_delivery_obligations_returns_only_obligations(self):
        """Only delivery obligations are returned, not payables."""
        system, _, debtor, creditor = _system_with_firms()
        _make_payable(system, debtor, creditor, 100, due_day=1)
        d_id = system.create_delivery_obligation(
            debtor.id, creditor.id, sku="WIDGET", quantity=5,
            unit_price=Decimal("10"), due_day=1,
        )
        result = list(due_delivery_obligations(system, 1))
        assert len(result) == 1
        assert result[0].id == d_id


# ===========================================================================
# settle_due integration - edge cases
# ===========================================================================


class TestSettleDueEdgeCases:
    def test_settle_due_with_no_payables(self):
        """No payables due -> returns empty list."""
        system, _, _, _ = _system_with_firms()
        result = settle_due(system, 1)
        assert result == []

    def test_settle_due_defaulted_debtor_skipped(self):
        """Defaulted debtor's payable is skipped silently."""
        system, _, debtor, creditor = _system_with_firms(default_mode="expel-agent")
        extra = Firm(id="E1", name="Extra", kind="firm")
        system.add_agent(extra)

        _make_payable(system, debtor, creditor, 100, due_day=1)
        debtor.defaulted = True
        system.state.defaulted_agent_ids.add(debtor.id)

        result = settle_due(system, 1)
        # Should not raise and should return empty (defaulted debtor skipped)
        assert result == []

    def test_settle_due_rollover_info_returned(self):
        """When rollover_enabled, settled payables return rollover info."""
        system, _, debtor, creditor = _system_with_firms()
        system.mint_cash(debtor.id, 100)
        _make_payable(system, debtor, creditor, 100, due_day=1, maturity_distance=3)

        result = settle_due(system, 1, rollover_enabled=True)
        assert len(result) == 1
        debtor_id, creditor_id, amount, mat_dist = result[0]
        assert debtor_id == debtor.id
        assert creditor_id == creditor.id
        assert amount == 100
        assert mat_dist == 3

    def test_settle_due_no_rollover_info_without_flag(self):
        """Without rollover_enabled, no rollover info returned."""
        system, _, debtor, creditor = _system_with_firms()
        system.mint_cash(debtor.id, 100)
        _make_payable(system, debtor, creditor, 100, due_day=1, maturity_distance=3)

        result = settle_due(system, 1, rollover_enabled=False)
        assert len(result) == 0

    def test_settle_due_no_rollover_info_without_maturity_distance(self):
        """Even with rollover_enabled, no rollover info if maturity_distance is None."""
        system, _, debtor, creditor = _system_with_firms()
        system.mint_cash(debtor.id, 100)
        _make_payable(system, debtor, creditor, 100, due_day=1, maturity_distance=None)

        result = settle_due(system, 1, rollover_enabled=True)
        assert len(result) == 0

    def test_settle_due_removed_payable_skipped(self):
        """Payable removed from contracts mid-iteration is skipped."""
        system, _, debtor, creditor = _system_with_firms()
        p = _make_payable(system, debtor, creditor, 100, due_day=1)
        # Remove from contracts dict (simulate removal by earlier settlement)
        _remove_contract(system, p.id)
        result = settle_due(system, 1)
        assert result == []

    def test_settle_due_risk_assessor_updated_on_success(self):
        """Risk assessor gets updated when payable settles successfully."""

        class MockRiskAssessor:
            def __init__(self):
                self.calls = []
            def update_history(self, day, issuer_id, defaulted):
                self.calls.append((day, issuer_id, defaulted))

        system, _, debtor, creditor = _system_with_firms()
        system.mint_cash(debtor.id, 100)
        _make_payable(system, debtor, creditor, 100, due_day=1)

        # Mock dealer subsystem with risk assessor
        mock_ra = MockRiskAssessor()

        class MockDealerSubsystem:
            risk_assessor = mock_ra

        system.state.dealer_subsystem = MockDealerSubsystem()

        settle_due(system, 1)
        assert len(mock_ra.calls) == 1
        assert mock_ra.calls[0] == (1, debtor.id, False)

    def test_settle_due_risk_assessor_updated_on_default(self):
        """Risk assessor gets updated on default (defaulted=True)."""

        class MockRiskAssessor:
            def __init__(self):
                self.calls = []
            def update_history(self, day, issuer_id, defaulted):
                self.calls.append((day, issuer_id, defaulted))

        system, _, debtor, creditor = _system_with_firms(default_mode="expel-agent")
        extra = Firm(id="E1", name="Extra", kind="firm")
        system.add_agent(extra)

        _make_payable(system, debtor, creditor, 100, due_day=1)

        mock_ra = MockRiskAssessor()

        class MockDealerSubsystem:
            risk_assessor = mock_ra

        system.state.dealer_subsystem = MockDealerSubsystem()

        settle_due(system, 1)
        assert len(mock_ra.calls) == 1
        assert mock_ra.calls[0] == (1, debtor.id, True)


# ===========================================================================
# rollover_settled_payables edge cases
# ===========================================================================


class TestRolloverSettledPayables:
    def test_defaulted_debtor_skipped(self):
        """Defaulted debtor is skipped during rollover."""
        system, _, debtor, creditor = _system_with_firms()
        debtor.defaulted = True
        system.state.defaulted_agent_ids.add(debtor.id)

        settled = [(debtor.id, creditor.id, 100, 3)]
        result = rollover_settled_payables(system, 1, settled)
        assert result == []

    def test_defaulted_creditor_skipped(self):
        """Defaulted creditor is skipped during rollover."""
        system, _, debtor, creditor = _system_with_firms()
        creditor.defaulted = True
        system.state.defaulted_agent_ids.add(creditor.id)

        settled = [(debtor.id, creditor.id, 100, 3)]
        result = rollover_settled_payables(system, 1, settled)
        assert result == []

    def test_nonexistent_debtor_skipped(self):
        """Non-existent debtor is skipped during rollover."""
        system, _, _, creditor = _system_with_firms()
        settled = [("GHOST", creditor.id, 100, 3)]
        result = rollover_settled_payables(system, 1, settled)
        assert result == []

    def test_nonexistent_creditor_skipped(self):
        """Non-existent creditor is skipped during rollover."""
        system, _, debtor, _ = _system_with_firms()
        settled = [(debtor.id, "GHOST", 100, 3)]
        result = rollover_settled_payables(system, 1, settled)
        assert result == []

    def test_partial_rollover_logs_event(self):
        """When creditor can't fully fund the rollover, logs RolloverPartial."""
        system, _, debtor, creditor = _system_with_firms()
        # Give creditor only partial cash
        system.mint_cash(creditor.id, 30)

        settled = [(debtor.id, creditor.id, 100, 3)]
        result = rollover_settled_payables(system, 1, settled, dealer_active=False)
        assert len(result) == 1

        partial_events = [e for e in system.state.events if e["kind"] == "RolloverPartial"]
        assert len(partial_events) == 1
        assert partial_events[0]["cash_transferred"] == 30
        assert partial_events[0]["amount"] == 100

    def test_model_c_no_cash_transfer_for_traders(self):
        """Model C: trader-to-trader rollover has no cash transfer."""
        system, _, debtor, creditor = _system_with_firms()
        # Give creditor cash that should NOT be transferred
        system.mint_cash(creditor.id, 200)

        cash_before = sum(
            system.state.contracts[cid].amount
            for cid in system.state.agents[creditor.id].asset_ids
            if cid in system.state.contracts and system.state.contracts[cid].kind == InstrumentKind.CASH
        )

        settled = [(debtor.id, creditor.id, 100, 3)]
        result = rollover_settled_payables(system, 1, settled, dealer_active=True)
        assert len(result) == 1

        cash_after = sum(
            system.state.contracts[cid].amount
            for cid in system.state.agents[creditor.id].asset_ids
            if cid in system.state.contracts and system.state.contracts[cid].kind == InstrumentKind.CASH
        )
        assert cash_after == cash_before  # no transfer

        rollover_events = [e for e in system.state.events if e["kind"] == "PayableRolledOver"]
        assert len(rollover_events) == 1
        assert rollover_events[0]["cash_transfer"] is False

    def test_new_due_day_based_on_max_existing(self):
        """New payable due_day = max(existing due_days) + maturity_distance."""
        system, _, debtor, creditor = _system_with_firms()
        system.mint_cash(creditor.id, 200)

        # Create existing payable with due_day=10
        _make_payable(system, debtor, creditor, 50, due_day=10)

        settled = [(debtor.id, creditor.id, 100, 3)]
        result = rollover_settled_payables(system, 1, settled, dealer_active=False)
        assert len(result) == 1

        new_payable = system.state.contracts[result[0]]
        # max_due_day should be 10 (from the existing payable), new_due_day = 10 + 3 = 13
        assert new_payable.due_day == 13

    def test_empty_settled_list(self):
        """Empty settled list returns no new payables."""
        system, _, _, _ = _system_with_firms()
        result = rollover_settled_payables(system, 1, [])
        assert result == []


# ===========================================================================
# _reconnect_ring edge cases for maturity_distance
# ===========================================================================


class TestReconnectRingMaturityFallback:
    """Cover the maturity_distance=None fallback paths in _reconnect_ring."""

    def _setup_ring(self, maturity_distance, due_day):
        """Create a 3-agent ring with specific maturity settings."""
        from bilancio.engines.settlement import _reconnect_ring
        system = System(default_mode="expel-agent")
        cb = CentralBank(id="CB", name="CB", kind="central_bank")
        system.add_agent(cb)

        agents = []
        for i in range(1, 4):
            agent = Household(id=f"H{i}", name=f"H{i}", kind="household")
            system.add_agent(agent)
            agents.append(agent)

        payables = []
        for i in range(3):
            debtor = agents[i]
            creditor = agents[(i + 1) % 3]
            p = Payable(
                id=system.new_contract_id("PAY"),
                kind=InstrumentKind.PAYABLE,
                amount=100,
                denom="X",
                asset_holder_id=creditor.id,
                liability_issuer_id=debtor.id,
                due_day=due_day,
                maturity_distance=maturity_distance,
            )
            system.add_contract(p)
            payables.append(p)

        return system, agents, payables

    def test_maturity_distance_none_with_due_day(self):
        """When maturity_distance is None but due_day is set, fallback to due_day - day."""
        from bilancio.engines.settlement import _reconnect_ring

        system, agents, payables = self._setup_ring(maturity_distance=None, due_day=7)
        system.state.day = 3

        # H2 defaults: H2->H3 is payables[1]
        # Simulate default
        successor_id = payables[1].asset_holder_id
        _remove_contract(system, payables[1].id)
        _expel_agent(system, "H2", trigger_contract_id=payables[1].id, trigger_kind=payables[1].kind)

        result = _reconnect_ring(system, "H2", successor_id, day=3)
        assert result is not None
        # Predecessor H1->H2 payable has due_day=7, maturity_distance=None
        # Fallback: max(1, 7 - 3) = 4
        assert result["maturity_distance"] == 4
        assert result["new_due_day"] == 7  # 3 + 4

    def test_maturity_distance_none_and_due_day_none(self):
        """When both maturity_distance and due_day are None, fallback to 1."""
        from bilancio.engines.settlement import _reconnect_ring

        system, agents, payables = self._setup_ring(maturity_distance=None, due_day=0)
        system.state.day = 3

        # Set predecessor payable's due_day to None after creation
        payables[0].due_day = None

        successor_id = payables[1].asset_holder_id
        _remove_contract(system, payables[1].id)
        _expel_agent(system, "H2", trigger_contract_id=payables[1].id, trigger_kind=payables[1].kind)

        result = _reconnect_ring(system, "H2", successor_id, day=3)
        assert result is not None
        assert result["maturity_distance"] == 1
        assert result["new_due_day"] == 4  # 3 + 1

    def test_no_predecessor_found(self):
        """When there's no predecessor payable, reconnection returns None."""
        from bilancio.engines.settlement import _reconnect_ring

        system = System(default_mode="expel-agent")
        cb = CentralBank(id="CB", name="CB", kind="central_bank")
        system.add_agent(cb)
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        system.add_agent(h1)
        system.add_agent(h2)

        # No predecessor payable exists at all
        result = _reconnect_ring(system, "H1", "H2", day=3)
        assert result is None


# ===========================================================================
# _expel_agent: aliased contracts getting tracked
# ===========================================================================


class TestExpelAgentAliasTracking:
    def test_aliased_contract_tracked_in_cancelled_aliases(self):
        """When a liability has an alias, it's added to cancelled_aliases and removed."""
        system, _, debtor, creditor = _system_with_firms(default_mode="expel-agent")
        extra = Firm(id="E1", name="Extra", kind="firm")
        system.add_agent(extra)

        p = _make_payable(system, debtor, creditor, 100, due_day=2)
        system.state.aliases["PAYALIAS"] = p.id

        # Schedule action referencing this alias
        system.state.scheduled_actions_by_day[5] = [
            {"transfer_claim": {"contract_alias": "PAYALIAS", "to_agent": "E1"}},
        ]

        _expel_agent(system, debtor.id, trigger_contract_id="TRIGGER")

        # The alias should have been removed from system aliases
        assert "PAYALIAS" not in system.state.aliases
        # The scheduled action should have been cancelled
        assert 5 not in system.state.scheduled_actions_by_day


# ===========================================================================
# _remove_contract: due_day index ValueError branch
# ===========================================================================


class TestRemoveContractDueDayEdge:
    def test_remove_contract_not_in_due_day_bucket(self):
        """Contract ID not in due_day bucket but due_day entry exists -> ValueError handled."""
        system, _, debtor, creditor = _system_with_firms()
        p = _make_payable(system, debtor, creditor, 100, due_day=5)
        # Manually remove from bucket but keep the bucket
        bucket = system.state.contracts_by_due_day[5]
        bucket.remove(p.id)
        bucket.append("OTHER_ID")  # keep bucket non-empty
        _remove_contract(system, p.id)
        assert p.id not in system.state.contracts

    def test_remove_contract_due_day_bucket_becomes_empty(self):
        """When removing the last item from a due_day bucket, the bucket itself is removed."""
        system, _, debtor, creditor = _system_with_firms()
        p = _make_payable(system, debtor, creditor, 100, due_day=5)
        _remove_contract(system, p.id)
        assert 5 not in system.state.contracts_by_due_day


# ===========================================================================
# Unknown payment method (line 646)
# ===========================================================================


class TestUnknownPaymentMethod:
    def test_unknown_settlement_method_raises(self):
        """An unknown payment method in settlement_order raises ValidationError."""
        # Create a custom policy that returns an unknown method
        custom_policy = PolicyEngine.default()
        custom_policy.mop_rank["firm"] = ["unknown_method"]

        system = System(policy=custom_policy)
        cb = CentralBank(id="CB", name="CB", kind="central_bank")
        debtor = Firm(id="D1", name="Debtor", kind="firm")
        creditor = Firm(id="C1", name="Creditor", kind="firm")
        system.add_agent(cb)
        system.add_agent(debtor)
        system.add_agent(creditor)

        system.mint_cash(debtor.id, 100)
        _make_payable(system, debtor, creditor, 100, due_day=1)

        with pytest.raises(ValidationError, match="unknown payment method"):
            settle_due(system, 1)
