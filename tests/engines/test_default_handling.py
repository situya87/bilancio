import pytest

from bilancio.core.errors import DefaultError
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.settlement import settle_due
from bilancio.engines.system import System


def _basic_system(default_mode: str = "fail-fast"):
    system = System(default_mode=default_mode)
    cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
    debtor = Firm(id="D1", name="Debtor", kind="firm")
    creditor = Firm(id="C1", name="Creditor", kind="firm")
    system.add_agent(cb)
    system.add_agent(debtor)
    system.add_agent(creditor)
    return system, cb, debtor, creditor


def _make_payable(
    system: System, debtor: Firm, creditor: Firm, amount: int, due_day: int
) -> Payable:
    payable = Payable(
        id=system.new_contract_id("PAY"),
        kind=InstrumentKind.PAYABLE,
        amount=amount,
        denom="X",
        asset_holder_id=creditor.id,
        liability_issuer_id=debtor.id,
        due_day=due_day,
    )
    system.add_contract(payable)
    return payable


def test_fail_fast_mode_raises_default():
    system, _, debtor, creditor = _basic_system()
    _make_payable(system, debtor, creditor, amount=100, due_day=1)

    with pytest.raises(DefaultError):
        settle_due(system, 1)


def test_expel_mode_handles_partial_payment_and_marks_agent():
    system, _, debtor, creditor = _basic_system(default_mode="expel-agent")
    payable = _make_payable(system, debtor, creditor, amount=100, due_day=1)
    trailing_payable = _make_payable(system, debtor, creditor, amount=50, due_day=2)

    system.state.aliases["PAY1"] = payable.id

    # Provide partial liquidity (60 of required 100)
    system.mint_cash(debtor.id, 60)

    # Scheduled action involving debtor should be cancelled once defaulted
    system.state.scheduled_actions_by_day[2] = [{"mint_cash": {"to": debtor.id, "amount": 10}}]
    system.state.scheduled_actions_by_day[3] = [
        {"transfer_claim": {"contract_alias": "PAY1", "to_agent": creditor.id}}
    ]

    settle_due(system, 1)

    assert payable.id not in system.state.contracts
    assert trailing_payable.id not in system.state.contracts
    assert debtor.defaulted is True
    assert debtor.id in system.state.defaulted_agent_ids
    assert payable.id not in creditor.asset_ids
    assert trailing_payable.id not in creditor.asset_ids
    assert not system.state.scheduled_actions_by_day
    assert "PAY1" not in system.state.aliases

    kinds = [event["kind"] for event in system.state.events]
    assert "PartialSettlement" in kinds
    assert "ObligationDefaulted" in kinds
    assert "AgentDefaulted" in kinds
    assert "ScheduledActionCancelled" in kinds
    assert "ObligationWrittenOff" in kinds

    cancelled_events = [e for e in system.state.events if e["kind"] == "ScheduledActionCancelled"]
    assert len(cancelled_events) == 2
    for evt in cancelled_events:
        assert evt["day"] == system.state.day
    scheduled_days = sorted(evt["scheduled_day"] for evt in cancelled_events)
    assert scheduled_days == [2, 3]

    partial_event = next(e for e in system.state.events if e["kind"] == "PartialSettlement")
    assert partial_event["amount_paid"] == 60
    assert partial_event["shortfall"] == 40
    assert partial_event["contract_id"] == payable.id
    assert partial_event["distribution"] == [{"method": "cash", "amount": 60}]

    default_event = next(e for e in system.state.events if e["kind"] == "ObligationDefaulted")
    assert default_event["contract_id"] == payable.id
    assert default_event["shortfall"] == 40
    assert default_event["amount"] == 40

    written_off_ids = [
        e["contract_id"] for e in system.state.events if e["kind"] == "ObligationWrittenOff"
    ]
    assert trailing_payable.id in written_off_ids

    agent_event = next(e for e in system.state.events if e["kind"] == "AgentDefaulted")
    assert agent_event["frm"] == debtor.id
