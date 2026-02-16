import pytest

from bilancio.core.errors import DefaultError
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.settlement import settle_due
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash


def test_phase_b_same_bank_settlement():
    """Test Phase B settlement when debtor and creditor use the same bank."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)
    sys.add_agent(h2)

    # Setup: H1 has 200 deposit at B1
    sys.mint_cash("H1", 200)
    deposit_cash(sys, "H1", "B1", 200)

    # Create payable: H1 owes H2 amount 150, due today
    payable_id = sys.new_contract_id("P")
    payable = Payable(
        id=payable_id,
        kind=InstrumentKind.PAYABLE,
        amount=150,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="H1",
        due_day=sys.state.day,
    )
    sys.add_contract(payable)

    # Phase B: Settle due payables
    settle_due(sys, sys.state.day)

    # Check H1 deposit reduced by 150
    assert sys.total_deposit("H1", "B1") == 50

    # Check H2 deposit increased by 150
    assert sys.total_deposit("H2", "B1") == 150

    # Check payable was removed
    assert payable_id not in sys.state.contracts

    # Check PayableSettled event logged
    events = [e for e in sys.state.events if e["kind"] == "PayableSettled"]
    assert len(events) == 1
    assert events[0]["debtor"] == "H1"
    assert events[0]["creditor"] == "H2"
    assert events[0]["amount"] == 150

    # No ClientPayment event (same bank)
    client_payment_events = [e for e in sys.state.events if e["kind"] == "ClientPayment"]
    assert len(client_payment_events) == 0

    sys.assert_invariants()


def test_phase_b_cross_bank_settlement_logs_client_payment():
    """Test Phase B settlement logs ClientPayment when debtor and creditor use different banks."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    b2 = Bank(id="B2", name="Bank 2", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(b2)
    sys.add_agent(h1)
    sys.add_agent(h2)

    # Setup: H1 has 300 deposit at B1, H2 has 1 deposit at B2 to establish bank relationship
    sys.mint_cash("H1", 300)
    sys.mint_cash("H2", 1)
    deposit_cash(sys, "H1", "B1", 300)
    deposit_cash(sys, "H2", "B2", 1)

    # Create payable: H1 owes H2 amount 100, due today
    payable_id = sys.new_contract_id("P")
    payable = Payable(
        id=payable_id,
        kind=InstrumentKind.PAYABLE,
        amount=100,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="H1",
        due_day=sys.state.day,
    )
    sys.add_contract(payable)

    # Phase B: Settle due payables
    settle_due(sys, sys.state.day)

    # Check H1 deposit reduced by 100
    assert sys.total_deposit("H1", "B1") == 200

    # Check H2 deposit increased by 100 at B2 (1 initial + 100 from payment)
    assert sys.total_deposit("H2", "B2") == 101

    # Check payable was removed
    assert payable_id not in sys.state.contracts

    # Check ClientPayment event logged (cross-bank payment)
    client_payment_events = [e for e in sys.state.events if e["kind"] == "ClientPayment"]
    assert len(client_payment_events) == 1
    assert client_payment_events[0]["payer"] == "H1"
    assert client_payment_events[0]["payer_bank"] == "B1"
    assert client_payment_events[0]["payee"] == "H2"
    assert client_payment_events[0]["payee_bank"] == "B2"
    assert client_payment_events[0]["amount"] == 100

    sys.assert_invariants()


def test_phase_b_cash_fallback():
    """Test Phase B settlement uses cash when deposits insufficient but policy allows."""
    # Create custom policy that allows cash fallback for households
    custom_policy = PolicyEngine.default()
    custom_policy.mop_rank["household"] = ["bank_deposit", "cash"]

    sys = System(policy=custom_policy)
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)
    sys.add_agent(h2)

    # Setup: H1 has 60 deposit at B1 and 80 cash
    sys.mint_cash("H1", 140)
    deposit_cash(sys, "H1", "B1", 60)

    # Create payable: H1 owes H2 amount 100, due today (more than deposit)
    payable_id = sys.new_contract_id("P")
    payable = Payable(
        id=payable_id,
        kind=InstrumentKind.PAYABLE,
        amount=100,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="H1",
        due_day=sys.state.day,
    )
    sys.add_contract(payable)

    # Phase B: Settle due payables
    settle_due(sys, sys.state.day)

    # Check H1 deposit is now 0 (used all 60)
    assert sys.total_deposit("H1", "B1") == 0

    # Check H1 cash reduced by 40 (100 - 60 from deposit)
    h1_cash = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["H1"].asset_ids
        if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    assert h1_cash == 40  # 80 - 40 used for payment

    # Check H2 has 60 deposit at B1 and 40 cash
    assert sys.total_deposit("H2", "B1") == 60  # from H1's deposit portion
    h2_cash = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["H2"].asset_ids
        if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    assert h2_cash == 40  # from H1's cash portion

    # Check payable was removed
    assert payable_id not in sys.state.contracts

    sys.assert_invariants()


def test_phase_b_default_on_insufficient_means():
    """Test Phase B settlement raises DefaultError when no sufficient funds available."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)
    sys.add_agent(h2)

    # Setup: H1 has only 30 deposit at B1 and 20 cash = 50 total
    sys.mint_cash("H1", 50)
    deposit_cash(sys, "H1", "B1", 30)

    # Create payable: H1 owes H2 amount 100, due today (more than available)
    payable_id = sys.new_contract_id("P")
    payable = Payable(
        id=payable_id,
        kind=InstrumentKind.PAYABLE,
        amount=100,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="H1",
        due_day=sys.state.day,
    )
    sys.add_contract(payable)

    # Phase B: Settle due payables should fail
    with pytest.raises(DefaultError, match="Insufficient funds to settle payable"):
        settle_due(sys, sys.state.day)

    # Check payable still exists (settlement failed)
    assert payable_id in sys.state.contracts
    assert sys.state.contracts[payable_id].amount == 100

    # Check balances unchanged
    assert sys.total_deposit("H1", "B1") == 30
    h1_cash = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["H1"].asset_ids
        if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    assert h1_cash == 20

    sys.assert_invariants()


def test_phase_b_bank_to_bank_reserves():
    """Test Phase B settlement between banks using reserves."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    b2 = Bank(id="B2", name="Bank 2", kind="bank")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(b2)

    # Setup: B1 has 500 reserves
    sys.mint_reserves("B1", 500)

    # Create payable: B1 owes B2 amount 200, due today
    payable_id = sys.new_contract_id("P")
    payable = Payable(
        id=payable_id,
        kind=InstrumentKind.PAYABLE,
        amount=200,
        denom="X",
        asset_holder_id="B2",
        liability_issuer_id="B1",
        due_day=sys.state.day,
    )
    sys.add_contract(payable)

    # Phase B: Settle due payables
    settle_due(sys, sys.state.day)

    # Check B1 reserves reduced by 200
    b1_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["B1"].asset_ids
        if cid in sys.state.contracts
        and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    assert b1_reserves == 300

    # Check B2 reserves increased by 200
    b2_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["B2"].asset_ids
        if cid in sys.state.contracts
        and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    assert b2_reserves == 200

    # Check payable was removed
    assert payable_id not in sys.state.contracts

    # Check PayableSettled event logged
    events = [e for e in sys.state.events if e["kind"] == "PayableSettled"]
    assert len(events) == 1
    assert events[0]["debtor"] == "B1"
    assert events[0]["creditor"] == "B2"
    assert events[0]["amount"] == 200

    sys.assert_invariants()
