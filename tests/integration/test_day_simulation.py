from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.simulation import run_day
from bilancio.engines.system import System
from bilancio.ops.banking import client_payment, deposit_cash


def test_run_day_basic():
    """Test basic day simulation with all phases working together."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    b2 = Bank(id="B2", name="Bank 2", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    h3 = Household(id="H3", name="Household 3", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(b2)
    sys.add_agent(h1)
    sys.add_agent(h2)
    sys.add_agent(h3)

    # Setup: Banks have reserves
    sys.mint_reserves("B1", 500)
    sys.mint_reserves("B2", 500)

    # Setup: Households have deposits
    sys.mint_cash("H1", 300)
    sys.mint_cash("H2", 200)
    sys.mint_cash("H3", 100)
    deposit_cash(sys, "H1", "B1", 300)  # H1@B1
    deposit_cash(sys, "H2", "B1", 200)  # H2@B1
    deposit_cash(sys, "H3", "B2", 100)  # H3@B2

    # Create payable due today: H1 owes H2 amount 150
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

    # Create some cross-bank payments during the day
    client_payment(sys, "H2", "B1", "H3", "B2", 80)  # B1 owes B2: 80

    initial_day = sys.state.day

    # Run the day simulation
    run_day(sys)

    # Check day incremented
    assert sys.state.day == initial_day + 1

    # Phase A: Check PhaseA event logged
    phase_a_events = [e for e in sys.state.events if e["kind"] == "PhaseA"]
    assert len(phase_a_events) == 1
    assert phase_a_events[0]["day"] == initial_day

    # Phase B: Check payable was settled
    assert payable_id not in sys.state.contracts

    # Check PayableSettled event
    settled_events = [e for e in sys.state.events if e["kind"] == "PayableSettled"]
    assert len(settled_events) == 1
    assert settled_events[0]["debtor"] == "H1"
    assert settled_events[0]["creditor"] == "H2"
    assert settled_events[0]["amount"] == 150

    # Check H1 and H2 balances after settlement (same bank)
    assert sys.total_deposit("H1", "B1") == 150  # 300 - 150
    assert sys.total_deposit("H2", "B1") == 270  # 200 + 150 - 80 (paid to H3)
    assert sys.total_deposit("H3", "B2") == 180  # 100 + 80

    # Phase C: Check reserves transferred for cross-bank net
    # B1 owed B2: 80, should be settled with reserves
    b1_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["B1"].asset_ids
        if cid in sys.state.contracts
        and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    b2_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["B2"].asset_ids
        if cid in sys.state.contracts
        and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    assert b1_reserves == 420  # 500 - 80
    assert b2_reserves == 580  # 500 + 80

    # Check InterbankCleared event
    cleared_events = [e for e in sys.state.events if e["kind"] == "InterbankCleared"]
    assert len(cleared_events) == 1
    assert cleared_events[0]["debtor_bank"] == "B1"
    assert cleared_events[0]["creditor_bank"] == "B2"
    assert cleared_events[0]["amount"] == 80

    sys.assert_invariants()


def test_overnight_settlement_next_day():
    """Test overnight payables from day t are settled on day t+1."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    b2 = Bank(id="B2", name="Bank 2", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h3 = Household(id="H3", name="Household 3", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(b2)
    sys.add_agent(h1)
    sys.add_agent(h3)

    # Setup: B1 has insufficient reserves, B2 has plenty
    sys.mint_reserves("B1", 30)
    sys.mint_reserves("B2", 500)

    # Setup: Customers have deposits
    sys.mint_cash("H1", 200)
    sys.mint_cash("H3", 100)
    deposit_cash(sys, "H1", "B1", 200)  # H1@B1
    deposit_cash(sys, "H3", "B2", 100)  # H3@B2

    # Create cross-bank payment that will require overnight payable
    client_payment(sys, "H1", "B1", "H3", "B2", 100)  # B1 owes B2: 100

    day_0 = sys.state.day

    # Run day 0 - should create overnight payable
    run_day(sys)

    # Check overnight payable created
    payables = [c for c in sys.state.contracts.values() if c.kind == InstrumentKind.PAYABLE]
    assert len(payables) == 1
    overnight_payable = payables[0]
    assert overnight_payable.liability_issuer_id == "B1"
    assert overnight_payable.asset_holder_id == "B2"
    assert overnight_payable.amount == 100
    assert overnight_payable.due_day == day_0 + 1  # due tomorrow

    # Check reserves unchanged on day 0 (insufficient)
    b1_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["B1"].asset_ids
        if cid in sys.state.contracts
        and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    assert b1_reserves == 30  # unchanged

    # Now add more reserves to B1 for next day settlement
    sys.mint_reserves("B1", 200)  # B1 now has 230 total

    # Run day 1 - overnight payable should be settled
    run_day(sys)

    # Check overnight payable settled and removed
    payables_after = [c for c in sys.state.contracts.values() if c.kind == InstrumentKind.PAYABLE]
    assert len(payables_after) == 0

    # Check reserves transferred on day 1
    b1_reserves_final = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["B1"].asset_ids
        if cid in sys.state.contracts
        and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    b2_reserves_final = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["B2"].asset_ids
        if cid in sys.state.contracts
        and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    assert b1_reserves_final == 130  # 230 - 100
    assert b2_reserves_final == 600  # 500 + 100

    # Check PayableSettled event on day 1
    settled_events = [
        e for e in sys.state.events if e["kind"] == "PayableSettled" and e["day"] == day_0 + 1
    ]
    assert len(settled_events) == 1
    assert settled_events[0]["debtor"] == "B1"
    assert settled_events[0]["creditor"] == "B2"
    assert settled_events[0]["amount"] == 100

    sys.assert_invariants()


def test_policy_order_drives_payment_choice():
    """Test that policy order determines payment method preference."""
    # Create custom policy that prioritizes cash over deposits for households
    custom_policy = PolicyEngine.default()
    custom_policy.mop_rank["household"] = ["cash", "bank_deposit"]  # cash first!

    sys = System(policy=custom_policy)
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)
    sys.add_agent(h2)

    # Setup: H1 has both cash and deposit
    sys.mint_cash("H1", 200)
    deposit_cash(sys, "H1", "B1", 100)  # H1 still has 100 cash + 100 deposit

    # Create payable: H1 owes H2 amount 80, due today
    payable_id = sys.new_contract_id("P")
    payable = Payable(
        id=payable_id,
        kind=InstrumentKind.PAYABLE,
        amount=80,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="H1",
        due_day=sys.state.day,
    )
    sys.add_contract(payable)

    # Run the day
    run_day(sys)

    # Check payable settled
    assert payable_id not in sys.state.contracts

    # With custom policy (cash first), H1 should have used cash, not deposit
    # H1 deposit should be unchanged
    assert sys.total_deposit("H1", "B1") == 100  # unchanged

    # H1 cash should be reduced by 80
    h1_cash = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["H1"].asset_ids
        if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    assert h1_cash == 20  # 100 - 80

    # H2 should have received 80 cash (direct transfer)
    h2_cash = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["H2"].asset_ids
        if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    assert h2_cash == 80

    # H2 should have no deposits (received cash directly)
    assert sys.total_deposit("H2", "B1") == 0

    sys.assert_invariants()


def test_policy_order_drives_payment_choice_default():
    """Test default policy (deposits first) for comparison."""
    sys = System()  # default policy: deposits first
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)
    sys.add_agent(h2)

    # Setup: H1 has both cash and deposit (identical to above test)
    sys.mint_cash("H1", 200)
    deposit_cash(sys, "H1", "B1", 100)  # H1 still has 100 cash + 100 deposit

    # Create payable: H1 owes H2 amount 80, due today
    payable_id = sys.new_contract_id("P")
    payable = Payable(
        id=payable_id,
        kind=InstrumentKind.PAYABLE,
        amount=80,
        denom="X",
        asset_holder_id="H2",
        liability_issuer_id="H1",
        due_day=sys.state.day,
    )
    sys.add_contract(payable)

    # Run the day
    run_day(sys)

    # Check payable settled
    assert payable_id not in sys.state.contracts

    # With default policy (deposits first), H1 should have used deposit, not cash
    # H1 deposit should be reduced by 80
    assert sys.total_deposit("H1", "B1") == 20  # 100 - 80

    # H1 cash should be unchanged
    h1_cash = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["H1"].asset_ids
        if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    assert h1_cash == 100  # unchanged

    # H2 should have received 80 as deposit (same bank transfer)
    assert sys.total_deposit("H2", "B1") == 80

    # H2 should have no cash (received deposit)
    h2_cash = sum(
        sys.state.contracts[cid].amount
        for cid in sys.state.agents["H2"].asset_ids
        if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    assert h2_cash == 0

    sys.assert_invariants()
