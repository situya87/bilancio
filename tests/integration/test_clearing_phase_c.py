import pytest
from bilancio.engines.system import System
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.engines.clearing import settle_intraday_nets, compute_intraday_nets
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.household import Household
from bilancio.ops.banking import deposit_cash, client_payment


def test_phase_c_netting_reserves():
    """Test Phase C clearing nets multiple cross-bank payments correctly using reserves."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    b2 = Bank(id="B2", name="Bank 2", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    h3 = Household(id="H3", name="Household 3", kind="household")
    h4 = Household(id="H4", name="Household 4", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(b2)
    sys.add_agent(h1)
    sys.add_agent(h2)
    sys.add_agent(h3)
    sys.add_agent(h4)
    
    # Setup: Both banks have reserves
    sys.mint_reserves("B1", 1000)
    sys.mint_reserves("B2", 1000)
    
    # Setup: Customers have deposits
    sys.mint_cash("H1", 200)
    sys.mint_cash("H2", 150)
    sys.mint_cash("H3", 300)
    sys.mint_cash("H4", 100)
    deposit_cash(sys, "H1", "B1", 200)  # H1@B1
    deposit_cash(sys, "H2", "B1", 150)  # H2@B1
    deposit_cash(sys, "H3", "B2", 300)  # H3@B2
    deposit_cash(sys, "H4", "B2", 100)  # H4@B2
    
    # Create cross-bank payments that will net out partially:
    # H1@B1 pays H3@B2: 80 (B1 owes B2: 80)
    client_payment(sys, "H1", "B1", "H3", "B2", 80)
    # H2@B1 pays H4@B2: 50 (B1 owes B2: 50 more, total 130)
    client_payment(sys, "H2", "B1", "H4", "B2", 50)
    # H3@B2 pays H1@B1: 60 (B2 owes B1: 60, nets against 130, so B1 still owes B2: 70)
    client_payment(sys, "H3", "B2", "H1", "B1", 60)
    
    current_day = sys.state.day
    
    # Test compute_intraday_nets
    nets = compute_intraday_nets(sys, current_day)
    
    # With lexical ordering: B1 < B2, so nets should be positive (B1 owes B2)
    assert len(nets) == 1
    assert nets[("B1", "B2")] == 70  # 80 + 50 - 60 = 70
    
    # Phase C: Settle intraday nets
    settle_intraday_nets(sys, current_day)
    
    # Check B1 reserves reduced by 70
    b1_reserves = sum(sys.state.contracts[cid].amount for cid in sys.state.agents["B1"].asset_ids 
                      if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT)
    assert b1_reserves == 930  # 1000 - 70
    
    # Check B2 reserves increased by 70
    b2_reserves = sum(sys.state.contracts[cid].amount for cid in sys.state.agents["B2"].asset_ids 
                      if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT)
    assert b2_reserves == 1070  # 1000 + 70
    
    # Check InterbankCleared event logged
    events = [e for e in sys.state.events if e["kind"] == "InterbankCleared"]
    assert len(events) == 1
    assert events[0]["debtor_bank"] == "B1"
    assert events[0]["creditor_bank"] == "B2"
    assert events[0]["amount"] == 70
    
    # No overnight payables created
    overnight_events = [e for e in sys.state.events if e["kind"] == "InterbankOvernightCreated"]
    assert len(overnight_events) == 0
    
    sys.assert_invariants()


def test_phase_c_overnight_creation():
    """Test Phase C clearing creates overnight payable when insufficient reserves."""
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
    
    # Setup: B1 has insufficient reserves (only 30), B2 has plenty
    sys.mint_reserves("B1", 30)
    sys.mint_reserves("B2", 1000)
    
    # Setup: Customers have deposits
    sys.mint_cash("H1", 200)
    sys.mint_cash("H3", 100)
    deposit_cash(sys, "H1", "B1", 200)  # H1@B1
    deposit_cash(sys, "H3", "B2", 100)  # H3@B2
    
    # Create cross-bank payment: H1@B1 pays H3@B2: 100 (needs 100 in reserves)
    client_payment(sys, "H1", "B1", "H3", "B2", 100)
    
    current_day = sys.state.day
    
    # Phase C: Settle intraday nets (should create overnight payable)
    settle_intraday_nets(sys, current_day)
    
    # Check B1 reserves unchanged (insufficient to cover 100)
    b1_reserves = sum(sys.state.contracts[cid].amount for cid in sys.state.agents["B1"].asset_ids 
                      if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT)
    assert b1_reserves == 30  # unchanged
    
    # Check B2 reserves unchanged
    b2_reserves = sum(sys.state.contracts[cid].amount for cid in sys.state.agents["B2"].asset_ids 
                      if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT)
    assert b2_reserves == 1000  # unchanged
    
    # Check overnight payable created: B1 owes B2, due tomorrow
    payables = [c for c in sys.state.contracts.values() if c.kind == InstrumentKind.PAYABLE]
    assert len(payables) == 1
    overnight_payable = payables[0]
    assert overnight_payable.liability_issuer_id == "B1"
    assert overnight_payable.asset_holder_id == "B2"
    assert overnight_payable.amount == 100
    assert overnight_payable.due_day == current_day + 1
    
    # Check InterbankOvernightCreated event logged
    overnight_events = [e for e in sys.state.events if e["kind"] == "InterbankOvernightCreated"]
    assert len(overnight_events) == 1
    assert overnight_events[0]["debtor_bank"] == "B1"
    assert overnight_events[0]["creditor_bank"] == "B2"
    assert overnight_events[0]["amount"] == 100
    assert overnight_events[0]["due_day"] == current_day + 1
    
    # No InterbankCleared event
    cleared_events = [e for e in sys.state.events if e["kind"] == "InterbankCleared"]
    assert len(cleared_events) == 0
    
    sys.assert_invariants()


def test_phase_c_no_nets_for_same_bank():
    """Test Phase C clearing does not create nets for same-bank payments."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    h3 = Household(id="H3", name="Household 3", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)
    sys.add_agent(h2)
    sys.add_agent(h3)
    
    # Setup: All customers at same bank
    sys.mint_cash("H1", 150)
    sys.mint_cash("H2", 200)
    sys.mint_cash("H3", 100)
    deposit_cash(sys, "H1", "B1", 150)  # H1@B1
    deposit_cash(sys, "H2", "B1", 200)  # H2@B1
    deposit_cash(sys, "H3", "B1", 100)  # H3@B1
    
    # Create same-bank payments (should not generate ClientPayment events)
    client_payment(sys, "H1", "B1", "H2", "B1", 50)  # H1 pays H2
    client_payment(sys, "H2", "B1", "H3", "B1", 80)  # H2 pays H3
    client_payment(sys, "H3", "B1", "H1", "B1", 30)  # H3 pays H1
    
    current_day = sys.state.day
    
    # Check no ClientPayment events (all same bank)
    client_payment_events = [e for e in sys.state.events if e["kind"] == "ClientPayment"]
    assert len(client_payment_events) == 0
    
    # Test compute_intraday_nets - should be empty
    nets = compute_intraday_nets(sys, current_day)
    assert len(nets) == 0
    
    # Phase C: Settle intraday nets (should be no-op)
    settle_intraday_nets(sys, current_day)
    
    # Check no interbank events
    interbank_events = [e for e in sys.state.events 
                       if e["kind"] in ["InterbankCleared", "InterbankOvernightCreated"]]
    assert len(interbank_events) == 0
    
    # Check no payables created
    payables = [c for c in sys.state.contracts.values() if c.kind == InstrumentKind.PAYABLE]
    assert len(payables) == 0
    
    # Check deposit balances (from same-bank transfers)
    assert sys.total_deposit("H1", "B1") == 130  # 150 - 50 + 30
    assert sys.total_deposit("H2", "B1") == 170  # 200 + 50 - 80
    assert sys.total_deposit("H3", "B1") == 150  # 100 + 80 - 30
    
    sys.assert_invariants()