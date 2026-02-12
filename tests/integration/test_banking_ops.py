import pytest
from bilancio.engines.system import System
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.core.errors import ValidationError
from bilancio.ops.banking import deposit_cash, withdraw_cash, client_payment


def test_deposit_cash():
    """Test depositing cash from customer to bank creates bank deposit."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)
    
    # CB mints cash to H1
    sys.mint_cash("H1", 120)
    
    # H1 deposits 120 to B1
    dep_id = deposit_cash(sys, "H1", "B1", 120)
    
    # Check cash moved H1→B1
    h1_cash = sum(sys.state.contracts[cid].amount for cid in h1.asset_ids 
                  if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH)
    b1_cash = sum(sys.state.contracts[cid].amount for cid in b1.asset_ids 
                  if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH)
    assert h1_cash == 0
    assert b1_cash == 120
    
    # Check H1 has deposit 120 at B1
    assert sys.total_deposit("H1", "B1") == 120
    
    # Check event logged
    events = [e for e in sys.state.events if e["kind"] == "CashDeposited"]
    assert len(events) == 1
    assert events[0]["customer"] == "H1"
    assert events[0]["bank"] == "B1"
    assert events[0]["amount"] == 120
    
    # Invariants hold
    sys.assert_invariants()


def test_withdraw_cash():
    """Test withdrawing cash from bank deposit."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)
    
    # Setup: H1 has 120 deposit at B1
    sys.mint_cash("H1", 120)
    deposit_cash(sys, "H1", "B1", 120)
    
    # Withdraw 70
    withdraw_cash(sys, "H1", "B1", 70)
    
    # Check H1 deposit now 50
    assert sys.total_deposit("H1", "B1") == 50
    
    # Check H1 cash +70
    h1_cash = sum(sys.state.contracts[cid].amount for cid in h1.asset_ids 
                  if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH)
    assert h1_cash == 70
    
    # Check bank cash -70
    b1_cash = sum(sys.state.contracts[cid].amount for cid in b1.asset_ids 
                  if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH)
    assert b1_cash == 50
    
    # Check event logged
    events = [e for e in sys.state.events if e["kind"] == "CashWithdrawn"]
    assert len(events) == 1
    assert events[0]["customer"] == "H1"
    assert events[0]["bank"] == "B1"
    assert events[0]["amount"] == 70
    
    sys.assert_invariants()


def test_withdraw_insufficient_bank_cash():
    """Test error when bank doesn't have enough cash on hand."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)
    
    # Setup: H1 deposits 100, but B1 only has 50 cash
    sys.mint_cash("H1", 100)
    deposit_cash(sys, "H1", "B1", 100)
    
    # Bank transfers 50 cash away (simulating it being used elsewhere)
    sys.mint_cash("B1", 0)  # B1 now has 100 cash from deposit
    # Transfer 60 away to simulate shortage
    b2 = Bank(id="B2", name="Bank 2", kind="bank")
    sys.add_agent(b2)
    sys.transfer_cash("B1", "B2", 60)
    
    # Try to withdraw 50 (should fail since B1 only has 40 cash)
    with pytest.raises(ValidationError, match="insufficient cash on hand"):
        withdraw_cash(sys, "H1", "B1", 50)
    
    # Deposit should remain unchanged
    assert sys.total_deposit("H1", "B1") == 100
    
    sys.assert_invariants()


def test_same_bank_payment():
    """Test payment between customers at same bank."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)
    sys.add_agent(h2)
    
    # Setup: H1 has 100 deposit at B1
    sys.mint_cash("H1", 100)
    deposit_cash(sys, "H1", "B1", 100)
    
    # H1@B1 pays H2@B1, 40
    client_payment(sys, "H1", "B1", "H2", "B1", 40)
    
    # Check H1 deposit -40
    assert sys.total_deposit("H1", "B1") == 60
    
    # Check H2 deposit +40
    assert sys.total_deposit("H2", "B1") == 40
    
    # No ClientPayment event (same bank)
    events = [e for e in sys.state.events if e["kind"] == "ClientPayment"]
    assert len(events) == 0
    
    sys.assert_invariants()


def test_cross_bank_payment():
    """Test payment between customers at different banks."""
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
    
    # Setup: H1 has 100 deposit at B1
    sys.mint_cash("H1", 100)
    deposit_cash(sys, "H1", "B1", 100)
    
    # H1@B1 pays H3@B2, 30
    client_payment(sys, "H1", "B1", "H3", "B2", 30)
    
    # Check H1 deposit -30
    assert sys.total_deposit("H1", "B1") == 70
    
    # Check H3 deposit +30 at B2
    assert sys.total_deposit("H3", "B2") == 30
    
    # ClientPayment event recorded
    events = [e for e in sys.state.events if e["kind"] == "ClientPayment"]
    assert len(events) == 1
    assert events[0]["payer"] == "H1"
    assert events[0]["payer_bank"] == "B1"
    assert events[0]["payee"] == "H3"
    assert events[0]["payee_bank"] == "B2"
    assert events[0]["amount"] == 30
    
    # No reserves moved (deferred to clearing)
    # Banks don't have reserve deposits yet
    
    sys.assert_invariants()


def test_insufficient_deposit_payment():
    """Test payment fails with insufficient deposit, succeeds with cash fallback."""
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
    
    # Setup: H1 has 30 deposit at B1 and 50 cash
    sys.mint_cash("H1", 80)
    deposit_cash(sys, "H1", "B1", 30)
    
    # Try to pay 60 (more than deposit) - should fail
    with pytest.raises(ValidationError, match="insufficient funds"):
        client_payment(sys, "H1", "B1", "H2", "B2", 60)
    
    # Balances unchanged
    assert sys.total_deposit("H1", "B1") == 30
    h1_from_sys = sys.state.agents["H1"]
    h1_cash = sum(sys.state.contracts[cid].amount for cid in h1_from_sys.asset_ids 
                  if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH)
    assert h1_cash == 50
    
    # Re-run with allow_cash_fallback=True
    client_payment(sys, "H1", "B1", "H2", "B2", 60, allow_cash_fallback=True)
    
    # H1 deposit should be 0 (used 30)
    assert sys.total_deposit("H1", "B1") == 0
    
    # H1 cash should be 20 (used 30 from 50)
    h1_from_sys = sys.state.agents["H1"]
    h1_cash = sum(sys.state.contracts[cid].amount for cid in h1_from_sys.asset_ids 
                  if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH)
    assert h1_cash == 20
    
    # H2 should have 30 cash (from fallback) and 30 deposit (from H1's deposit)
    h2_from_sys = sys.state.agents["H2"]
    h2_cash = sum(sys.state.contracts[cid].amount for cid in h2_from_sys.asset_ids 
                  if cid in sys.state.contracts and sys.state.contracts[cid].kind == InstrumentKind.CASH)
    assert h2_cash == 30  # from cash fallback
    assert sys.total_deposit("H2", "B2") == 30  # only the deposit portion
    
    sys.assert_invariants()


def test_deposit_coalescing():
    """Test that multiple deposits for same customer-bank pair are coalesced."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    h2 = Household(id="H2", name="Household 2", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)
    sys.add_agent(h2)
    
    # Multiple deposits from H1 to B1
    sys.mint_cash("H1", 150)
    deposit_cash(sys, "H1", "B1", 50)
    deposit_cash(sys, "H1", "B1", 30)
    deposit_cash(sys, "H1", "B1", 20)
    
    # Total should be 100
    assert sys.total_deposit("H1", "B1") == 100
    
    # Should have just one deposit instrument (coalesced)
    dep_ids = sys.deposit_ids("H1", "B1")
    assert len(dep_ids) == 1
    
    # Multiple payments to H2 should also coalesce
    client_payment(sys, "H1", "B1", "H2", "B1", 10)
    client_payment(sys, "H1", "B1", "H2", "B1", 15)
    client_payment(sys, "H1", "B1", "H2", "B1", 5)
    
    # H2 should have 30 total in one instrument
    assert sys.total_deposit("H2", "B1") == 30
    h2_deps = sys.deposit_ids("H2", "B1")
    assert len(h2_deps) == 1
    
    sys.assert_invariants()


def test_no_deposit_at_bank_error():
    """Test error when trying to withdraw from bank with no deposit."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)
    
    # H1 has no deposits
    with pytest.raises(ValidationError, match="no deposit at this bank"):
        withdraw_cash(sys, "H1", "B1", 50)
    
    sys.assert_invariants()


def test_insufficient_deposit_for_withdrawal():
    """Test error when trying to withdraw more than deposit balance."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="Household 1", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)
    
    # Setup: H1 has 50 deposit at B1
    sys.mint_cash("H1", 50)
    deposit_cash(sys, "H1", "B1", 50)
    
    # Try to withdraw 100
    with pytest.raises(ValidationError, match="insufficient deposit balance"):
        withdraw_cash(sys, "H1", "B1", 100)
    
    # Deposit unchanged
    assert sys.total_deposit("H1", "B1") == 50
    
    sys.assert_invariants()