import pytest
from decimal import Decimal
from bilancio.engines.system import System
from bilancio.domain.agents.household import Household
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.delivery import DeliveryObligation
from bilancio.core.errors import ValidationError


def test_settle_deliverable_obligation():
    """Test settling a delivery obligation (e.g., promise to deliver goods)."""
    sys = System()
    seller = Household(id="SELLER", name="Seller", kind="household")
    buyer = Household(id="BUYER", name="Buyer", kind="household")
    sys.add_agent(seller)
    sys.add_agent(buyer)
    
    # Create a promise to deliver 10 chairs
    promise_id = sys.create_delivery_obligation(
        from_agent="SELLER",  # Seller has liability to deliver
        to_agent="BUYER",     # Buyer has claim to receive
        sku="CHAIR",
        quantity=10,
        unit_price=Decimal("0"),
        due_day=1
    )
    
    # Verify the obligation exists
    assert promise_id in sys.state.contracts
    assert promise_id in buyer.asset_ids
    assert promise_id in seller.liability_ids
    
    # Settle the obligation (e.g., after chairs are delivered)
    sys.settle_obligation(promise_id)
    
    # Verify the obligation is extinguished
    assert promise_id not in sys.state.contracts
    assert promise_id not in buyer.asset_ids
    assert promise_id not in seller.liability_ids
    
    # Verify invariants still hold
    sys.assert_invariants()


def test_settle_payable_obligation():
    """Test settling a payable obligation."""
    sys = System()
    debtor = Household(id="DEBTOR", name="Debtor", kind="household")
    creditor = Household(id="CREDITOR", name="Creditor", kind="household")
    sys.add_agent(debtor)
    sys.add_agent(creditor)
    
    # Create a payable
    payable = Payable(
        id="PAY1",
        kind=InstrumentKind.PAYABLE,
        amount=100,
        denom="USD",
        asset_holder_id="CREDITOR",
        liability_issuer_id="DEBTOR",
        due_day=1
    )
    sys.add_contract(payable)
    
    # Verify the payable exists
    assert "PAY1" in sys.state.contracts
    assert "PAY1" in creditor.asset_ids
    assert "PAY1" in debtor.liability_ids
    
    # Settle the payable (e.g., after payment is made)
    sys.settle_obligation("PAY1")
    
    # Verify the payable is extinguished
    assert "PAY1" not in sys.state.contracts
    assert "PAY1" not in creditor.asset_ids
    assert "PAY1" not in debtor.liability_ids
    
    sys.assert_invariants()


def test_settle_nonexistent_contract_fails():
    """Test that settling a non-existent contract raises an error."""
    sys = System()
    h1 = Household(id="H1", name="H1", kind="household")
    sys.add_agent(h1)
    
    with pytest.raises(ValidationError, match="Contract FAKE123 not found"):
        sys.settle_obligation("FAKE123")


def test_settle_contract_not_in_holder_assets_fails():
    """Test error when contract is not in holder's assets (data inconsistency)."""
    sys = System()
    h1 = Household(id="H1", name="H1", kind="household")
    h2 = Household(id="H2", name="H2", kind="household")
    sys.add_agent(h1)
    sys.add_agent(h2)
    
    # Create a delivery obligation
    d_id = sys.create_delivery_obligation("H1", "H2", "ITEM", 1, Decimal("0"), due_day=1)
    
    # Manually corrupt the data by removing from holder's assets
    h2.asset_ids.remove(d_id)
    
    # Should fail with validation error
    with pytest.raises(ValidationError, match="not in holder's assets"):
        sys.settle_obligation(d_id)


def test_settle_contract_not_in_issuer_liabilities_fails():
    """Test error when contract is not in issuer's liabilities (data inconsistency)."""
    sys = System()
    h1 = Household(id="H1", name="H1", kind="household")
    h2 = Household(id="H2", name="H2", kind="household")
    sys.add_agent(h1)
    sys.add_agent(h2)
    
    # Create a delivery obligation
    d_id = sys.create_delivery_obligation("H1", "H2", "ITEM", 1, Decimal("0"), due_day=1)
    
    # Manually corrupt the data by removing from issuer's liabilities
    h1.liability_ids.remove(d_id)
    
    # Should fail with validation error
    with pytest.raises(ValidationError, match="not in issuer's liabilities"):
        sys.settle_obligation(d_id)


def test_settle_obligation_is_atomic():
    """Test that settle_obligation is atomic - failures roll back all changes."""
    sys = System()
    h1 = Household(id="H1", name="H1", kind="household")
    h2 = Household(id="H2", name="H2", kind="household")
    sys.add_agent(h1)
    sys.add_agent(h2)
    
    # Create a delivery obligation
    d_id = sys.create_delivery_obligation("H1", "H2", "ITEM", 5, Decimal("0"), due_day=1)
    
    # Corrupt data to force failure partway through
    original_assets = h2.asset_ids.copy()
    original_liabilities = h1.liability_ids.copy()
    h1.liability_ids.remove(d_id)  # This will cause settle to fail
    
    # Attempt to settle (should fail)
    with pytest.raises(ValidationError):
        sys.settle_obligation(d_id)
    
    # Verify nothing changed (atomic rollback)
    assert d_id in sys.state.contracts
    # Need to get the agent from sys.state after rollback
    h2_from_sys = sys.state.agents["H2"]
    assert d_id in h2_from_sys.asset_ids  # Should still be there despite partial execution
    # Note: h1.liability_ids was manually corrupted, so we can't check that
    
    # Fix the corruption and verify we can still settle properly
    h1_from_sys = sys.state.agents["H1"]
    h1_from_sys.liability_ids.append(d_id)
    sys.settle_obligation(d_id)
    assert d_id not in sys.state.contracts


def test_settle_obligation_logs_event():
    """Test that settling an obligation logs the appropriate event."""
    sys = System()
    h1 = Household(id="H1", name="H1", kind="household")
    h2 = Household(id="H2", name="H2", kind="household")
    sys.add_agent(h1)
    sys.add_agent(h2)
    
    # Create a delivery obligation
    d_id = sys.create_delivery_obligation("H1", "H2", "WIDGET", 3, Decimal("0"), due_day=1)
    
    # Clear existing events to make test cleaner
    sys.state.events.clear()
    
    # Settle the obligation
    sys.settle_obligation(d_id)
    
    # Check that an ObligationSettled event was logged
    assert len(sys.state.events) == 1
    event = sys.state.events[0]
    assert event["kind"] == "ObligationSettled"
    assert event["contract_id"] == d_id
    assert event["holder_id"] == "H2"
    assert event["issuer_id"] == "H1"
    assert event["contract_kind"] == "delivery_obligation"
    assert event["amount"] == 3


def test_complete_chair_transaction_with_settlement():
    """Test the complete chair transaction example with settlement."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    seller = Household(id="YOU", name="Chair Seller", kind="household")
    buyer = Household(id="ME", name="Chair Buyer", kind="household")
    
    sys.add_agent(cb)
    sys.add_agent(seller)
    sys.add_agent(buyer)
    
    # Step 1: Give buyer cash to pay
    sys.mint_cash("ME", 100)
    
    # Step 2: Create promise to deliver chairs
    chair_promise_id = sys.create_delivery_obligation(
        from_agent="YOU",  # Seller promises to deliver
        to_agent="ME",     # Buyer has claim to receive
        sku="CHAIR",
        quantity=10,
        unit_price=Decimal("0"),  # Test uses zero price
        due_day=1
    )
    
    # Step 3: Buyer pays seller
    sys.transfer_cash("ME", "YOU", 100)
    
    # Verify state after payment
    seller_cash = sum(sys.state.contracts[cid].amount for cid in seller.asset_ids
                      if sys.state.contracts[cid].kind == InstrumentKind.CASH)
    buyer_cash = sum(sys.state.contracts[cid].amount for cid in buyer.asset_ids
                     if sys.state.contracts[cid].kind == InstrumentKind.CASH)
    assert seller_cash == 100
    assert buyer_cash == 0
    assert chair_promise_id in buyer.asset_ids
    assert chair_promise_id in seller.liability_ids
    
    # Step 4: Later, seller delivers chairs (represented by settling the obligation)
    sys.settle_obligation(chair_promise_id)
    
    # Verify final state - obligation is extinguished
    assert chair_promise_id not in sys.state.contracts
    assert chair_promise_id not in buyer.asset_ids
    assert chair_promise_id not in seller.liability_ids
    
    # Seller still has the cash
    seller_cash = sum(sys.state.contracts[cid].amount for cid in seller.asset_ids
                      if sys.state.contracts[cid].kind == InstrumentKind.CASH)
    assert seller_cash == 100
    
    sys.assert_invariants()


def test_settle_multiple_obligations():
    """Test settling multiple obligations in sequence."""
    sys = System()
    h1 = Household(id="H1", name="H1", kind="household")
    h2 = Household(id="H2", name="H2", kind="household")
    h3 = Household(id="H3", name="H3", kind="household")
    sys.add_agent(h1)
    sys.add_agent(h2)
    sys.add_agent(h3)
    
    # Create multiple obligations
    d1 = sys.create_delivery_obligation("H1", "H2", "ITEM1", 5, Decimal("0"), due_day=1)
    d2 = sys.create_delivery_obligation("H2", "H3", "ITEM2", 3, Decimal("0"), due_day=1)
    d3 = sys.create_delivery_obligation("H1", "H3", "ITEM3", 7, Decimal("0"), due_day=1)
    
    # Settle them one by one
    sys.settle_obligation(d1)
    assert d1 not in sys.state.contracts
    assert d2 in sys.state.contracts
    assert d3 in sys.state.contracts
    
    sys.settle_obligation(d2)
    assert d2 not in sys.state.contracts
    assert d3 in sys.state.contracts
    
    sys.settle_obligation(d3)
    assert d3 not in sys.state.contracts
    
    # All obligations should be settled
    assert len([cid for cid in h1.liability_ids if cid in sys.state.contracts]) == 0
    assert len([cid for cid in h2.asset_ids if cid in sys.state.contracts]) == 0
    assert len([cid for cid in h3.asset_ids if cid in sys.state.contracts]) == 0
    
    sys.assert_invariants()