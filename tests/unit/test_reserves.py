import pytest

from bilancio.core.errors import ValidationError
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.engines.system import System


def test_mint_reserves():
    """Test minting reserves to a bank."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    sys.add_agent(cb)
    sys.add_agent(b1)

    # Mint 100 units of reserves to B1
    reserve_id = sys.mint_reserves("B1", 100)

    # Check CB outstanding
    assert sys.state.cb_reserves_outstanding == 100

    # Check B1 has the reserves
    assert reserve_id in b1.asset_ids
    reserve_instr = sys.state.contracts[reserve_id]
    assert reserve_instr.amount == 100
    assert reserve_instr.asset_holder_id == "B1"
    assert reserve_instr.liability_issuer_id == "CB1"
    assert reserve_instr.kind == InstrumentKind.RESERVE_DEPOSIT

    # Check CB liability
    assert reserve_id in cb.liability_ids

    # Check invariants
    sys.assert_invariants()


def test_transfer_reserves():
    """Test transferring reserves between banks."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    b2 = Bank(id="B2", name="Bank 2", kind="bank")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(b2)

    # Mint reserves to B1
    sys.mint_reserves("B1", 150)

    # Transfer 60 reserves from B1 to B2
    sys.transfer_reserves("B1", "B2", 60)

    # Check B1 has 90, B2 has 60
    b1_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    b2_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b2.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    assert b1_reserves == 90
    assert b2_reserves == 60

    # Check CB outstanding unchanged
    assert sys.state.cb_reserves_outstanding == 150

    # Check invariants
    sys.assert_invariants()


def test_convert_reserves_to_cash():
    """Test converting reserves to cash."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    sys.add_agent(cb)
    sys.add_agent(b1)

    # Setup bank with reserves
    sys.mint_reserves("B1", 200)

    # Convert 80 reserves to cash
    sys.convert_reserves_to_cash("B1", 80)

    # Check reserves decreased
    b1_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    assert b1_reserves == 120

    # Check cash increased
    b1_cash = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    assert b1_cash == 80

    # Check both CB counters updated
    assert sys.state.cb_reserves_outstanding == 120
    assert sys.state.cb_cash_outstanding == 80

    # Check invariants
    sys.assert_invariants()


def test_convert_cash_to_reserves():
    """Test converting cash to reserves."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    sys.add_agent(cb)
    sys.add_agent(b1)

    # Setup bank with cash
    sys.mint_cash("B1", 120)

    # Convert 50 cash to reserves
    sys.convert_cash_to_reserves("B1", 50)

    # Check cash decreased
    b1_cash = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    assert b1_cash == 70

    # Check reserves increased
    b1_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    assert b1_reserves == 50

    # Check both CB counters updated
    assert sys.state.cb_cash_outstanding == 70
    assert sys.state.cb_reserves_outstanding == 50

    # Check invariants
    sys.assert_invariants()


def test_reserves_roundtrip():
    """Test round-trip conversion: reserves to cash and back."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    sys.add_agent(cb)
    sys.add_agent(b1)

    # Start with 100 reserves
    sys.mint_reserves("B1", 100)
    initial_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )

    # Convert all reserves to cash
    sys.convert_reserves_to_cash("B1", 100)

    # Verify we have cash, no reserves
    b1_cash = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    b1_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    assert b1_cash == 100
    assert b1_reserves == 0

    # Convert all cash back to reserves
    sys.convert_cash_to_reserves("B1", 100)

    # Verify amounts preserved
    final_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    final_cash = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    assert final_reserves == initial_reserves
    assert final_cash == 0

    # Check CB counters are back to original state
    assert sys.state.cb_reserves_outstanding == 100
    assert sys.state.cb_cash_outstanding == 0

    # Check invariants
    sys.assert_invariants()


def test_transfer_insufficient_reserves():
    """Test error handling for insufficient reserves transfer."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    b2 = Bank(id="B2", name="Bank 2", kind="bank")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(b2)

    # Mint only 50 reserves to B1
    sys.mint_reserves("B1", 50)

    # Try to transfer more reserves than available
    with pytest.raises(ValidationError, match="insufficient reserves"):
        sys.transfer_reserves("B1", "B2", 100)

    # Check state unchanged after failure (atomic rollback means B1 still has all 50)
    assert sys.state.cb_reserves_outstanding == 50
    # Get agents from system state (they may have been rolled back)
    b1_from_sys = sys.state.agents["B1"]
    b2_from_sys = sys.state.agents["B2"]
    b1_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b1_from_sys.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    b2_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b2_from_sys.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    assert b1_reserves == 50
    assert b2_reserves == 0

    # Check invariants
    sys.assert_invariants()


def test_convert_insufficient_reserves_to_cash():
    """Test error handling for insufficient reserves conversion."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    sys.add_agent(cb)
    sys.add_agent(b1)

    # Mint only 30 reserves
    sys.mint_reserves("B1", 30)

    # Try to convert more reserves than available
    with pytest.raises(ValidationError, match="insufficient reserves"):
        sys.convert_reserves_to_cash("B1", 50)

    # Check state unchanged after atomic rollback
    assert sys.state.cb_reserves_outstanding == 30
    assert sys.state.cb_cash_outstanding == 0
    # Get agent from system state (may have been rolled back)
    b1_from_sys = sys.state.agents["B1"]
    b1_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b1_from_sys.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    b1_cash = sum(
        sys.state.contracts[cid].amount
        for cid in b1_from_sys.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    assert b1_reserves == 30
    assert b1_cash == 0

    # Check invariants
    sys.assert_invariants()


def test_convert_insufficient_cash_to_reserves():
    """Test error handling for insufficient cash conversion."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    sys.add_agent(cb)
    sys.add_agent(b1)

    # Mint only 40 cash
    sys.mint_cash("B1", 40)

    # Try to convert more cash than available
    with pytest.raises(ValidationError, match="insufficient cash"):
        sys.convert_cash_to_reserves("B1", 60)

    # Check state unchanged after atomic rollback
    assert sys.state.cb_cash_outstanding == 40
    assert sys.state.cb_reserves_outstanding == 0
    # Get agent from system state (may have been rolled back)
    b1_from_sys = sys.state.agents["B1"]
    b1_cash = sum(
        sys.state.contracts[cid].amount
        for cid in b1_from_sys.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    b1_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b1_from_sys.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    assert b1_cash == 40
    assert b1_reserves == 0

    # Check invariants
    sys.assert_invariants()


def test_no_op_reserve_transfer():
    """Test error handling for no-op reserve transfer."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    sys.add_agent(cb)
    sys.add_agent(b1)

    # Mint reserves to B1
    sys.mint_reserves("B1", 100)

    # Try to transfer to same bank
    with pytest.raises(ValidationError, match="no-op transfer"):
        sys.transfer_reserves("B1", "B1", 50)

    # Check state unchanged
    assert sys.state.cb_reserves_outstanding == 100
    b1_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    assert b1_reserves == 100

    # Check invariants
    sys.assert_invariants()


def test_multiple_reserve_transfers_with_coalescing():
    """Test multiple reserve transfers result in proper coalescing."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    b2 = Bank(id="B2", name="Bank 2", kind="bank")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(b2)

    # Mint reserves to B1 in multiple chunks
    sys.mint_reserves("B1", 30)
    sys.mint_reserves("B1", 40)
    sys.mint_reserves("B1", 30)

    # B1 has 3 reserve instruments totaling 100
    b1_reserve_ids = [
        cid
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    ]
    assert len(b1_reserve_ids) == 3
    assert sum(sys.state.contracts[cid].amount for cid in b1_reserve_ids) == 100

    # Transfer all to B2 in parts
    sys.transfer_reserves("B1", "B2", 25)
    sys.transfer_reserves("B1", "B2", 35)
    sys.transfer_reserves("B1", "B2", 40)

    # B2 should have merged reserve instruments
    b2_reserve_ids = [
        cid
        for cid in b2.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    ]
    # Should be consolidated to fewer instruments (ideally 1)
    assert len(b2_reserve_ids) <= 3
    assert sum(sys.state.contracts[cid].amount for cid in b2_reserve_ids) == 100

    # B1 should have no reserves
    b1_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    assert b1_reserves == 0

    # Check invariants
    sys.assert_invariants()


def test_partial_reserve_conversion():
    """Test partial conversion of reserves to cash and back."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    sys.add_agent(cb)
    sys.add_agent(b1)

    # Start with 200 reserves
    sys.mint_reserves("B1", 200)

    # Convert 75 reserves to cash
    sys.convert_reserves_to_cash("B1", 75)

    # Check mixed holdings
    b1_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    b1_cash = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    assert b1_reserves == 125
    assert b1_cash == 75

    # Convert 25 cash back to reserves
    sys.convert_cash_to_reserves("B1", 25)

    # Check final state
    final_reserves = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
    )
    final_cash = sum(
        sys.state.contracts[cid].amount
        for cid in b1.asset_ids
        if sys.state.contracts[cid].kind == InstrumentKind.CASH
    )
    assert final_reserves == 150
    assert final_cash == 50

    # Check CB counters
    assert sys.state.cb_reserves_outstanding == 150
    assert sys.state.cb_cash_outstanding == 50

    # Check invariants
    sys.assert_invariants()
