"""Coverage tests for bilancio.engines.clearing.

Targets uncovered lines:
- Line 76: net_amount == 0 (skip zero nets)
- Lines 85-87: negative net_amount path (bank_b owes bank_a)
- Line 97: contract is None check
- Line 102: not debtor_reserve_ids (no reserves at all)
- Lines 136-150: ValidationError fallback path (create overnight payable)
"""

from bilancio.core.errors import ValidationError
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.engines.clearing import compute_intraday_nets, settle_intraday_nets
from bilancio.engines.system import System
from bilancio.ops.banking import client_payment, deposit_cash


def _setup_two_bank_system() -> tuple:
    """Create a basic two-bank system with central bank and two households."""
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    b2 = Bank(id="B2", name="Bank 2", kind="bank")
    h1 = Household(id="H1", name="HH1", kind="household")
    h2 = Household(id="H2", name="HH2", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(b2)
    sys.add_agent(h1)
    sys.add_agent(h2)
    return sys, cb, b1, b2, h1, h2


class TestComputeIntradayNetsZero:
    """Cover line 76: net_amount == 0 results in skip during settlement."""

    def test_equal_opposing_payments_net_to_zero(self):
        """When payments between banks cancel exactly, net is zero."""
        sys, cb, b1, b2, h1, h2 = _setup_two_bank_system()

        # Give banks reserves and households deposits
        sys.mint_reserves("B1", 1000)
        sys.mint_reserves("B2", 1000)
        sys.mint_cash("H1", 100)
        sys.mint_cash("H2", 100)
        deposit_cash(sys, "H1", "B1", 100)
        deposit_cash(sys, "H2", "B2", 100)

        # H1@B1 pays H2@B2: 50
        client_payment(sys, "H1", "B1", "H2", "B2", 50)
        # H2@B2 pays H1@B1: 50 (exactly cancels)
        client_payment(sys, "H2", "B2", "H1", "B1", 50)

        current_day = sys.state.day
        nets = compute_intraday_nets(sys, current_day)

        # Net is exactly zero between B1 and B2
        assert nets.get(("B1", "B2"), 0) == 0

        # Settlement should be a no-op (no events created)
        settle_intraday_nets(sys, current_day)

        interbank_events = [
            e
            for e in sys.state.events
            if e["kind"] in ("InterbankCleared", "InterbankOvernightCreated")
        ]
        assert len(interbank_events) == 0

        sys.assert_invariants()


class TestComputeIntradayNetsNegative:
    """Cover lines 85-87: negative net_amount (bank_b owes bank_a)."""

    def test_creditor_bank_lexically_greater(self):
        """When the larger-named bank is the net debtor, net is negative.

        If B2 owes more to B1 than B1 owes to B2, the net for (B1, B2)
        is negative, so bank_b becomes the debtor.
        """
        sys, cb, b1, b2, h1, h2 = _setup_two_bank_system()

        sys.mint_reserves("B1", 1000)
        sys.mint_reserves("B2", 1000)
        sys.mint_cash("H1", 200)
        sys.mint_cash("H2", 200)
        deposit_cash(sys, "H1", "B1", 200)
        deposit_cash(sys, "H2", "B2", 200)

        # H2@B2 pays H1@B1: 150 (B2 owes B1: 150)
        client_payment(sys, "H2", "B2", "H1", "B1", 150)
        # H1@B1 pays H2@B2: 30 (B1 owes B2: 30, net B2 owes B1: 120)
        client_payment(sys, "H1", "B1", "H2", "B2", 30)

        current_day = sys.state.day
        nets = compute_intraday_nets(sys, current_day)

        # With lexical ordering: (B1, B2). B1 owes B2: +30, B2 owes B1: -150
        # Net = 30 - 150 = -120 (negative means B2 owes B1)
        assert ("B1", "B2") in nets
        assert nets[("B1", "B2")] == -120

        # Settle: B2 should transfer reserves to B1
        settle_intraday_nets(sys, current_day)

        # Verify B2 reserves decreased by 120
        b2_reserves = sum(
            sys.state.contracts[cid].amount
            for cid in sys.state.agents["B2"].asset_ids
            if cid in sys.state.contracts
            and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
        )
        assert b2_reserves == 880  # 1000 - 120

        # Verify B1 reserves increased by 120
        b1_reserves = sum(
            sys.state.contracts[cid].amount
            for cid in sys.state.agents["B1"].asset_ids
            if cid in sys.state.contracts
            and sys.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
        )
        assert b1_reserves == 1120  # 1000 + 120

        # InterbankCleared event should show B2 as debtor
        cleared = [e for e in sys.state.events if e["kind"] == "InterbankCleared"]
        assert len(cleared) == 1
        assert cleared[0]["debtor_bank"] == "B2"
        assert cleared[0]["creditor_bank"] == "B1"
        assert cleared[0]["amount"] == 120

        sys.assert_invariants()


class TestNoReservesAvailable:
    """Cover lines 97 (contract is None) and 102 (no reserve IDs)."""

    def test_debtor_bank_has_no_reserves(self):
        """When debtor bank has zero reserve deposits, overnight payable is created."""
        sys, cb, b1, b2, h1, h2 = _setup_two_bank_system()

        # B1 has NO reserves, B2 has reserves
        sys.mint_reserves("B2", 1000)
        sys.mint_cash("H1", 200)
        sys.mint_cash("H2", 100)
        deposit_cash(sys, "H1", "B1", 200)
        deposit_cash(sys, "H2", "B2", 100)

        # H1@B1 pays H2@B2: 100 (B1 owes B2: 100, but B1 has no reserves)
        client_payment(sys, "H1", "B1", "H2", "B2", 100)

        current_day = sys.state.day

        # Settle: B1 has no reserves, so overnight payable should be created
        settle_intraday_nets(sys, current_day)

        # Check overnight payable was created
        overnight_events = [e for e in sys.state.events if e["kind"] == "InterbankOvernightCreated"]
        assert len(overnight_events) == 1
        assert overnight_events[0]["debtor_bank"] == "B1"
        assert overnight_events[0]["creditor_bank"] == "B2"
        assert overnight_events[0]["amount"] == 100
        assert overnight_events[0]["due_day"] == current_day + 1

        # No InterbankCleared events
        cleared = [e for e in sys.state.events if e["kind"] == "InterbankCleared"]
        assert len(cleared) == 0

    def test_stale_contract_reference_is_none(self):
        """Cover line 97: contract is None for a stale asset_id reference.

        Manually insert a stale contract ID into the bank's asset_ids
        that does not exist in state.contracts.
        """
        sys, cb, b1, b2, h1, h2 = _setup_two_bank_system()

        # Give B1 a small reserve and a stale reference
        sys.mint_reserves("B1", 50)
        sys.mint_reserves("B2", 1000)

        # Inject a stale contract ID that doesn't exist in state.contracts
        sys.state.agents["B1"].asset_ids.append("STALE_CONTRACT_ID")

        sys.mint_cash("H1", 200)
        sys.mint_cash("H2", 100)
        deposit_cash(sys, "H1", "B1", 200)
        deposit_cash(sys, "H2", "B2", 100)

        # H1@B1 pays H2@B2: 40 (B1 owes B2: 40, B1 has 50 reserves)
        client_payment(sys, "H1", "B1", "H2", "B2", 40)

        current_day = sys.state.day

        # Settlement should still work -- skip the stale ID, use the real reserve
        settle_intraday_nets(sys, current_day)

        # Should clear successfully because B1 has enough real reserves (50 >= 40)
        cleared = [e for e in sys.state.events if e["kind"] == "InterbankCleared"]
        assert len(cleared) == 1
        assert cleared[0]["amount"] == 40


class TestValidationErrorFallback:
    """Cover lines 136-150: ValidationError fallback creates overnight payable."""

    def test_transfer_reserves_raises_validation_error(self):
        """When transfer_reserves raises ValidationError, fallback creates overnight payable."""
        sys, cb, b1, b2, h1, h2 = _setup_two_bank_system()

        sys.mint_reserves("B1", 1000)
        sys.mint_reserves("B2", 1000)
        sys.mint_cash("H1", 200)
        sys.mint_cash("H2", 100)
        deposit_cash(sys, "H1", "B1", 200)
        deposit_cash(sys, "H2", "B2", 100)

        # Create a cross-bank payment
        client_payment(sys, "H1", "B1", "H2", "B2", 80)

        current_day = sys.state.day

        # Patch transfer_reserves to raise ValidationError

        def failing_transfer(*args, **kwargs):
            raise ValidationError("simulated transfer failure")

        sys.transfer_reserves = failing_transfer

        # Settlement should fall back to creating overnight payable
        settle_intraday_nets(sys, current_day)

        # Verify overnight payable was created as fallback
        overnight_events = [e for e in sys.state.events if e["kind"] == "InterbankOvernightCreated"]
        assert len(overnight_events) == 1
        assert overnight_events[0]["debtor_bank"] == "B1"
        assert overnight_events[0]["creditor_bank"] == "B2"
        assert overnight_events[0]["amount"] == 80
        assert overnight_events[0]["due_day"] == current_day + 1

        # No InterbankCleared because transfer failed
        cleared = [e for e in sys.state.events if e["kind"] == "InterbankCleared"]
        assert len(cleared) == 0

        # Verify a Payable contract was created
        payables = [c for c in sys.state.contracts.values() if c.kind == InstrumentKind.PAYABLE]
        assert len(payables) == 1
        assert payables[0].amount == 80
        assert payables[0].liability_issuer_id == "B1"
        assert payables[0].asset_holder_id == "B2"
        assert payables[0].due_day == current_day + 1


class TestEdgeCases:
    """Additional edge cases for clearing coverage."""

    def test_no_events_produces_empty_nets(self):
        """No events at all yields empty nets dict."""
        sys = System()
        cb = CentralBank(id="CB1", name="CB", kind="central_bank")
        sys.add_agent(cb)

        nets = compute_intraday_nets(sys, 0)
        assert nets == {}

    def test_wrong_day_ignored(self):
        """Events from a different day are not included in nets."""
        sys, cb, b1, b2, h1, h2 = _setup_two_bank_system()

        sys.mint_reserves("B1", 1000)
        sys.mint_reserves("B2", 1000)
        sys.mint_cash("H1", 200)
        sys.mint_cash("H2", 100)
        deposit_cash(sys, "H1", "B1", 200)
        deposit_cash(sys, "H2", "B2", 100)

        # Payment on day 0
        client_payment(sys, "H1", "B1", "H2", "B2", 50)

        # Query for day 999 -- should find nothing
        nets = compute_intraday_nets(sys, 999)
        assert len(nets) == 0

    def test_settle_with_multiple_bank_pairs(self):
        """Test clearing handles multiple bank pairs correctly."""
        sys = System()
        cb = CentralBank(id="CB1", name="CB", kind="central_bank")
        b1 = Bank(id="B1", name="Bank 1", kind="bank")
        b2 = Bank(id="B2", name="Bank 2", kind="bank")
        b3 = Bank(id="B3", name="Bank 3", kind="bank")
        h1 = Household(id="H1", name="HH1", kind="household")
        h2 = Household(id="H2", name="HH2", kind="household")
        h3 = Household(id="H3", name="HH3", kind="household")
        for agent in [cb, b1, b2, b3, h1, h2, h3]:
            sys.add_agent(agent)

        sys.mint_reserves("B1", 1000)
        sys.mint_reserves("B2", 1000)
        sys.mint_reserves("B3", 1000)
        sys.mint_cash("H1", 200)
        sys.mint_cash("H2", 200)
        sys.mint_cash("H3", 200)
        deposit_cash(sys, "H1", "B1", 200)
        deposit_cash(sys, "H2", "B2", 200)
        deposit_cash(sys, "H3", "B3", 200)

        # Cross-bank payments: B1->B2, B2->B3
        client_payment(sys, "H1", "B1", "H2", "B2", 60)
        client_payment(sys, "H2", "B2", "H3", "B3", 40)

        current_day = sys.state.day
        nets = compute_intraday_nets(sys, current_day)

        # (B1, B2): B1 owes B2 = 60
        # (B2, B3): B2 owes B3 = 40
        assert nets[("B1", "B2")] == 60
        assert nets[("B2", "B3")] == 40

        settle_intraday_nets(sys, current_day)

        cleared = [e for e in sys.state.events if e["kind"] == "InterbankCleared"]
        assert len(cleared) == 2

        sys.assert_invariants()
