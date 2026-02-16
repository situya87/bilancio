"""
Tests for dealer bridge module.

This test file covers the four main functions in bilancio.dealer.bridge:
1. assign_bucket - assigns tickets to maturity buckets
2. payables_to_tickets - converts payables to tradable tickets
3. tickets_to_trader_holdings - groups tickets by owner
4. apply_trade_results_to_payables - updates payable holders from trades
"""

from decimal import Decimal

import pytest

from bilancio.dealer.bridge import (
    apply_trade_results_to_payables,
    assign_bucket,
    payables_to_tickets,
    tickets_to_trader_holdings,
)
from bilancio.dealer.models import DEFAULT_BUCKETS, BucketConfig, Ticket
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable


class TestAssignBucket:
    """Tests for assign_bucket function."""

    def test_short_bucket_lower_bound(self):
        """Test assignment to short bucket at lower bound (tau=1)."""
        result = assign_bucket(1, DEFAULT_BUCKETS)
        assert result == "short"

    def test_short_bucket_middle(self):
        """Test assignment to short bucket in middle of range (tau=2)."""
        result = assign_bucket(2, DEFAULT_BUCKETS)
        assert result == "short"

    def test_short_bucket_upper_bound(self):
        """Test assignment to short bucket at upper bound (tau=3)."""
        result = assign_bucket(3, DEFAULT_BUCKETS)
        assert result == "short"

    def test_mid_bucket_lower_bound(self):
        """Test assignment to mid bucket at lower bound (tau=4)."""
        result = assign_bucket(4, DEFAULT_BUCKETS)
        assert result == "mid"

    def test_mid_bucket_middle(self):
        """Test assignment to mid bucket in middle of range (tau=6)."""
        result = assign_bucket(6, DEFAULT_BUCKETS)
        assert result == "mid"

    def test_mid_bucket_upper_bound(self):
        """Test assignment to mid bucket at upper bound (tau=8)."""
        result = assign_bucket(8, DEFAULT_BUCKETS)
        assert result == "mid"

    def test_long_bucket_lower_bound(self):
        """Test assignment to long bucket at lower bound (tau=9)."""
        result = assign_bucket(9, DEFAULT_BUCKETS)
        assert result == "long"

    def test_long_bucket_unbounded(self):
        """Test assignment to long bucket with high tau (tau=100)."""
        result = assign_bucket(100, DEFAULT_BUCKETS)
        assert result == "long"

    def test_no_matching_bucket_zero(self):
        """Test error when tau=0 (no matching bucket)."""
        with pytest.raises(ValueError, match="No bucket found for remaining_tau=0"):
            assign_bucket(0, DEFAULT_BUCKETS)

    def test_no_matching_bucket_negative(self):
        """Test error when tau is negative."""
        with pytest.raises(ValueError, match="No bucket found for remaining_tau=-1"):
            assign_bucket(-1, DEFAULT_BUCKETS)

    def test_custom_buckets(self):
        """Test with custom bucket configuration."""
        custom_buckets = [
            BucketConfig("early", 1, 5),
            BucketConfig("late", 6, None),
        ]
        assert assign_bucket(3, custom_buckets) == "early"
        assert assign_bucket(5, custom_buckets) == "early"
        assert assign_bucket(6, custom_buckets) == "late"
        assert assign_bucket(20, custom_buckets) == "late"


class TestPayablesToTickets:
    """Tests for payables_to_tickets function."""

    def test_single_payable_single_ticket(self):
        """Test converting a single payable to a single ticket."""
        payables = {
            "P1": Payable(
                id="P1",
                kind=InstrumentKind.PAYABLE,
                amount=100,  # $1.00 in minor units
                denom="USD",
                asset_holder_id="A1",
                liability_issuer_id="A2",
                due_day=5,
            )
        }

        tickets, payable_to_tickets = payables_to_tickets(
            payables, current_day=0, bucket_configs=DEFAULT_BUCKETS, ticket_size=Decimal(1)
        )

        # Should have exactly 1 ticket
        assert len(tickets) == 1
        assert len(payable_to_tickets) == 1
        assert "P1" in payable_to_tickets
        assert len(payable_to_tickets["P1"]) == 1

        # Check ticket properties
        ticket_id = payable_to_tickets["P1"][0]
        ticket = tickets[ticket_id]
        assert ticket.issuer_id == "A2"
        assert ticket.owner_id == "A1"
        assert ticket.face == Decimal(1)
        assert ticket.maturity_day == 5
        assert ticket.remaining_tau == 5
        assert ticket.bucket_id == "mid"
        assert ticket.serial == 0

    def test_multiple_tickets_from_single_payable(self):
        """Test converting a payable with face value > 1 to multiple tickets."""
        payables = {
            "P1": Payable(
                id="P1",
                kind=InstrumentKind.PAYABLE,
                amount=500,  # $5.00 in minor units
                denom="USD",
                asset_holder_id="A1",
                liability_issuer_id="A2",
                due_day=2,
            )
        }

        tickets, payable_to_tickets = payables_to_tickets(
            payables, current_day=0, bucket_configs=DEFAULT_BUCKETS, ticket_size=Decimal(1)
        )

        # Should have exactly 5 tickets
        assert len(tickets) == 5
        assert len(payable_to_tickets["P1"]) == 5

        # All tickets should be in short bucket (tau=2)
        for ticket_id in payable_to_tickets["P1"]:
            ticket = tickets[ticket_id]
            assert ticket.bucket_id == "short"
            assert ticket.face == Decimal(1)
            assert ticket.remaining_tau == 2

        # Check serial numbers are sequential
        serials = [tickets[tid].serial for tid in payable_to_tickets["P1"]]
        assert serials == [0, 1, 2, 3, 4]

    def test_multiple_payables(self):
        """Test converting multiple payables to tickets."""
        payables = {
            "P1": Payable(
                id="P1",
                kind=InstrumentKind.PAYABLE,
                amount=200,  # $2.00
                denom="USD",
                asset_holder_id="A1",
                liability_issuer_id="A2",
                due_day=3,
            ),
            "P2": Payable(
                id="P2",
                kind=InstrumentKind.PAYABLE,
                amount=300,  # $3.00
                denom="USD",
                asset_holder_id="A3",
                liability_issuer_id="A4",
                due_day=10,
            ),
        }

        tickets, payable_to_tickets = payables_to_tickets(
            payables, current_day=0, bucket_configs=DEFAULT_BUCKETS, ticket_size=Decimal(1)
        )

        # Should have 2 + 3 = 5 tickets total
        assert len(tickets) == 5
        assert len(payable_to_tickets["P1"]) == 2
        assert len(payable_to_tickets["P2"]) == 3

        # P1 should be in short bucket (tau=3)
        for ticket_id in payable_to_tickets["P1"]:
            assert tickets[ticket_id].bucket_id == "short"
            assert tickets[ticket_id].remaining_tau == 3

        # P2 should be in long bucket (tau=10)
        for ticket_id in payable_to_tickets["P2"]:
            assert tickets[ticket_id].bucket_id == "long"
            assert tickets[ticket_id].remaining_tau == 10

    def test_different_ticket_sizes(self):
        """Test with different ticket sizes."""
        payables = {
            "P1": Payable(
                id="P1",
                kind=InstrumentKind.PAYABLE,
                amount=1000,  # $10.00
                denom="USD",
                asset_holder_id="A1",
                liability_issuer_id="A2",
                due_day=5,
            )
        }

        # Ticket size of $2.00
        tickets, payable_to_tickets = payables_to_tickets(
            payables, current_day=0, bucket_configs=DEFAULT_BUCKETS, ticket_size=Decimal(2)
        )

        # Should have 5 tickets of $2.00 each
        assert len(tickets) == 5
        assert len(payable_to_tickets["P1"]) == 5

        for ticket_id in payable_to_tickets["P1"]:
            assert tickets[ticket_id].face == Decimal(2)

    def test_current_day_affects_bucket_assignment(self):
        """Test that current_day affects bucket assignment through remaining_tau."""
        payables = {
            "P1": Payable(
                id="P1",
                kind=InstrumentKind.PAYABLE,
                amount=100,
                denom="USD",
                asset_holder_id="A1",
                liability_issuer_id="A2",
                due_day=10,
            )
        }

        # Day 0: tau=10, should be long
        tickets1, _ = payables_to_tickets(payables, current_day=0, bucket_configs=DEFAULT_BUCKETS)
        ticket_id1 = list(tickets1.keys())[0]
        assert tickets1[ticket_id1].bucket_id == "long"
        assert tickets1[ticket_id1].remaining_tau == 10

        # Day 5: tau=5, should be mid
        tickets2, _ = payables_to_tickets(payables, current_day=5, bucket_configs=DEFAULT_BUCKETS)
        ticket_id2 = list(tickets2.keys())[0]
        assert tickets2[ticket_id2].bucket_id == "mid"
        assert tickets2[ticket_id2].remaining_tau == 5

        # Day 8: tau=2, should be short
        tickets3, _ = payables_to_tickets(payables, current_day=8, bucket_configs=DEFAULT_BUCKETS)
        ticket_id3 = list(tickets3.keys())[0]
        assert tickets3[ticket_id3].bucket_id == "short"
        assert tickets3[ticket_id3].remaining_tau == 2

    def test_empty_payables(self):
        """Test with empty payables dictionary."""
        tickets, payable_to_tickets = payables_to_tickets(
            {}, current_day=0, bucket_configs=DEFAULT_BUCKETS
        )

        assert len(tickets) == 0
        assert len(payable_to_tickets) == 0

    def test_error_payable_already_matured(self):
        """Test error when payable has already matured."""
        payables = {
            "P1": Payable(
                id="P1",
                kind=InstrumentKind.PAYABLE,
                amount=100,
                denom="USD",
                asset_holder_id="A1",
                liability_issuer_id="A2",
                due_day=5,
            )
        }

        with pytest.raises(ValueError, match="Payable P1 has already matured"):
            payables_to_tickets(payables, current_day=5, bucket_configs=DEFAULT_BUCKETS)

        with pytest.raises(ValueError, match="Payable P1 has already matured"):
            payables_to_tickets(payables, current_day=10, bucket_configs=DEFAULT_BUCKETS)

    def test_error_payable_no_due_day(self):
        """Test error when payable has no due_day."""
        payables = {
            "P1": Payable(
                id="P1",
                kind=InstrumentKind.PAYABLE,
                amount=100,
                denom="USD",
                asset_holder_id="A1",
                liability_issuer_id="A2",
                due_day=None,  # Missing due_day
            )
        }

        with pytest.raises(ValueError, match="Payable P1 has no due_day"):
            payables_to_tickets(payables, current_day=0, bucket_configs=DEFAULT_BUCKETS)

    def test_error_face_value_not_divisible_by_ticket_size(self):
        """Test error when face value is not evenly divisible by ticket_size."""
        payables = {
            "P1": Payable(
                id="P1",
                kind=InstrumentKind.PAYABLE,
                amount=150,  # $1.50, not divisible by ticket_size=1
                denom="USD",
                asset_holder_id="A1",
                liability_issuer_id="A2",
                due_day=5,
            )
        }

        with pytest.raises(ValueError, match="not divisible by ticket size"):
            payables_to_tickets(
                payables, current_day=0, bucket_configs=DEFAULT_BUCKETS, ticket_size=Decimal(1)
            )

    def test_ticket_ids_are_unique(self):
        """Test that all generated ticket IDs are unique."""
        payables = {
            "P1": Payable(
                id="P1",
                kind=InstrumentKind.PAYABLE,
                amount=500,
                denom="USD",
                asset_holder_id="A1",
                liability_issuer_id="A2",
                due_day=5,
            ),
            "P2": Payable(
                id="P2",
                kind=InstrumentKind.PAYABLE,
                amount=300,
                denom="USD",
                asset_holder_id="A3",
                liability_issuer_id="A4",
                due_day=7,
            ),
        }

        tickets, _ = payables_to_tickets(payables, current_day=0, bucket_configs=DEFAULT_BUCKETS)

        # All ticket IDs should be unique
        ticket_ids = list(tickets.keys())
        assert len(ticket_ids) == len(set(ticket_ids))


class TestTicketsToTraderHoldings:
    """Tests for tickets_to_trader_holdings function."""

    def test_single_trader_single_ticket(self):
        """Test grouping a single ticket for a single trader."""
        ticket = Ticket(
            id="T1",
            issuer_id="A2",
            owner_id="A1",
            face=Decimal(1),
            maturity_day=5,
            remaining_tau=5,
            bucket_id="mid",
        )
        tickets = {"T1": ticket}
        agent_ids = {"A1"}

        holdings = tickets_to_trader_holdings(tickets, agent_ids)

        assert len(holdings) == 1
        assert "A1" in holdings
        assert len(holdings["A1"]) == 1
        assert holdings["A1"][0] == ticket

    def test_multiple_traders_multiple_tickets(self):
        """Test grouping tickets for multiple traders."""
        tickets = {
            "T1": Ticket(
                id="T1",
                issuer_id="A3",
                owner_id="A1",
                face=Decimal(1),
                maturity_day=5,
                remaining_tau=5,
                bucket_id="mid",
            ),
            "T2": Ticket(
                id="T2",
                issuer_id="A3",
                owner_id="A1",
                face=Decimal(1),
                maturity_day=5,
                remaining_tau=5,
                bucket_id="mid",
            ),
            "T3": Ticket(
                id="T3",
                issuer_id="A4",
                owner_id="A2",
                face=Decimal(1),
                maturity_day=7,
                remaining_tau=7,
                bucket_id="mid",
            ),
        }
        agent_ids = {"A1", "A2"}

        holdings = tickets_to_trader_holdings(tickets, agent_ids)

        assert len(holdings) == 2
        assert len(holdings["A1"]) == 2
        assert len(holdings["A2"]) == 1
        assert holdings["A1"][0].id in ["T1", "T2"]
        assert holdings["A1"][1].id in ["T1", "T2"]
        assert holdings["A2"][0].id == "T3"

    def test_trader_with_no_tickets(self):
        """Test that traders with no tickets get empty lists."""
        tickets = {
            "T1": Ticket(
                id="T1",
                issuer_id="A2",
                owner_id="A1",
                face=Decimal(1),
                maturity_day=5,
                remaining_tau=5,
                bucket_id="mid",
            ),
        }
        agent_ids = {"A1", "A3", "A4"}  # A3 and A4 have no tickets

        holdings = tickets_to_trader_holdings(tickets, agent_ids)

        assert len(holdings) == 3
        assert len(holdings["A1"]) == 1
        assert len(holdings["A3"]) == 0
        assert len(holdings["A4"]) == 0

    def test_filter_by_agent_ids(self):
        """Test that only tickets owned by specified agents are included."""
        tickets = {
            "T1": Ticket(
                id="T1",
                issuer_id="A3",
                owner_id="A1",
                face=Decimal(1),
                maturity_day=5,
                remaining_tau=5,
                bucket_id="mid",
            ),
            "T2": Ticket(
                id="T2",
                issuer_id="A4",
                owner_id="A2",
                face=Decimal(1),
                maturity_day=7,
                remaining_tau=7,
                bucket_id="mid",
            ),
            "T3": Ticket(
                id="T3",
                issuer_id="A5",
                owner_id="A5",
                face=Decimal(1),
                maturity_day=10,
                remaining_tau=10,
                bucket_id="long",
            ),
        }
        agent_ids = {"A1", "A2"}  # A5 not included

        holdings = tickets_to_trader_holdings(tickets, agent_ids)

        # Only A1 and A2 should be in holdings
        assert len(holdings) == 2
        assert "A1" in holdings
        assert "A2" in holdings
        assert "A5" not in holdings

        # A1 and A2 should each have 1 ticket
        assert len(holdings["A1"]) == 1
        assert len(holdings["A2"]) == 1

    def test_empty_tickets(self):
        """Test with empty tickets dictionary."""
        agent_ids = {"A1", "A2"}
        holdings = tickets_to_trader_holdings({}, agent_ids)

        assert len(holdings) == 2
        assert len(holdings["A1"]) == 0
        assert len(holdings["A2"]) == 0

    def test_empty_agent_ids(self):
        """Test with empty agent_ids set."""
        tickets = {
            "T1": Ticket(
                id="T1",
                issuer_id="A2",
                owner_id="A1",
                face=Decimal(1),
                maturity_day=5,
                remaining_tau=5,
                bucket_id="mid",
            ),
        }

        holdings = tickets_to_trader_holdings(tickets, set())

        assert len(holdings) == 0

    def test_tickets_are_references_not_copies(self):
        """Test that holdings contain references to original ticket objects."""
        ticket = Ticket(
            id="T1",
            issuer_id="A2",
            owner_id="A1",
            face=Decimal(1),
            maturity_day=5,
            remaining_tau=5,
            bucket_id="mid",
        )
        tickets = {"T1": ticket}
        agent_ids = {"A1"}

        holdings = tickets_to_trader_holdings(tickets, agent_ids)

        # Should be the same object (not a copy)
        assert holdings["A1"][0] is ticket


class TestApplyTradeResultsToPayables:
    """Tests for apply_trade_results_to_payables function."""

    def test_single_trade_updates_holder(self):
        """Test that a single trade updates the payable holder."""
        payable = Payable(
            id="P1",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="USD",
            asset_holder_id="A1",
            liability_issuer_id="A2",
            due_day=5,
        )
        payables = {"P1": payable}
        ticket_to_payable = {"T1": "P1"}
        trade_results = [
            {
                "ticket_id": "T1",
                "old_owner": "A1",
                "new_owner": "A3",
                "price": Decimal("0.95"),
            }
        ]

        apply_trade_results_to_payables(payables, ticket_to_payable, trade_results)

        # Payable holder should be updated to new owner
        assert payables["P1"].holder_id == "A3"
        # Original asset_holder_id should remain unchanged
        assert payables["P1"].asset_holder_id == "A1"

    def test_multiple_trades_same_payable(self):
        """Test multiple trades for tickets from the same payable."""
        payable = Payable(
            id="P1",
            kind=InstrumentKind.PAYABLE,
            amount=300,
            denom="USD",
            asset_holder_id="A1",
            liability_issuer_id="A2",
            due_day=5,
        )
        payables = {"P1": payable}
        ticket_to_payable = {
            "T1": "P1",
            "T2": "P1",
            "T3": "P1",
        }
        trade_results = [
            {
                "ticket_id": "T1",
                "old_owner": "A1",
                "new_owner": "A3",
                "price": Decimal("0.95"),
            },
            {
                "ticket_id": "T2",
                "old_owner": "A1",
                "new_owner": "A3",
                "price": Decimal("0.95"),
            },
            {
                "ticket_id": "T3",
                "old_owner": "A1",
                "new_owner": "A3",
                "price": Decimal("0.95"),
            },
        ]

        apply_trade_results_to_payables(payables, ticket_to_payable, trade_results)

        # All tickets from same payable should result in same holder
        assert payables["P1"].holder_id == "A3"

    def test_trades_for_different_payables(self):
        """Test trades for tickets from different payables."""
        payables = {
            "P1": Payable(
                id="P1",
                kind=InstrumentKind.PAYABLE,
                amount=100,
                denom="USD",
                asset_holder_id="A1",
                liability_issuer_id="A2",
                due_day=5,
            ),
            "P2": Payable(
                id="P2",
                kind=InstrumentKind.PAYABLE,
                amount=100,
                denom="USD",
                asset_holder_id="A3",
                liability_issuer_id="A4",
                due_day=7,
            ),
        }
        ticket_to_payable = {
            "T1": "P1",
            "T2": "P2",
        }
        trade_results = [
            {
                "ticket_id": "T1",
                "old_owner": "A1",
                "new_owner": "A5",
                "price": Decimal("0.95"),
            },
            {
                "ticket_id": "T2",
                "old_owner": "A3",
                "new_owner": "A6",
                "price": Decimal("0.92"),
            },
        ]

        apply_trade_results_to_payables(payables, ticket_to_payable, trade_results)

        # Each payable should have its holder updated independently
        assert payables["P1"].holder_id == "A5"
        assert payables["P2"].holder_id == "A6"

    def test_ticket_not_in_mapping_skipped(self):
        """Test that tickets not in ticket_to_payable mapping are skipped."""
        payable = Payable(
            id="P1",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="USD",
            asset_holder_id="A1",
            liability_issuer_id="A2",
            due_day=5,
        )
        payables = {"P1": payable}
        ticket_to_payable = {"T1": "P1"}
        trade_results = [
            {
                "ticket_id": "T1",
                "old_owner": "A1",
                "new_owner": "A3",
                "price": Decimal("0.95"),
            },
            {
                "ticket_id": "T999",  # Not in mapping
                "old_owner": "A1",
                "new_owner": "A4",
                "price": Decimal("0.90"),
            },
        ]

        # Should not raise error, just skip T999
        apply_trade_results_to_payables(payables, ticket_to_payable, trade_results)

        assert payables["P1"].holder_id == "A3"

    def test_empty_trade_results(self):
        """Test with empty trade results."""
        payable = Payable(
            id="P1",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="USD",
            asset_holder_id="A1",
            liability_issuer_id="A2",
            due_day=5,
        )
        payables = {"P1": payable}
        ticket_to_payable = {"T1": "P1"}

        # No trades should not modify anything
        apply_trade_results_to_payables(payables, ticket_to_payable, [])

        assert payables["P1"].holder_id is None

    def test_error_payable_not_found(self):
        """Test error when payable_id from mapping is not in payables dict."""
        payables = {
            "P1": Payable(
                id="P1",
                kind=InstrumentKind.PAYABLE,
                amount=100,
                denom="USD",
                asset_holder_id="A1",
                liability_issuer_id="A2",
                due_day=5,
            ),
        }
        ticket_to_payable = {
            "T1": "P1",
            "T2": "P999",  # P999 doesn't exist
        }
        trade_results = [
            {
                "ticket_id": "T2",
                "old_owner": "A1",
                "new_owner": "A3",
                "price": Decimal("0.95"),
            },
        ]

        with pytest.raises(KeyError, match="Payable P999 not found"):
            apply_trade_results_to_payables(payables, ticket_to_payable, trade_results)

    def test_modifies_payables_in_place(self):
        """Test that function modifies payables dict in-place."""
        payable = Payable(
            id="P1",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="USD",
            asset_holder_id="A1",
            liability_issuer_id="A2",
            due_day=5,
        )
        payables = {"P1": payable}
        ticket_to_payable = {"T1": "P1"}
        trade_results = [
            {
                "ticket_id": "T1",
                "old_owner": "A1",
                "new_owner": "A3",
                "price": Decimal("0.95"),
            }
        ]

        # Keep reference to original payable object
        original_payable = payables["P1"]

        apply_trade_results_to_payables(payables, ticket_to_payable, trade_results)

        # Should modify the same object (not create a new one)
        assert payables["P1"] is original_payable
        assert original_payable.holder_id == "A3"

    def test_trade_result_extra_fields_ignored(self):
        """Test that extra fields in trade results are ignored."""
        payable = Payable(
            id="P1",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="USD",
            asset_holder_id="A1",
            liability_issuer_id="A2",
            due_day=5,
        )
        payables = {"P1": payable}
        ticket_to_payable = {"T1": "P1"}
        trade_results = [
            {
                "ticket_id": "T1",
                "old_owner": "A1",
                "new_owner": "A3",
                "price": Decimal("0.95"),
                "timestamp": 12345,
                "venue": "dealer",
                "extra_field": "should_be_ignored",
            }
        ]

        # Should work fine, extra fields ignored
        apply_trade_results_to_payables(payables, ticket_to_payable, trade_results)

        assert payables["P1"].holder_id == "A3"
