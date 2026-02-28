"""
Tests for ticket-based integrated pricing.

Verifies that compute_integrated_rate() correctly averages rates across
multiple ticket positions, matching the closed-form midpoint formula.
"""

import math
from decimal import Decimal

import pytest

from bilancio.banking.pricing_kernel import (
    PricingParams,
    compute_integrated_rate,
    compute_quotes,
    compute_tilted_midline,
)


@pytest.fixture
def params():
    """Standard pricing params matching existing test suite."""
    return PricingParams(
        reserve_remuneration_rate=Decimal("0.01"),
        cb_borrowing_rate=Decimal("0.03"),
        reserve_target=100000,
        symmetric_capacity=50000,
        ticket_size=10000,
        reserve_floor=10000,
    )


class TestIntegratedRateSingleTicket:
    """When amount <= ticket_size, integrated rate == spot rate."""

    def test_single_ticket_loan_matches_spot(self, params):
        """Loan of exactly one ticket size: integrated == spot."""
        inventory = 0
        ct = Decimal("0")
        ri = Decimal("0")

        spot = compute_quotes(inventory, ct, ri, params, day=1)
        _, integrated_r_L = compute_integrated_rate(
            current_inventory=inventory,
            amount=params.ticket_size,  # exactly 1 ticket
            direction=-1,
            cash_tightness=ct,
            risk_index=ri,
            params=params,
        )

        assert integrated_r_L == spot.loan_rate

    def test_sub_ticket_loan_matches_spot(self, params):
        """Loan smaller than ticket_size: still 1 ticket, matches spot."""
        inventory = 5000
        ct = Decimal("0.3")
        ri = Decimal("0.1")

        spot = compute_quotes(inventory, ct, ri, params, day=1)
        _, integrated_r_L = compute_integrated_rate(
            current_inventory=inventory,
            amount=params.ticket_size // 2,  # half a ticket
            direction=-1,
            cash_tightness=ct,
            risk_index=ri,
            params=params,
        )

        assert integrated_r_L == spot.loan_rate

    def test_single_ticket_deposit_matches_spot(self, params):
        """Deposit of one ticket size: integrated r_D == spot r_D."""
        inventory = 0
        ct = Decimal("0")
        ri = Decimal("0")

        spot = compute_quotes(inventory, ct, ri, params, day=1)
        integrated_r_D, _ = compute_integrated_rate(
            current_inventory=inventory,
            amount=params.ticket_size,
            direction=+1,
            cash_tightness=ct,
            risk_index=ri,
            params=params,
        )

        assert integrated_r_D == spot.deposit_rate


class TestIntegratedRateMultiTicketLoan:
    """Multi-ticket loans get more expensive (inventory walks down)."""

    def test_three_ticket_loan_more_expensive(self, params):
        """Loan = 3 tickets → integrated r_L > spot r_L."""
        inventory = 0
        ct = Decimal("0")
        ri = Decimal("0")

        spot = compute_quotes(inventory, ct, ri, params, day=1)
        _, integrated_r_L = compute_integrated_rate(
            current_inventory=inventory,
            amount=3 * params.ticket_size,
            direction=-1,
            cash_tightness=ct,
            risk_index=ri,
            params=params,
        )

        # Loans walk inventory DOWN → midline tilts UP → higher r_L
        assert integrated_r_L > spot.loan_rate

    def test_midpoint_position_is_correct(self, params):
        """3-ticket loan: midpoint at x₀ - 1*S (middle of 0,1,2)."""
        inventory = 20000
        ct = Decimal("0.2")
        ri = Decimal("0.1")

        _, integrated_r_L = compute_integrated_rate(
            current_inventory=inventory,
            amount=3 * params.ticket_size,
            direction=-1,
            cash_tightness=ct,
            risk_index=ri,
            params=params,
        )

        # Midpoint: inventory + direction * (n-1)*S/2 = 20000 + (-1)*2*10000/2 = 20000 - 10000 = 10000
        midpoint_inv = 10000
        expected_midline = compute_tilted_midline(midpoint_inv, ct, ri, params)
        expected_r_L = expected_midline + params.inside_width / 2

        assert integrated_r_L == expected_r_L


class TestIntegratedRateMultiTicketDeposit:
    """Multi-ticket deposits: bank pays less for large inflows."""

    def test_three_ticket_deposit_cheaper(self, params):
        """Deposit = 3 tickets → integrated r_D < spot r_D."""
        inventory = 0
        ct = Decimal("0")
        ri = Decimal("0")

        spot = compute_quotes(inventory, ct, ri, params, day=1)
        integrated_r_D, _ = compute_integrated_rate(
            current_inventory=inventory,
            amount=3 * params.ticket_size,
            direction=+1,
            cash_tightness=ct,
            risk_index=ri,
            params=params,
        )

        # Deposits walk inventory UP → midline tilts DOWN → lower r_D
        assert integrated_r_D < spot.deposit_rate


class TestIntegratedRateMatchesManualAverage:
    """Verify integrated rate matches manual tick-by-tick average."""

    def test_five_ticket_loan_matches_average(self, params):
        """Compute rate at each position manually, verify average matches."""
        inventory = 10000
        ct = Decimal("0.1")
        ri = Decimal("0.05")
        n_tickets = 5

        # Manual: compute r_L at each tick position
        manual_r_L_sum = Decimal("0")
        for i in range(n_tickets):
            tick_inv = inventory + (-1) * i * params.ticket_size
            midline = compute_tilted_midline(tick_inv, ct, ri, params)
            r_L_at_tick = midline + params.inside_width / 2
            manual_r_L_sum += r_L_at_tick

        manual_avg_r_L = manual_r_L_sum / n_tickets

        # Integrated (closed-form)
        _, integrated_r_L = compute_integrated_rate(
            current_inventory=inventory,
            amount=n_tickets * params.ticket_size,
            direction=-1,
            cash_tightness=ct,
            risk_index=ri,
            params=params,
        )

        assert integrated_r_L == manual_avg_r_L

    def test_four_ticket_deposit_matches_average(self, params):
        """Deposit direction: manual average matches closed-form."""
        inventory = -5000
        ct = Decimal("0.3")
        ri = Decimal("0.2")
        n_tickets = 4

        # Manual: compute r_D at each tick position
        manual_r_D_sum = Decimal("0")
        for i in range(n_tickets):
            tick_inv = inventory + (+1) * i * params.ticket_size
            midline = compute_tilted_midline(tick_inv, ct, ri, params)
            r_D_at_tick = midline - params.inside_width / 2
            # Apply same ceiling/floor discipline as compute_integrated_rate
            r_D_at_tick = min(r_D_at_tick, params.cb_borrowing_rate)
            r_D_at_tick = max(r_D_at_tick, Decimal("0"))
            manual_r_D_sum += r_D_at_tick

        manual_avg_r_D = manual_r_D_sum / n_tickets

        # Integrated (closed-form)
        integrated_r_D, _ = compute_integrated_rate(
            current_inventory=inventory,
            amount=n_tickets * params.ticket_size,
            direction=+1,
            cash_tightness=ct,
            risk_index=ri,
            params=params,
        )

        # Note: the closed-form applies ceiling discipline AFTER averaging
        # at the midpoint, while manual applies it per-tick. These match
        # when no individual tick hits the ceiling/floor.
        assert integrated_r_D == manual_avg_r_D

    def test_partial_last_ticket(self, params):
        """Amount not divisible by ticket_size: ceil rounds up."""
        amount = int(params.ticket_size * 2.5)  # 25000 → ceil = 3 tickets
        n_tickets = math.ceil(amount / params.ticket_size)
        assert n_tickets == 3

        inventory = 0
        ct = Decimal("0")
        ri = Decimal("0")

        _, integrated_r_L = compute_integrated_rate(
            current_inventory=inventory,
            amount=amount,
            direction=-1,
            cash_tightness=ct,
            risk_index=ri,
            params=params,
        )

        # Should be the same as a full 3-ticket loan
        _, three_ticket_r_L = compute_integrated_rate(
            current_inventory=inventory,
            amount=3 * params.ticket_size,
            direction=-1,
            cash_tightness=ct,
            risk_index=ri,
            params=params,
        )

        assert integrated_r_L == three_ticket_r_L
