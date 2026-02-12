"""
Unit tests for the L1 dealer pricing kernel.

This module verifies the dealer kernel formulas against the specification
using pytest and Decimal throughout. All numerical examples are from the
specification document.

References:
- Specification Section 8: L1 Dealer Pricing Formulas
- Examples Document: Balanced dealer state example
"""

import pytest
from decimal import Decimal
from copy import deepcopy

from bilancio.dealer import (
    Ticket,
    DealerState,
    VBTState,
    KernelParams,
    recompute_dealer_state,
    can_interior_buy,
    can_interior_sell,
    assert_c2_quote_bounds,
    assert_c5_equity_basis,
)
from bilancio.core.ids import new_id


# Helper function to create test fixtures
def make_dealer_vbt(
    a: int = 2,
    cash: Decimal = Decimal(2),
    M: Decimal = Decimal(1),
    O: Decimal = Decimal("0.30"),
    S: Decimal = Decimal(1),
) -> tuple[DealerState, VBTState, KernelParams]:
    """
    Create dealer/VBT with specified parameters for testing.

    Args:
        a: Number of tickets in dealer inventory
        cash: Dealer cash holdings
        M: VBT mid price
        O: VBT outside spread
        S: Standard ticket size (face value)

    Returns:
        Tuple of (DealerState, VBTState, KernelParams)
    """
    params = KernelParams(S=S)

    # Create VBT state
    vbt = VBTState(
        bucket_id="test",
        agent_id="vbt_agent",
        M=M,
        O=O,
    )
    vbt.recompute_quotes()

    # Create dealer state with inventory
    dealer = DealerState(
        bucket_id="test",
        agent_id="dealer_agent",
        cash=cash,
    )

    # Add tickets to inventory
    for i in range(a):
        ticket = Ticket(
            id=f"ticket_{i}",
            issuer_id="issuer_1",
            owner_id=dealer.agent_id,
            face=S,
            maturity_day=100,
            remaining_tau=10,
            bucket_id="test",
            serial=i,
        )
        dealer.inventory.append(ticket)

    # Recompute dealer state
    recompute_dealer_state(dealer, vbt, params)

    return dealer, vbt, params


class TestCapacityComputation:
    """Test K* = floor(V/M) and X* = S*K*."""

    def test_balanced_state(self):
        """Example: a=2, C=2, M=1 -> V=4, K*=4, X*=4."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal(1),
        )

        # Check V = M*a + C = 1*2 + 2 = 4
        assert dealer.V == Decimal(4)

        # Check K* = floor(V/M) = floor(4/1) = 4
        assert dealer.K_star == 4

        # Check X* = S*K* = 1*4 = 4
        assert dealer.X_star == Decimal(4)

    def test_fractional_capacity(self):
        """V=3.97 -> K*=3 (floor), X*=3."""
        # Set up: a=2, C=1.97, M=1.0
        # V = M*a + C = 1.0*2 + 1.97 = 3.97
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal("1.97"),
            M=Decimal(1),
        )

        # Check V
        assert dealer.V == Decimal("3.97")

        # Check K* = floor(3.97/1.0) = floor(3.97) = 3
        assert dealer.K_star == 3

        # Check X* = S*K* = 1*3 = 3
        assert dealer.X_star == Decimal(3)

    def test_zero_cash(self):
        """a=2, C=0, M=1 -> V=2, K*=2."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(0),
            M=Decimal(1),
        )

        # Check V = M*a + C = 1*2 + 0 = 2
        assert dealer.V == Decimal(2)

        # Check K* = floor(2/1) = 2
        assert dealer.K_star == 2

        # Check X* = S*K* = 1*2 = 2
        assert dealer.X_star == Decimal(2)


class TestLayoffProbability:
    """Test λ = S/(X* + S)."""

    def test_standard_case(self):
        """X*=4, S=1 -> λ = 1/5 = 0.2."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal(1),
        )

        # X* = 4, S = 1
        assert dealer.X_star == Decimal(4)

        # λ = S/(X*+S) = 1/(4+1) = 1/5 = 0.2
        expected_lambda = Decimal(1) / Decimal(5)
        assert dealer.lambda_ == expected_lambda

    def test_small_capacity(self):
        """X*=1, S=1 -> λ = 1/2 = 0.5."""
        # Set up: a=1, C=0, M=1 -> V=1, K*=1, X*=1
        dealer, vbt, params = make_dealer_vbt(
            a=1,
            cash=Decimal(0),
            M=Decimal(1),
        )

        assert dealer.X_star == Decimal(1)

        # λ = S/(X*+S) = 1/(1+1) = 1/2 = 0.5
        expected_lambda = Decimal(1) / Decimal(2)
        assert dealer.lambda_ == expected_lambda

    def test_zero_capacity(self):
        """X*=0 -> λ = 1 (degenerate case)."""
        # Set up: a=0, C=0, M=1 -> V=0, K*=0, X*=0
        dealer, vbt, params = make_dealer_vbt(
            a=0,
            cash=Decimal(0),
            M=Decimal(1),
        )

        assert dealer.X_star == Decimal(0)

        # λ = 1 (degenerate case)
        assert dealer.lambda_ == Decimal(1)


class TestInsideWidth:
    """Test I = λ * O."""

    def test_standard_case(self):
        """λ=0.2, O=0.30 -> I = 0.06."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal(1),
            O=Decimal("0.30"),
        )

        # λ = 1/5 = 0.2
        assert dealer.lambda_ == Decimal(1) / Decimal(5)

        # I = λ*O = 0.2*0.30 = 0.06
        expected_I = Decimal("0.2") * Decimal("0.30")
        assert dealer.I == expected_I

    def test_width_shrinks_with_capacity(self):
        """Higher capacity -> lower λ -> narrower I."""
        # Low capacity case: a=1, C=0 -> X*=1, λ=0.5
        dealer_low, vbt_low, params_low = make_dealer_vbt(
            a=1,
            cash=Decimal(0),
            M=Decimal(1),
            O=Decimal("0.30"),
        )

        # High capacity case: a=2, C=2 -> X*=4, λ=0.2
        dealer_high, vbt_high, params_high = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal(1),
            O=Decimal("0.30"),
        )

        # Verify λ decreases with capacity
        assert dealer_high.X_star > dealer_low.X_star
        assert dealer_high.lambda_ < dealer_low.lambda_

        # Verify I decreases with capacity
        assert dealer_high.I < dealer_low.I


class TestMidlineFormula:
    """Test p(x) = M - O/(X*+2S) * (x - X*/2)."""

    def test_balanced_inventory(self):
        """At x=X*/2=2 (balanced): p(2)=M=1.0."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal(1),
            O=Decimal("0.30"),
        )

        # X* = 4, x = a*S = 2*1 = 2 = X*/2 (balanced)
        assert dealer.X_star == Decimal(4)
        assert dealer.x == Decimal(2)

        # At balanced inventory, p(x) = M
        assert dealer.midline == Decimal(1)

    def test_low_inventory(self):
        """At x=0: p(0) > M (dealer wants to buy, higher price)."""
        # Create dealer with zero inventory
        dealer, vbt, params = make_dealer_vbt(
            a=0,
            cash=Decimal(4),
            M=Decimal(1),
            O=Decimal("0.30"),
        )

        # x=0, X*=4
        assert dealer.x == Decimal(0)
        assert dealer.X_star == Decimal(4)

        # p(0) = M - O/(X*+2S) * (0 - X*/2)
        #      = M - O/(X*+2S) * (-X*/2)
        #      = M + O*X* / (2*(X*+2S))
        #      > M (since O, X* > 0)
        assert dealer.midline > vbt.M

    def test_high_inventory(self):
        """At x=X*: p(X*) < M (dealer wants to sell, lower price)."""
        # Create dealer with x=X*=4 (full capacity)
        # Need a=4, C=0, M=1 -> V=4, K*=4, X*=4
        dealer, vbt, params = make_dealer_vbt(
            a=4,
            cash=Decimal(0),
            M=Decimal(1),
            O=Decimal("0.30"),
        )

        # x=4, X*=4
        assert dealer.x == Decimal(4)
        assert dealer.X_star == Decimal(4)

        # p(4) = M - O/(X*+2S) * (4 - 4/2)
        #      = M - O/(X*+2S) * (4 - 2)
        #      = M - 2*O/(X*+2S)
        #      < M (since O > 0)
        assert dealer.midline < vbt.M

    def test_midline_slope(self):
        """Verify slope = O/(X*+2S)."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal(1),
            O=Decimal("0.30"),
        )

        # X* = 4, S = 1
        # slope = O/(X*+2S) = 0.30/(4+2) = 0.30/6 = 0.05
        S = params.S
        expected_slope = vbt.O / (dealer.X_star + 2 * S)
        assert expected_slope == Decimal("0.05")

        # Verify the midline formula at specific points
        # p(x) = M - slope * (x - X*/2)

        # At x=0 (no inventory):
        x_test = Decimal(0)
        expected_p = vbt.M - expected_slope * (x_test - dealer.X_star / 2)

        # Create dealer with x=0 but same X* (maintain V=4 via cash)
        dealer_test, vbt_test, params_test = make_dealer_vbt(
            a=0,
            cash=Decimal(4),  # V = 0*1 + 4 = 4, K* = 4, X* = 4
            M=Decimal(1),
            O=Decimal("0.30"),
        )

        # Verify X* is same
        assert dealer_test.X_star == dealer.X_star

        # Verify midline matches formula
        # At x=0: p(0) = M - slope * (0 - 2) = M + 2*slope
        expected_p_0 = vbt.M + 2 * expected_slope
        assert abs(dealer_test.midline - expected_p_0) < Decimal("1e-10")

        # At x=4 (full capacity):
        dealer_test2, vbt_test2, params_test2 = make_dealer_vbt(
            a=4,
            cash=Decimal(0),  # V = 4*1 + 0 = 4, K* = 4, X* = 4
            M=Decimal(1),
            O=Decimal("0.30"),
        )

        # Verify X* is same
        assert dealer_test2.X_star == dealer.X_star

        # At x=4: p(4) = M - slope * (4 - 2) = M - 2*slope
        expected_p_4 = vbt.M - 2 * expected_slope
        assert abs(dealer_test2.midline - expected_p_4) < Decimal("1e-10")


class TestInteriorQuotes:
    """Test a(x) = p(x) + I/2, b(x) = p(x) - I/2."""

    def test_quotes_at_balanced(self):
        """At x=2: a(2)=0.83, b(2)=0.77."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal("0.80"),
            O=Decimal("0.30"),
        )

        # V = 0.80*2 + 2 = 3.60, K* = floor(3.60/0.80) = 4, X* = 4
        # At balanced inventory (x=2), p(2) = M = 0.80
        # λ = S/(X*+S) = 1/5 = 0.2
        # I = λ*O = 0.2*0.30 = 0.06
        # a(2) = p(2) + I/2 = 0.80 + 0.03 = 0.83
        # b(2) = p(2) - I/2 = 0.80 - 0.03 = 0.77

        assert dealer.midline == Decimal("0.80")
        assert dealer.I == Decimal("0.06")

        # Since interior quotes aren't clipped in this case:
        # ask = a(x) = p(x) + I/2
        # bid = b(x) = p(x) - I/2
        half_I = dealer.I / 2
        expected_ask = dealer.midline + half_I
        expected_bid = dealer.midline - half_I

        # Verify quotes (accounting for possible clipping)
        # In this case, A = M + O/2 = 0.80 + 0.15 = 0.95
        # B = M - O/2 = 0.80 - 0.15 = 0.65
        # So a(2) = 0.83 < 0.95 = A (not clipped)
        # And b(2) = 0.77 > 0.65 = B (not clipped)

        assert dealer.ask == expected_ask
        assert dealer.bid == expected_bid

        # Check exact values
        assert dealer.ask == Decimal("0.83")
        assert dealer.bid == Decimal("0.77")

    def test_spread_equals_inside_width(self):
        """a(x) - b(x) = I."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal("0.80"),
            O=Decimal("0.30"),
        )

        # Spread should equal I (assuming no clipping)
        spread = dealer.ask - dealer.bid
        assert spread == dealer.I


class TestClippedQuotes:
    """Test a_c(x) = min(A, a(x)), b_c(x) = max(B, b(x))."""

    def test_no_clipping_in_normal_case(self):
        """Interior quotes stay within outside bounds."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal("0.80"),
            O=Decimal("0.30"),
        )

        # A = M + O/2 = 0.80 + 0.15 = 0.95
        # B = M - O/2 = 0.80 - 0.15 = 0.65
        assert vbt.A == Decimal("0.95")
        assert vbt.B == Decimal("0.65")

        # a(2) = 0.83 < 0.95, so no clipping
        # b(2) = 0.77 > 0.65, so no clipping
        assert dealer.ask < vbt.A
        assert dealer.bid > vbt.B

        # Verify not pinned
        assert not dealer.is_pinned_ask
        assert not dealer.is_pinned_bid

    def test_quotes_never_exceed_bounds(self):
        """Property: b_c >= B and a_c <= A always."""
        # Test various inventory levels
        for a in [0, 1, 2, 3, 4]:
            dealer, vbt, params = make_dealer_vbt(
                a=a,
                cash=Decimal(2),
                M=Decimal("0.80"),
                O=Decimal("0.30"),
            )

            # Quotes must be within bounds
            assert dealer.bid >= vbt.B
            assert dealer.ask <= vbt.A


class TestPinDetection:
    """Test pin flags consistency."""

    def test_not_pinned_normal(self):
        """Normal case: not pinned."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal("0.80"),
            O=Decimal("0.30"),
        )

        # In balanced state with sufficient capacity, should not be pinned
        assert not dealer.is_pinned_ask
        assert not dealer.is_pinned_bid

    def test_pin_detection_consistency(self):
        """is_pinned_ask iff ask == A."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal(1),
            O=Decimal("0.30"),
        )

        # Check consistency
        assert dealer.is_pinned_ask == (dealer.ask == vbt.A)
        assert dealer.is_pinned_bid == (dealer.bid == vbt.B)


class TestGuardRegime:
    """Test M <= M_MIN behavior."""

    def test_guard_activates(self):
        """When M <= 0.02, X*=0 and quotes pinned."""
        from bilancio.dealer import M_MIN

        # Set M = M_MIN = 0.02
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=M_MIN,
            O=Decimal("0.30"),
        )

        # In guard regime: X* = 0
        assert dealer.X_star == Decimal(0)

        # Quotes pinned to outside
        assert dealer.is_pinned_ask
        assert dealer.is_pinned_bid

    def test_guard_pins_both_quotes(self):
        """Both ask and bid pinned to outside in guard."""
        from bilancio.dealer import M_MIN

        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=M_MIN - Decimal("0.01"),  # Below threshold
            O=Decimal("0.30"),
        )

        # Quotes must equal outside quotes
        assert dealer.ask == vbt.A
        assert dealer.bid == vbt.B


class TestFeasibilityChecks:
    """Test can_interior_buy and can_interior_sell."""

    def test_interior_buy_feasible(self):
        """x + S <= X* and C >= b_c(x)."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal(1),
        )

        # x=2, X*=4, so x+S=3 <= 4 ✓
        # C=2 >= b_c(2)=0.97 ✓
        assert can_interior_buy(dealer, params)

    def test_interior_buy_capacity_fails(self):
        """x + S > X* -> infeasible."""
        # Set up: a=4, C=0 -> x=4, X*=4
        dealer, vbt, params = make_dealer_vbt(
            a=4,
            cash=Decimal(0),
            M=Decimal(1),
        )

        # x=4, X*=4, so x+S=5 > 4 ✗
        assert not can_interior_buy(dealer, params)

    def test_interior_buy_cash_fails(self):
        """C < b_c(x) -> infeasible."""
        # Set up: a=2, C=0.50 (insufficient cash)
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal("0.50"),
            M=Decimal(1),
        )

        # bid = 0.97, cash = 0.50, so 0.50 < 0.97 ✗
        assert not can_interior_buy(dealer, params)

    def test_interior_sell_feasible(self):
        """x >= S."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal(1),
        )

        # x=2, S=1, so 2 >= 1 ✓
        assert can_interior_sell(dealer, params)

    def test_interior_sell_no_inventory(self):
        """x < S -> infeasible."""
        # Set up: a=0 -> x=0
        dealer, vbt, params = make_dealer_vbt(
            a=0,
            cash=Decimal(4),
            M=Decimal(1),
        )

        # x=0, S=1, so 0 < 1 ✗
        assert not can_interior_sell(dealer, params)


class TestInvariants:
    """Test C2 and C5 invariants hold after kernel computation."""

    def test_c2_quote_bounds(self):
        """Quotes within outside bounds."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal(1),
            O=Decimal("0.30"),
        )

        # Should not raise
        assert_c2_quote_bounds(dealer, vbt)

    def test_c2_quote_bounds_various_states(self):
        """Test C2 across various inventory states."""
        for a in [0, 1, 2, 3, 4]:
            for cash in [Decimal(0), Decimal(1), Decimal(2), Decimal(5)]:
                dealer, vbt, params = make_dealer_vbt(
                    a=a,
                    cash=cash,
                    M=Decimal(1),
                    O=Decimal("0.30"),
                )

                # Should never violate C2
                assert_c2_quote_bounds(dealer, vbt)

    def test_c5_equity_basis(self):
        """V = C + M*a."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal(1),
        )

        # Should not raise
        assert_c5_equity_basis(dealer, vbt)

    def test_c5_equity_basis_various_states(self):
        """Test C5 across various states."""
        for a in [0, 1, 2, 3, 4]:
            for cash in [Decimal(0), Decimal(1), Decimal(2), Decimal(5)]:
                for M in [Decimal("0.5"), Decimal(1), Decimal("1.5")]:
                    dealer, vbt, params = make_dealer_vbt(
                        a=a,
                        cash=cash,
                        M=M,
                        O=Decimal("0.30"),
                    )

                    # Should never violate C5
                    assert_c5_equity_basis(dealer, vbt)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_inventory_zero_cash(self):
        """Empty dealer: a=0, C=0."""
        dealer, vbt, params = make_dealer_vbt(
            a=0,
            cash=Decimal(0),
            M=Decimal(1),
        )

        assert dealer.V == Decimal(0)
        assert dealer.K_star == 0
        assert dealer.X_star == Decimal(0)

    def test_large_inventory(self):
        """Large inventory: a=100."""
        dealer, vbt, params = make_dealer_vbt(
            a=100,
            cash=Decimal(0),
            M=Decimal(1),
        )

        # V = 100, K* = 100, X* = 100
        assert dealer.V == Decimal(100)
        assert dealer.K_star == 100
        assert dealer.X_star == Decimal(100)

        # λ should be small with large capacity
        assert dealer.lambda_ < Decimal("0.01")

    def test_high_outside_spread(self):
        """Wide outside spread: O=1.0."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal(1),
            O=Decimal(1),
        )

        # I should be larger with wider O
        assert dealer.I == dealer.lambda_ * Decimal(1)

    def test_low_mid_price(self):
        """Low VBT mid: M=0.10."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal("0.10"),
            O=Decimal("0.05"),
        )

        # V = M*a + C = 0.10*2 + 2 = 2.20
        assert dealer.V == Decimal("2.20")

        # K* = floor(2.20 / 0.10) = floor(22) = 22
        assert dealer.K_star == 22

        # X* = 1 * 22 = 22
        assert dealer.X_star == Decimal(22)


class TestKernelRecomputation:
    """Test that kernel recomputation is idempotent and consistent."""

    def test_idempotency(self):
        """Recomputing twice should give same result."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal(1),
        )

        # Save state after first computation
        V_1 = dealer.V
        ask_1 = dealer.ask
        bid_1 = dealer.bid

        # Recompute again
        recompute_dealer_state(dealer, vbt, params)

        # Should be identical
        assert dealer.V == V_1
        assert dealer.ask == ask_1
        assert dealer.bid == bid_1

    def test_consistency_after_inventory_change(self):
        """Adding inventory should update all derived quantities."""
        dealer, vbt, params = make_dealer_vbt(
            a=2,
            cash=Decimal(2),
            M=Decimal(1),
        )

        # Save initial state
        x_before = dealer.x
        V_before = dealer.V

        # Add one ticket
        ticket = Ticket(
            id="ticket_new",
            issuer_id="issuer_1",
            owner_id=dealer.agent_id,
            face=params.S,
            maturity_day=100,
            remaining_tau=10,
            bucket_id="test",
            serial=999,
        )
        dealer.inventory.append(ticket)

        # Recompute
        recompute_dealer_state(dealer, vbt, params)

        # Verify changes
        assert dealer.x == x_before + params.S
        assert dealer.a == 3
        assert dealer.V > V_before  # V increased (assuming M > 0)
