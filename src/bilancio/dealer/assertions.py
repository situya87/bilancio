"""
Programmatic assertions for dealer ring simulation (C1-C6).

These assertions implement the "fail loudly" checks from the specification's
examples document. They verify invariants after every event (trade, settlement,
rebucketing) to catch bugs early.

References:
- Specification Section 6 (Events)
- Specification Section 7 (Kernel and Quotes)
- Specification Section 9 (VBT Anchors)
- Examples Document Section 1 (Programmatic Assertions)
"""

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import DealerState, VBTState
    from .kernel import KernelParams


# Tolerance constants for floating-point and integer comparisons
EPSILON_CASH = Decimal("1e-10")  # Tolerance for cash comparisons
EPSILON_QTY = 0  # Tolerance for ticket counts (integer)


def assert_c1_double_entry(
    cash_changes: dict[str, Decimal],
    qty_changes: dict[str, int],
) -> None:
    """
    C1: Conservation - sum of cash/qty changes is zero.

    For every executed event (trade, settlement, rebucketing), the sum of all
    cash changes across parties must be zero, and the sum of all ticket quantity
    changes must be zero. This enforces double-entry bookkeeping.

    Args:
        cash_changes: Dictionary mapping party ID to cash change for this event.
                     Positive = cash received, negative = cash paid.
        qty_changes: Dictionary mapping party ID to ticket count change.
                    Positive = tickets received, negative = tickets given.

    Raises:
        AssertionError: If cash or quantity is not conserved within tolerance.

    Example:
        # Customer sells ticket to dealer at price 0.95
        assert_c1_double_entry(
            cash_changes={"customer": Decimal("0.95"), "dealer": Decimal("-0.95")},
            qty_changes={"customer": -1, "dealer": 1}
        )

    References:
        - Examples Doc Section 1.1 (C1. Double-entry and conservation)
        - Specification Section 6 (Event tables with mirrored entries)
    """
    total_cash: Decimal = sum(cash_changes.values(), Decimal(0))
    total_qty = sum(qty_changes.values())

    assert abs(total_cash) <= EPSILON_CASH, (
        f"C1 VIOLATION: Cash not conserved across parties. "
        f"Total cash change: {total_cash} (tolerance: {EPSILON_CASH}). "
        f"Cash changes by party: {cash_changes}"
    )

    assert abs(total_qty) <= EPSILON_QTY, (
        f"C1 VIOLATION: Ticket quantities not conserved across parties. "
        f"Total qty change: {total_qty} (tolerance: {EPSILON_QTY}). "
        f"Quantity changes by party: {qty_changes}"
    )


def assert_c2_quote_bounds(dealer: "DealerState", vbt: "VBTState") -> None:
    """
    C2: Quotes within outside bounds.

    Verifies that dealer interior quotes remain within VBT outside bounds:
    - b_c(x) >= B (dealer bid >= outside bid)
    - a_c(x) <= A (dealer ask <= outside ask)

    Also verifies pin detection consistency:
    - (dealer.ask == A) if and only if dealer.is_pinned_ask
    - (dealer.bid == B) if and only if dealer.is_pinned_bid

    Args:
        dealer: Current dealer state with computed quotes.
        vbt: VBT state with outside quotes A, B.

    Raises:
        AssertionError: If quotes violate bounds or pin flags are inconsistent.

    References:
        - Examples Doc Section 1.2 (C2. Quote bounds and pin detection)
        - Specification Section 7.5 (Interior quotes and clipping)
    """
    # Check bid bound
    assert dealer.bid >= vbt.B, (
        f"C2 VIOLATION: Dealer bid {dealer.bid} < outside bid {vbt.B}. "
        f"Bucket: {dealer.bucket_id}, Inventory x: {dealer.x}"
    )

    # Check ask bound
    assert dealer.ask <= vbt.A, (
        f"C2 VIOLATION: Dealer ask {dealer.ask} > outside ask {vbt.A}. "
        f"Bucket: {dealer.bucket_id}, Inventory x: {dealer.x}"
    )

    # Check pin detection consistency (ask)
    ask_at_outside = (dealer.ask == vbt.A)
    assert ask_at_outside == dealer.is_pinned_ask, (
        f"C2 VIOLATION: Ask pin detection inconsistent. "
        f"Dealer ask {dealer.ask} == VBT ask {vbt.A}: {ask_at_outside}, "
        f"but is_pinned_ask: {dealer.is_pinned_ask}. "
        f"Bucket: {dealer.bucket_id}"
    )

    # Check pin detection consistency (bid)
    bid_at_outside = (dealer.bid == vbt.B)
    assert bid_at_outside == dealer.is_pinned_bid, (
        f"C2 VIOLATION: Bid pin detection inconsistent. "
        f"Dealer bid {dealer.bid} == VBT bid {vbt.B}: {bid_at_outside}, "
        f"but is_pinned_bid: {dealer.is_pinned_bid}. "
        f"Bucket: {dealer.bucket_id}"
    )


def assert_c3_feasibility(
    dealer: "DealerState",
    side: str,
    params: "KernelParams",
) -> None:
    """
    C3: Pre-check feasibility before interior execution.

    Before applying an interior fill, verify the dealer can execute:
    - Customer SELL (dealer BUY): x + S <= X* and C >= b_c(x)
    - Customer BUY (dealer SELL): x >= S

    This prevents the dealer from over-extending capacity or selling inventory
    they don't have.

    Args:
        dealer: Current dealer state before execution.
        side: "BUY" (dealer buys from customer) or "SELL" (dealer sells to customer).
        params: Kernel parameters including ticket size S.

    Raises:
        AssertionError: If interior execution is not feasible.

    References:
        - Examples Doc Section 1.3 (C3. Feasibility pre-checks)
        - Specification Section 7.6 (Commit-to-quote feasibility)
    """
    S = params.S

    if side == "BUY":
        # Dealer buys from customer (customer SELL)
        # Check capacity constraint
        assert dealer.x + S <= dealer.X_star, (
            f"C3 VIOLATION: Dealer BUY capacity exceeded. "
            f"Current inventory x: {dealer.x}, ticket size S: {S}, "
            f"capacity X*: {dealer.X_star}. "
            f"x + S = {dealer.x + S} > {dealer.X_star}. "
            f"Bucket: {dealer.bucket_id}"
        )

        # Check cash constraint
        assert dealer.cash >= dealer.bid, (
            f"C3 VIOLATION: Dealer BUY cash insufficient. "
            f"Cash: {dealer.cash}, required bid: {dealer.bid}. "
            f"Bucket: {dealer.bucket_id}"
        )

    elif side == "SELL":
        # Dealer sells to customer (customer BUY)
        # Check inventory constraint
        assert dealer.x >= S, (
            f"C3 VIOLATION: Dealer SELL inventory insufficient. "
            f"Current inventory x: {dealer.x}, ticket size S: {S}. "
            f"Bucket: {dealer.bucket_id}"
        )

    else:
        raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'.")


def assert_c4_passthrough_invariant(
    dealer_before: "DealerState",
    dealer_after: "DealerState",
) -> None:
    """
    C4: Passthrough leaves dealer state unchanged.

    If the event is executed via pass-through to the VBT at a pin (ask pinned
    at A or bid pinned at B), the dealer's (x, C) must be exactly unchanged.

    This ensures that when the dealer routes an order to the VBT, the dealer's
    own balance sheet is not affected.

    Args:
        dealer_before: Dealer state snapshot before the passthrough event.
        dealer_after: Dealer state snapshot after the passthrough event.

    Raises:
        AssertionError: If dealer inventory or cash changed during passthrough.

    References:
        - Examples Doc Section 1.4 (C4. Pass-through invariants)
        - Specification Events 9-10 (Passthrough at pins)
    """
    # Check inventory unchanged
    assert dealer_after.x == dealer_before.x, (
        f"C4 VIOLATION: Dealer inventory changed during passthrough. "
        f"Before: x={dealer_before.x}, After: x={dealer_after.x}. "
        f"Bucket: {dealer_after.bucket_id}"
    )

    # Check cash unchanged
    assert dealer_after.cash == dealer_before.cash, (
        f"C4 VIOLATION: Dealer cash changed during passthrough. "
        f"Before: C={dealer_before.cash}, After: C={dealer_after.cash}. "
        f"Bucket: {dealer_after.bucket_id}"
    )


def assert_c5_equity_basis(
    dealer: "DealerState",
    vbt: "VBTState",
) -> None:
    """
    C5: Equity = C + M*a at VBT mid.

    Dealer/VBT equity is measured at VBT mid M. The accounting identity
    V = C + M*a must hold, where:
    - V is the dealer's mid-valued equity
    - C is cash holdings
    - M is the VBT mid price
    - a is the number of tickets held

    Args:
        dealer: Current dealer state.
        vbt: VBT state with current mid M.

    Raises:
        AssertionError: If equity identity is violated beyond tolerance.

    References:
        - Examples Doc Section 1.5 (C5. Equity basis by role)
        - Specification Section 5.2 (Equity conventions)
    """
    # Compute equity at VBT mid: V = C + M*x where x = a*S (face inventory)
    equity_computed = dealer.cash + vbt.M * dealer.x

    assert abs(equity_computed - dealer.V) <= EPSILON_CASH, (
        f"C5 VIOLATION: Equity identity failed. "
        f"Computed E = C + M*x = {dealer.cash} + {vbt.M}*{dealer.x} = {equity_computed}, "
        f"but dealer.V = {dealer.V}. "
        f"Difference: {abs(equity_computed - dealer.V)} (tolerance: {EPSILON_CASH}). "
        f"Bucket: {dealer.bucket_id}"
    )


def assert_c6_anchor_timing(
    vbt_before: "VBTState",
    vbt_after: "VBTState",
    during_order_flow: bool,
) -> None:
    """
    C6: Anchors don't change during order flow.

    At the end of period t, compute loss rate and update (M, O). During period
    t+1 order flow, do NOT mutate anchors. This enforces the timing discipline
    that anchors are only updated based on realized losses, not contemporaneous
    order flow.

    Args:
        vbt_before: VBT state snapshot before order flow.
        vbt_after: VBT state snapshot after order flow.
        during_order_flow: True if checking within an order flow phase, False otherwise.

    Raises:
        AssertionError: If anchors changed during order flow.

    References:
        - Examples Doc Section 1.6 (C6. Anchor-update timing discipline)
        - Specification Section 9 (Loss-based anchor updates)
        - Specification Section 11 (Event loop timing)
    """
    if during_order_flow:
        # During order flow, anchors must not change
        assert vbt_after.M == vbt_before.M, (
            f"C6 VIOLATION: VBT mid M changed during order flow. "
            f"Before: M={vbt_before.M}, After: M={vbt_after.M}. "
            f"Bucket: {vbt_after.bucket_id}"
        )

        assert vbt_after.O == vbt_before.O, (
            f"C6 VIOLATION: VBT spread O changed during order flow. "
            f"Before: O={vbt_before.O}, After: O={vbt_after.O}. "
            f"Bucket: {vbt_after.bucket_id}"
        )


def run_all_assertions(
    dealer: "DealerState",
    vbt: "VBTState",
    params: "KernelParams",
) -> None:
    """
    Run C2 and C5 assertions (state checks).

    This is a convenience wrapper for running the assertions that check
    current state invariants (not event-specific). Use this after dealer
    state recomputation to verify the kernel produced valid results.

    Args:
        dealer: Current dealer state.
        vbt: Current VBT state.
        params: Kernel parameters.

    Raises:
        AssertionError: If any invariant is violated.

    Example:
        # After recomputing dealer state
        recompute_dealer_state(dealer, vbt, params)
        run_all_assertions(dealer, vbt, params)
    """
    assert_c2_quote_bounds(dealer, vbt)
    assert_c5_equity_basis(dealer, vbt)
