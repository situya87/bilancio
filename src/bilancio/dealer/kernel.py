"""
L1 dealer pricing kernel.

This module implements the core L1 dealer pricing formulas from Section 8
of the specification. It computes all derived dealer quantities from the
current inventory and VBT anchor prices.

The kernel is the mathematical heart of the dealer system, computing:
- Capacity: Maximum fundable buy tickets (K*, X*)
- Layoff probability: Risk-neutral hedging probability (λ)
- Inside width: Competitive equilibrium spread (I)
- Midline: Inventory-sensitive mid price p(x)
- Clipped quotes: Final bid/ask quotes b_c(x), a_c(x)

All arithmetic uses Decimal for precision - never float.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR

from .models import DealerState, VBTState, Ticket

# Guard threshold - when M <= M_MIN, dealer pins to outside quotes
M_MIN = Decimal("0.02")


@dataclass
class KernelParams:
    """
    Parameters for dealer kernel computation.

    Attributes:
        S: Standard ticket size (face value)
    """
    S: Decimal = Decimal(1)


def recompute_dealer_state(
    dealer: DealerState,
    vbt: VBTState,
    params: KernelParams,
) -> None:
    """
    Recompute all derived dealer quantities from current (inventory, cash).

    This is the core L1 kernel from specification Section 8. It updates
    all derived dealer quantities in-place based on:
    - Current inventory and cash (dealer balance sheet)
    - VBT anchor prices M, O (and derived A, B)
    - Ticket size S

    The kernel implements the complete pricing formula hierarchy:
    1. Compute current inventory (a, x)
    2. Check guard regime (M <= M_MIN) and pin if needed
    3. Compute capacity (K*, X*) and ladder parameters (N, λ)
    4. Compute inside width (I) from layoff probability
    5. Compute inventory-sensitive midline p(x)
    6. Compute interior quotes a(x), b(x)
    7. Clip to outside bounds and detect pins

    Args:
        dealer: DealerState to update (modified in-place)
        vbt: VBTState providing anchor prices
        params: KernelParams with ticket size S

    References:
        - Section 8: L1 Dealer Pricing Formulas
        - Section 8.4: Midline Formula
        - Section 8.5: Interior Quote Formulas
    """
    S = params.S
    M = vbt.M
    O = vbt.O
    A = vbt.A
    B = vbt.B

    # Step 1: Compute current inventory
    # a = number of tickets held
    # x = face inventory = a * S
    dealer.a = len(dealer.inventory)
    dealer.x = S * dealer.a

    # Step 2: Guard regime check
    # If M <= M_MIN, collapse to outside-only (no interior market)
    if M <= M_MIN:
        dealer.V = dealer.cash
        dealer.K_star = 0
        dealer.X_star = Decimal(0)
        dealer.N = 1
        dealer.lambda_ = Decimal(1)
        dealer.I = O  # Not used when pinned, but set for completeness
        dealer.midline = M
        dealer.bid = B
        dealer.ask = A
        dealer.is_pinned_bid = True
        dealer.is_pinned_ask = True
        return

    # Step 3: Normal computation (M > M_MIN)

    # V = Mid-valued inventory = M * a + C
    # This is the dealer's "equity" valued at VBT mid
    dealer.V = M * dealer.a + dealer.cash

    # K* = floor(V / M)
    # Maximum number of tickets dealer can buy without borrowing
    # Uses ROUND_FLOOR to ensure integer result
    K_star_decimal = (dealer.V / M).quantize(Decimal(1), rounding=ROUND_FLOOR)
    dealer.K_star = int(K_star_decimal)

    # X* = S * K*
    # One-sided capacity in face units (max buyable face value)
    dealer.X_star = S * dealer.K_star

    # N = K* + 1
    # Number of ladder rungs (including zero inventory)
    dealer.N = dealer.K_star + 1

    # λ = S / (X* + S)
    # Layoff probability (reflecting random walk boundary condition)
    # This is the risk-neutral probability of hitting the upper capacity bound
    if dealer.X_star + S > 0:
        dealer.lambda_ = S / (dealer.X_star + S)
    else:
        # Degenerate case: zero capacity
        dealer.lambda_ = Decimal(1)

    # I = λ * O
    # Inside width (competitive equilibrium spread)
    # Dealer spread is compressed by layoff probability
    dealer.I = dealer.lambda_ * O

    # Step 4: Inventory-sensitive midline (Section 8.4)
    # p(x) = M - (O / (X* + 2S)) * (x - X*/2)
    #
    # This formula makes the midline:
    # - Decrease as inventory increases (dealer wants to sell)
    # - Increase as inventory decreases (dealer wants to buy)
    # - Centered at balanced inventory x = X*/2
    if dealer.X_star + 2 * S > 0:
        slope = O / (dealer.X_star + 2 * S)
        dealer.midline = M - slope * (dealer.x - dealer.X_star / 2)
    else:
        # Degenerate case: zero capacity
        dealer.midline = M

    # Step 5: Interior quotes (Section 8.5)
    # a(x) = p(x) + I/2  (interior ask)
    # b(x) = p(x) - I/2  (interior bid)
    half_inside = dealer.I / 2
    a_interior = dealer.midline + half_inside
    b_interior = dealer.midline - half_inside

    # Step 6: Clipped quotes
    # a_c(x) = min(A, a(x))  (clip ask at outside ask)
    # b_c(x) = max(B, b(x))  (clip bid at outside bid)
    dealer.bid = max(B, b_interior)
    # Ask is floored at bid to ensure non-negative spread:
    # the dealer should never sell for less than it would buy.
    dealer.ask = max(dealer.bid, min(A, a_interior))

    # Step 7: Pin detection
    # Dealer is "pinned" when its quote equals the outside quote
    # This signals that the dealer wants to route to VBT
    dealer.is_pinned_ask = (dealer.ask >= A)
    dealer.is_pinned_bid = (dealer.bid == B)


def can_interior_buy(dealer: DealerState, params: KernelParams) -> bool:
    """
    Check if dealer can execute interior BUY (dealer buys ticket from customer).

    Interior buy is feasible if and only if:
    1. x + S <= X* (capacity constraint: has room for one more ticket)
    2. C >= b_c(x) (cash constraint: can pay the bid price)

    Args:
        dealer: DealerState to check
        params: KernelParams with ticket size S

    Returns:
        True if dealer can execute interior buy, False if must route to VBT

    References:
        - Section 8.6: Feasibility Checks
    """
    S = params.S
    has_capacity = (dealer.x + S <= dealer.X_star)
    has_cash = (dealer.cash >= dealer.bid)
    return has_capacity and has_cash


def can_interior_sell(dealer: DealerState, params: KernelParams) -> bool:
    """
    Check if dealer can execute interior SELL (dealer sells ticket to customer).

    Interior sell is feasible if and only if:
    - x >= S (inventory constraint: has at least one ticket to sell)
    - X* > 0 (not in Guard mode)

    Under Guard mode (M <= M_MIN), X* = 0 and all trades route to VBT
    regardless of dealer inventory. This is an "operational freeze of
    interior execution."

    Args:
        dealer: DealerState to check
        params: KernelParams with ticket size S

    Returns:
        True if dealer can execute interior sell, False if must route to VBT

    References:
        - Section 8.6: Feasibility Checks
        - Section 7.2: Guard regime when M <= M_MIN
    """
    S = params.S
    has_inventory = (dealer.x >= S)
    not_in_guard = (dealer.X_star > 0)
    return has_inventory and not_in_guard


@dataclass
class ExecutionResult:
    """
    Result of a trade execution.

    Attributes:
        executed: True if trade was executed
        price: Execution price (bid or ask)
        is_passthrough: True if trade was routed to VBT (outside market)
        ticket: Ticket that was transferred (for BUYs), None for failed trades
    """
    executed: bool
    price: Decimal
    is_passthrough: bool
    ticket: Ticket | None = None
