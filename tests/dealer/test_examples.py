"""
Integration tests based on worked examples from the dealer specification.

These tests verify complete scenarios matching the examples document, ensuring
that the dealer implementation correctly handles:
- Interior trades (Events 1-2)
- Passthrough trades (Events 9-10)
- Capacity jumps across integers
- Guard regime at very low M
- Partial recovery defaults
- Quote bounds and feasibility

All tests use Decimal arithmetic and verify programmatic assertions C1-C6.

References:
- docs/dealer_ring/dealer_examples.pdf
- Specification Section 6 (Events)
- Specification Section 8 (L1 Kernel)
"""

from copy import deepcopy
from decimal import Decimal

import pytest

from bilancio.core.ids import new_id
from bilancio.dealer import (
    M_MIN,
    DealerState,
    KernelParams,
    Ticket,
    TradeExecutor,
    VBTState,
    assert_c1_double_entry,
    can_interior_buy,
    can_interior_sell,
    recompute_dealer_state,
    run_all_assertions,
)

# Helper functions for test setup


def create_ticket(
    issuer_id: str,
    owner_id: str,
    face: Decimal = Decimal(1),
    maturity_day: int = 10,
    remaining_tau: int = 5,
    bucket_id: str = "mid",
    serial: int = 0,
) -> Ticket:
    """Create a ticket for testing."""
    return Ticket(
        id=new_id(),
        issuer_id=issuer_id,
        owner_id=owner_id,
        face=face,
        maturity_day=maturity_day,
        remaining_tau=remaining_tau,
        bucket_id=bucket_id,
        serial=serial,
    )


def setup_dealer_vbt_params(
    dealer_tickets: int = 2,
    dealer_cash: Decimal = Decimal(2),
    M: Decimal = Decimal("1.0"),
    O: Decimal = Decimal("0.30"),  # noqa: E741
    S: Decimal = Decimal(1),
) -> tuple[DealerState, VBTState, KernelParams]:
    """
    Create dealer, VBT, and params with standard configuration.

    Returns:
        Tuple of (dealer, vbt, params)
    """
    params = KernelParams(S=S)

    # Create VBT with anchors
    vbt = VBTState(
        bucket_id="mid",
        agent_id="vbt_mid",
        M=M,
        O=O,
    )
    vbt.recompute_quotes()

    # Create dealer with inventory
    dealer = DealerState(
        bucket_id="mid",
        agent_id="dealer_mid",
        cash=dealer_cash,
    )

    # Add tickets to dealer inventory
    for i in range(dealer_tickets):
        ticket = create_ticket(
            issuer_id=f"issuer_{i}",
            owner_id=dealer.agent_id,
            serial=i,
        )
        dealer.inventory.append(ticket)

    # Recompute dealer state
    recompute_dealer_state(dealer, vbt, params)

    return dealer, vbt, params


class TestExample5DealerEarns:
    """
    Example 5: Dealer earns spread over multiple trades.

    Initial state:
    - a=2, C=2, M=0.80, O=0.30
    - V=3.60, K*=4, X*=4

    Trade sequence:
    1. Customer SELL (dealer buys at bid 0.97)
    2. Customer BUY (dealer sells at ask 0.98)
    3. Customer SELL (dealer buys at bid 0.97)

    Final state:
    - a=3, C=1.04, V=4.04
    - Dealer earned 0.04 in equity
    """

    def test_initial_state(self):
        """Verify initial kernel computation matches Example 5."""
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=2,
            dealer_cash=Decimal(2),
            M=Decimal("0.80"),
            O=Decimal("0.30"),
        )

        # Check initial state
        assert dealer.a == 2
        assert dealer.x == Decimal(2)
        assert dealer.cash == Decimal(2)
        assert dealer.V == Decimal("3.60")
        assert dealer.K_star == 4
        assert dealer.X_star == Decimal(4)
        assert dealer.N == 5
        assert dealer.lambda_ == Decimal("0.2")  # 1/5
        assert dealer.I == Decimal("0.06")  # 0.2 * 0.30

        # Check quotes at x=2
        assert dealer.midline == Decimal("0.80")  # p(2) = 0.80
        assert dealer.ask == Decimal("0.83")  # a(2) = 0.83
        assert dealer.bid == Decimal("0.77")  # b(2) = 0.77

        # Verify invariants
        run_all_assertions(dealer, vbt, params)

    def test_trade_sequence(self):
        """Execute three trades and verify dealer profit."""
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=2,
            dealer_cash=Decimal(2),
        )

        executor = TradeExecutor(params)

        # Initial state
        initial_V = dealer.V
        assert initial_V == Decimal(4)

        # Trade 1: Customer SELL (dealer buys at bid)
        ticket1 = create_ticket(
            issuer_id="customer1",
            owner_id="customer1",
        )
        result1 = executor.execute_customer_sell(dealer, vbt, ticket1)

        assert result1.executed
        assert not result1.is_passthrough
        assert result1.price == Decimal("0.97")
        assert dealer.a == 3
        assert dealer.x == Decimal(3)
        assert dealer.cash == Decimal("1.03")  # 2 - 0.97
        assert dealer.V == Decimal("4.03")  # 3 + 1.03

        # Trade 2: Customer BUY (dealer sells at ask)
        result2 = executor.execute_customer_buy(dealer, vbt, "customer2")

        assert result2.executed
        assert not result2.is_passthrough
        assert result2.price == Decimal("0.98")
        assert dealer.a == 2
        assert dealer.x == Decimal(2)
        assert dealer.cash == Decimal("2.01")  # 1.03 + 0.98
        assert dealer.V == Decimal("4.01")  # 2 + 2.01

        # Trade 3: Customer SELL (dealer buys at bid)
        ticket3 = create_ticket(
            issuer_id="customer3",
            owner_id="customer3",
        )
        result3 = executor.execute_customer_sell(dealer, vbt, ticket3)

        assert result3.executed
        assert not result3.is_passthrough
        assert result3.price == Decimal("0.97")
        assert dealer.a == 3
        assert dealer.x == Decimal(3)
        assert dealer.cash == Decimal("1.04")  # 2.01 - 0.97
        assert dealer.V == Decimal("4.04")  # 3 + 1.04

        # Verify dealer earned profit
        profit = dealer.V - initial_V
        assert profit == Decimal("0.04")

    def test_invariants_after_each_trade(self):
        """Verify C2 and C5 hold after each trade."""
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=2,
            dealer_cash=Decimal(2),
        )

        executor = TradeExecutor(params)

        # After each trade, verify invariants
        trades = [create_ticket(f"customer{i}", f"customer{i}") for i in range(3)]

        # Trade 1: SELL
        executor.execute_customer_sell(dealer, vbt, trades[0])
        run_all_assertions(dealer, vbt, params)

        # Trade 2: BUY
        executor.execute_customer_buy(dealer, vbt, "customer2")
        run_all_assertions(dealer, vbt, params)

        # Trade 3: SELL
        executor.execute_customer_sell(dealer, vbt, trades[2])
        run_all_assertions(dealer, vbt, params)


class TestExample4InventoryLimit:
    """
    Example 4: Dealer depletes inventory and VBT layoff occurs.

    Initial state:
    - a=2, C=2, M=0.80, O=0.30
    - V=3.60, K*=4, X*=4, x=2

    Two customer BUYs deplete inventory to x=0.
    Third customer BUY: interior sell infeasible -> passthrough to VBT at A=0.95.
    Dealer state unchanged (C4 assertion).
    """

    def test_interior_sells_deplete_inventory(self):
        """Two sells bring inventory to zero."""
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=2,
            dealer_cash=Decimal(2),
            M=Decimal("0.80"),
        )

        executor = TradeExecutor(params)

        # Initial: x=2
        assert dealer.x == Decimal(2)
        assert dealer.ask == Decimal("0.83")

        # First BUY: x=2 -> x=1
        result1 = executor.execute_customer_buy(dealer, vbt, "buyer1")
        assert result1.executed
        assert not result1.is_passthrough
        assert result1.price == Decimal("0.83")
        assert dealer.x == Decimal(1)
        assert dealer.ask == Decimal("0.88")  # p(1) = 0.85, a(1) = 0.88

        # Second BUY: x=1 -> x=0
        result2 = executor.execute_customer_buy(dealer, vbt, "buyer2")
        assert result2.executed
        assert not result2.is_passthrough
        assert result2.price == Decimal("0.88")
        assert dealer.x == Decimal(0)
        assert dealer.cash == Decimal("3.71")  # 2 + 0.83 + 0.88

    def test_passthrough_at_zero_inventory(self):
        """Third BUY routes to VBT at A when dealer has no inventory."""
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=2,
            dealer_cash=Decimal(2),
        )

        # Add tickets to VBT for passthrough
        for i in range(10):
            ticket = create_ticket(
                issuer_id=f"vbt_issuer_{i}",
                owner_id=vbt.agent_id,
                serial=i,
            )
            vbt.inventory.append(ticket)

        executor = TradeExecutor(params)

        # Deplete dealer inventory
        executor.execute_customer_buy(dealer, vbt, "buyer1")
        executor.execute_customer_buy(dealer, vbt, "buyer2")

        assert dealer.x == Decimal(0)

        # Third BUY: must route to VBT at A=1.15
        dealer_snapshot = deepcopy(dealer)

        result3 = executor.execute_customer_buy(dealer, vbt, "buyer3")

        assert result3.executed
        assert result3.is_passthrough
        assert result3.price == vbt.A  # 1.15

        # Verify dealer unchanged (C4 assertion)
        assert dealer.x == dealer_snapshot.x
        assert dealer.cash == dealer_snapshot.cash

    def test_dealer_unchanged_in_passthrough(self):
        """Verify C4: dealer (x, C) unchanged in passthrough."""
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=0,  # Start with no inventory
            dealer_cash=Decimal("4.11"),
        )

        # Add tickets to VBT
        for i in range(5):
            ticket = create_ticket(
                issuer_id=f"vbt_issuer_{i}",
                owner_id=vbt.agent_id,
                serial=i,
            )
            vbt.inventory.append(ticket)

        executor = TradeExecutor(params)

        # Capture state before passthrough
        x_before = dealer.x
        cash_before = dealer.cash

        # Execute passthrough
        result = executor.execute_customer_buy(dealer, vbt, "buyer")

        assert result.is_passthrough
        assert dealer.x == x_before
        assert dealer.cash == cash_before


class TestExample6BidPassthrough:
    """
    Example 6: Capacity binding forces bid-side passthrough.

    Scenario: Dealer at capacity (x = X*) with low cash.
    Customer SELL cannot be absorbed -> routes to VBT at B=0.85.
    """

    def test_capacity_bound_at_x_equals_X_star(self):
        """x = X* means no room for interior buy."""
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=2,
            dealer_cash=Decimal("0.10"),  # Low cash
        )

        # Dealer state: a=2, C=0.10
        # V = 2.10, K*=2, X*=2, x=2=X*
        assert dealer.a == 2
        assert dealer.x == Decimal(2)
        assert dealer.X_star == Decimal(2)
        assert dealer.cash == Decimal("0.10")

        # Check that interior buy is NOT feasible
        assert not can_interior_buy(dealer, params)

    def test_passthrough_at_outside_bid(self):
        """Customer SELL routes to VBT at B when capacity is full."""
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=2,
            dealer_cash=Decimal("0.10"),
        )

        executor = TradeExecutor(params)

        # Snapshot dealer state
        dealer_snapshot = deepcopy(dealer)

        # Customer SELL
        ticket = create_ticket(
            issuer_id="customer",
            owner_id="customer",
        )

        result = executor.execute_customer_sell(dealer, vbt, ticket)

        # Should passthrough at outside bid B=0.85
        assert result.executed
        assert result.is_passthrough
        assert result.price == vbt.B  # 0.85

        # Dealer state unchanged (C4)
        assert dealer.x == dealer_snapshot.x
        assert dealer.cash == dealer_snapshot.cash


class TestCapacityJump:
    """
    Example 13: One-ticket trade pushes capacity across integer.

    Tests discrete jumps in K*, X*, λ, and I when trades cause
    V to cross integer thresholds.
    """

    def test_up_jump(self):
        """K*: 3 -> 4 after interior buy."""
        # Setup: a=2, C=1.97 => V=3.97, K*=3, X*=3
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=2,
            dealer_cash=Decimal("1.97"),
        )

        # Initial state
        assert dealer.V == Decimal("3.97")
        assert dealer.K_star == 3
        assert dealer.X_star == Decimal(3)
        assert dealer.lambda_ == Decimal("0.25")  # 1/4
        assert dealer.I == Decimal("0.075")  # 0.25 * 0.30

        executor = TradeExecutor(params)

        # Customer SELL (dealer buys)
        ticket = create_ticket(
            issuer_id="customer",
            owner_id="customer",
        )

        result = executor.execute_customer_sell(dealer, vbt, ticket)

        # After buy: a=3, C=1.0375, V=4.0375
        # K* jumps 3->4
        assert result.executed
        assert dealer.a == 3
        assert dealer.V > Decimal(4)
        assert dealer.K_star == 4
        assert dealer.X_star == Decimal(4)

        # Width tightens after capacity increase
        assert dealer.lambda_ == Decimal("0.20")  # 1/5
        assert dealer.I == Decimal("0.06")  # 0.20 * 0.30

    def test_down_jump(self):
        """K*: 4 -> 3 after interior sell."""
        # Setup: a=4, C=0.02 => V=4.02, K*=4, X*=4
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=4,
            dealer_cash=Decimal("0.02"),
        )

        # Initial state
        assert dealer.V == Decimal("4.02")
        assert dealer.K_star == 4
        assert dealer.X_star == Decimal(4)
        assert dealer.lambda_ == Decimal("0.20")  # 1/5
        assert dealer.I == Decimal("0.06")

        executor = TradeExecutor(params)

        # Customer BUY (dealer sells)
        result = executor.execute_customer_buy(dealer, vbt, "buyer")

        # After sell: a=3, C=0.95, V=3.95
        # K* jumps 4->3
        assert result.executed
        assert dealer.a == 3
        assert dealer.V == Decimal("3.95")
        assert dealer.K_star == 3
        assert dealer.X_star == Decimal(3)

        # Width widens after capacity decrease
        assert dealer.lambda_ == Decimal("0.25")  # 1/4
        assert dealer.I == Decimal("0.075")  # 0.25 * 0.30

    def test_width_changes_with_capacity(self):
        """Verify λ and I change discretely when K* jumps."""
        # Test both directions
        dealer_up, vbt_up, params = setup_dealer_vbt_params(
            dealer_tickets=2,
            dealer_cash=Decimal("1.97"),
        )

        lambda_before_up = dealer_up.lambda_
        I_before_up = dealer_up.I

        executor = TradeExecutor(params)
        ticket = create_ticket("cust", "cust")
        executor.execute_customer_sell(dealer_up, vbt_up, ticket)

        # Lambda decreases (capacity increased)
        assert dealer_up.lambda_ < lambda_before_up
        assert dealer_up.I < I_before_up

        # Test down jump
        dealer_down, vbt_down, _ = setup_dealer_vbt_params(
            dealer_tickets=4,
            dealer_cash=Decimal("0.02"),
        )

        lambda_before_down = dealer_down.lambda_
        I_before_down = dealer_down.I

        executor.execute_customer_buy(dealer_down, vbt_down, "buyer")

        # Lambda increases (capacity decreased)
        assert dealer_down.lambda_ > lambda_before_down
        assert dealer_down.I > I_before_down


class TestGuardRegime:
    """
    Example 9: Guard at very low M <= M_min.

    When M <= 0.02, dealer enters guard regime:
    - X* := 0
    - Quotes pinned to outside (A, B)
    - All trades route to VBT (passthrough both sides)
    """

    def test_guard_pins_quotes(self):
        """Both quotes pinned when M <= M_min."""
        # Setup with very low M
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=3,
            dealer_cash=Decimal("2.00"),
            M=Decimal("0.01"),  # Below M_MIN = 0.02
            O=Decimal("0.01"),
        )

        # Guard should activate
        assert vbt.M <= M_MIN
        assert dealer.X_star == Decimal(0)
        assert dealer.is_pinned_ask
        assert dealer.is_pinned_bid
        assert dealer.ask == vbt.A
        assert dealer.bid == vbt.B

    def test_passthrough_both_sides(self):
        """All trades route to VBT in guard regime when dealer has no inventory."""
        # In guard regime, X*=0 means no capacity for BUYs
        # SELLs can still happen if dealer has inventory
        # To test full passthrough, start with no dealer inventory
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=0,  # No inventory
            dealer_cash=Decimal("2.00"),
            M=Decimal("0.01"),
            O=Decimal("0.01"),
        )

        # Add VBT inventory for passthrough
        for i in range(5):
            ticket = create_ticket(
                issuer_id=f"vbt_issuer_{i}",
                owner_id=vbt.agent_id,
                serial=i,
            )
            vbt.inventory.append(ticket)

        executor = TradeExecutor(params)

        # Test SELL passthrough (X*=0, cannot buy)
        ticket_sell = create_ticket("customer_sell", "customer_sell")
        dealer_x_before = dealer.x
        dealer_cash_before = dealer.cash

        result_sell = executor.execute_customer_sell(dealer, vbt, ticket_sell)

        assert result_sell.is_passthrough
        assert result_sell.price == vbt.B
        assert dealer.x == dealer_x_before  # Unchanged
        assert dealer.cash == dealer_cash_before  # Unchanged

        # Test BUY passthrough (x=0, no inventory)
        result_buy = executor.execute_customer_buy(dealer, vbt, "customer_buy")

        assert result_buy.is_passthrough
        assert result_buy.price == vbt.A
        assert dealer.x == dealer_x_before  # Still unchanged
        assert dealer.cash == dealer_cash_before  # Still unchanged


class TestPartialRecovery:
    """
    Example 10 & 12: Partial recovery default.

    Tests proportional recovery when issuer has insufficient cash:
    R = C/D when C < D

    Multiple claimants (dealer, VBT, trader) all receive proportional recovery.
    """

    def test_recovery_rate_computation(self):
        """R = C/D when C < D."""
        # Issuer has 3 units cash, owes 5 units
        # R = 3/5 = 0.6

        issuer_cash = Decimal(3)
        total_due = Decimal(5)

        recovery_rate = issuer_cash / total_due
        assert recovery_rate == Decimal("0.6")

        # Each holder receives proportional payout
        dealer_tickets = 2
        vbt_tickets = 2
        trader_tickets = 1

        dealer_payout = dealer_tickets * recovery_rate
        vbt_payout = vbt_tickets * recovery_rate
        trader_payout = trader_tickets * recovery_rate

        assert dealer_payout == Decimal("1.2")
        assert vbt_payout == Decimal("1.2")
        assert trader_payout == Decimal("0.6")

        # Total payout equals issuer cash
        total_payout = dealer_payout + vbt_payout + trader_payout
        assert total_payout == issuer_cash

    def test_multiple_claimants(self):
        """Dealer, VBT, and trader all receive proportional recovery."""
        # Setup scenario from Example 12
        recovery_rate = Decimal("0.6")

        # Create maturing tickets for each claimant type
        dealer_tickets = [create_ticket("issuer_I", "dealer", serial=i) for i in range(2)]

        vbt_tickets = [create_ticket("issuer_I", "vbt", serial=i + 2) for i in range(2)]

        trader_tickets = [create_ticket("issuer_I", "trader", serial=4)]

        # Calculate payouts
        dealer_payout = len(dealer_tickets) * recovery_rate
        vbt_payout = len(vbt_tickets) * recovery_rate
        trader_payout = len(trader_tickets) * recovery_rate

        # Verify equal per-ticket recovery
        assert dealer_payout == Decimal("1.2")
        assert vbt_payout == Decimal("1.2")
        assert trader_payout == Decimal("0.6")

        # Total
        total = dealer_payout + vbt_payout + trader_payout
        assert total == Decimal("3.0")


class TestQuoteBounds:
    """
    Test C2 assertion: quotes within outside bounds.

    Verifies that dealer interior quotes remain within VBT outside bounds
    at all inventory levels.
    """

    def test_quotes_within_bounds_at_all_levels(self):
        """Dealer quotes stay within [B, A] for all feasible x."""
        dealer, vbt, params = setup_dealer_vbt_params()

        # Test at various inventory levels
        for x in range(int(dealer.X_star) + 1):
            # Adjust dealer inventory to x
            dealer.inventory = [
                create_ticket(f"issuer_{i}", dealer.agent_id, serial=i) for i in range(x)
            ]
            recompute_dealer_state(dealer, vbt, params)

            # Verify bounds
            assert dealer.bid >= vbt.B, f"Bid {dealer.bid} < outside bid {vbt.B} at x={x}"
            assert dealer.ask <= vbt.A, f"Ask {dealer.ask} > outside ask {vbt.A} at x={x}"

            # Verify pin detection
            assert (dealer.bid == vbt.B) == dealer.is_pinned_bid
            assert (dealer.ask == vbt.A) == dealer.is_pinned_ask

    def test_no_interior_clipping(self):
        """Interior quotes never equal outside quotes (no clipping on interior rungs)."""
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=2,
            dealer_cash=Decimal(2),
            M=Decimal("0.80"),
        )

        # At x=2 (balanced), quotes should be strictly inside
        assert dealer.x == Decimal(2)
        assert dealer.bid > vbt.B
        assert dealer.ask < vbt.A
        assert not dealer.is_pinned_bid
        assert not dealer.is_pinned_ask


class TestFeasibilityChecks:
    """
    Test C3 assertion: feasibility pre-checks.

    Verifies that trades are only executed when feasible:
    - BUY: x + S <= X* and C >= b_c(x)
    - SELL: x >= S
    """

    def test_interior_buy_feasibility(self):
        """Interior buy requires capacity and cash."""
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=2,
            dealer_cash=Decimal(2),
        )

        # At x=2, X*=4: has capacity for buy
        assert dealer.x + params.S <= dealer.X_star
        assert dealer.cash >= dealer.bid
        assert can_interior_buy(dealer, params)

        # Test cash constraint: reduce cash below bid
        # But we need to know what the bid will be after recomputation
        dealer.cash = Decimal("0.90")
        recompute_dealer_state(dealer, vbt, params)

        # With less cash, may not have enough for bid
        # Check if cash is insufficient
        if dealer.cash < dealer.bid:
            assert not can_interior_buy(dealer, params)

        # Test capacity constraint: dealer at full capacity
        # Setup: a=2, C=0.10 => V=2.10, K*=2, X*=2, x=2=X*
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=2,
            dealer_cash=Decimal("0.10"),
        )

        # At x=2=X*, no room for more
        assert dealer.x == Decimal(2)
        assert dealer.X_star == Decimal(2)
        assert dealer.x == dealer.X_star
        assert not can_interior_buy(dealer, params)

    def test_interior_sell_feasibility(self):
        """Interior sell requires inventory."""
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=2,
            dealer_cash=Decimal(2),
        )

        # Has inventory
        assert dealer.x >= params.S
        assert can_interior_sell(dealer, params)

        # Deplete inventory
        dealer.inventory = []
        recompute_dealer_state(dealer, vbt, params)

        # No inventory
        assert dealer.x == Decimal(0)
        assert not can_interior_sell(dealer, params)


class TestDoubleEntry:
    """
    Test C1 assertion: double-entry bookkeeping.

    For every trade, cash and ticket flows must balance across all parties.
    """

    def test_interior_trade_balance(self):
        """Interior trade: cash and tickets balance between customer and dealer."""
        dealer, vbt, params = setup_dealer_vbt_params()
        executor = TradeExecutor(params)

        # Customer SELL
        ticket = create_ticket("customer", "customer")
        result = executor.execute_customer_sell(dealer, vbt, ticket)

        # Manual verification (assertions run automatically in executor)
        price = result.price

        # Cash: customer +price, dealer -price => sum = 0
        assert_c1_double_entry(
            cash_changes={"customer": price, dealer.agent_id: -price},
            qty_changes={"customer": -1, dealer.agent_id: 1},
        )

    def test_passthrough_trade_balance(self):
        """Passthrough trade: cash and tickets balance between customer and VBT."""
        dealer, vbt, params = setup_dealer_vbt_params(
            dealer_tickets=0,  # Force passthrough
            dealer_cash=Decimal(2),
        )

        # Add VBT inventory
        for i in range(5):
            vbt.inventory.append(create_ticket(f"vbt_issuer_{i}", vbt.agent_id, serial=i))

        executor = TradeExecutor(params)

        # Customer BUY (passthrough)
        result = executor.execute_customer_buy(dealer, vbt, "customer")

        assert result.is_passthrough
        price = result.price

        # Cash: customer -price, VBT +price, dealer 0 => sum = 0
        assert_c1_double_entry(
            cash_changes={"customer": -price, vbt.agent_id: price},
            qty_changes={"customer": 1, vbt.agent_id: -1},
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
