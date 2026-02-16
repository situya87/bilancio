"""
Metrics data structures and computation for functional dealer analysis.

This module implements the measurement framework defined in Section 8 of the
"New Kalecki Ring with Dealers" specification. It provides comprehensive
metrics to evaluate whether the dealer satisfies functional dealer criteria.

Key metrics tracked:
1. Trade log with pre/post state (8.1)
2. Dealer P&L and profitability (8.2)
3. Trader investment returns and liquidity use (8.3)
4. Repayment-priority diagnostics (8.4)

References:
    - Section 8: "Measurement and data collection for functional dealer properties"
    - D1-D3: Functional dealer and trader behaviour definitions
    - R1-R6: Requirements for simulation analysis
"""

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any


@dataclass
class TradeRecord:
    """
    Record of a single dealer trade with full state information.

    Implements Section 8.1 trade log requirements:
    - time t, bucket b, side, agent i, price p, issuer, maturity

    Extended with pre/post state for diagnostics:
    - Dealer state before/after
    - Trader safety margin before/after (Section 8.4)
    - Liquidity-driven flag (Section 8.3)

    Plan 022 extensions:
    - run_id/regime for experiment tracking
    - hit_inventory_limit flag for VBT routing analysis
    """

    # Core trade identifiers
    day: int
    bucket: str
    side: str  # "BUY" or "SELL" (from customer perspective)
    trader_id: str
    ticket_id: str

    # Ticket details
    issuer_id: str
    maturity_day: int
    face_value: Decimal
    price: Decimal
    unit_price: Decimal  # price / face_value
    is_passthrough: bool

    # === Fields with defaults below ===

    # Run context (Plan 022 - Phase 1)
    run_id: str = ""  # e.g., "grid_abc123"
    regime: str = ""  # "passive" or "active"

    # Pre-trade state
    dealer_inventory_before: int = 0
    dealer_cash_before: Decimal = Decimal(0)
    dealer_bid_before: Decimal = Decimal(0)
    dealer_ask_before: Decimal = Decimal(0)
    vbt_mid_before: Decimal = Decimal(0)
    trader_cash_before: Decimal = Decimal(0)
    trader_safety_margin_before: Decimal = Decimal(0)

    # Post-trade state
    dealer_inventory_after: int = 0
    dealer_cash_after: Decimal = Decimal(0)
    dealer_bid_after: Decimal = Decimal(0)
    dealer_ask_after: Decimal = Decimal(0)
    trader_cash_after: Decimal = Decimal(0)
    trader_safety_margin_after: Decimal = Decimal(0)

    # Flags
    is_liquidity_driven: bool = False  # Seller had shortfall (Section 8.3)
    reduces_margin_below_zero: bool = False  # BUY reduced margin < 0 (Section 8.4)

    # VBT routing analysis (Plan 022 - Phase 1)
    # hit_inventory_limit is different from is_passthrough:
    # - is_passthrough=True means trade went to VBT
    # - hit_inventory_limit=True means VBT was used *because* dealer was at capacity
    #   (side=SELL and dealer at max_capacity, or side=BUY and dealer at 0 inventory)
    hit_inventory_limit: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with serialized Decimals."""
        return {
            # Run context (Plan 022)
            "run_id": self.run_id,
            "regime": self.regime,
            # Core trade identifiers
            "day": self.day,
            "bucket": self.bucket,
            "side": self.side,
            "trader_id": self.trader_id,
            "ticket_id": self.ticket_id,
            "issuer_id": self.issuer_id,
            "maturity_day": self.maturity_day,
            "face_value": str(self.face_value),
            "price": str(self.price),
            "unit_price": str(self.unit_price),
            "is_passthrough": self.is_passthrough,
            "dealer_inventory_before": self.dealer_inventory_before,
            "dealer_cash_before": str(self.dealer_cash_before),
            "dealer_bid_before": str(self.dealer_bid_before),
            "dealer_ask_before": str(self.dealer_ask_before),
            "vbt_mid_before": str(self.vbt_mid_before),
            "trader_cash_before": str(self.trader_cash_before),
            "trader_safety_margin_before": str(self.trader_safety_margin_before),
            "dealer_inventory_after": self.dealer_inventory_after,
            "dealer_cash_after": str(self.dealer_cash_after),
            "dealer_bid_after": str(self.dealer_bid_after),
            "dealer_ask_after": str(self.dealer_ask_after),
            "trader_cash_after": str(self.trader_cash_after),
            "trader_safety_margin_after": str(self.trader_safety_margin_after),
            "is_liquidity_driven": self.is_liquidity_driven,
            "reduces_margin_below_zero": self.reduces_margin_below_zero,
            # VBT routing analysis (Plan 022)
            "hit_inventory_limit": self.hit_inventory_limit,
        }


@dataclass
class DealerSnapshot:
    """
    Snapshot of dealer state at a point in time.

    Implements Section 8.1 dealer state requirements:
    - Inventory a_t^(b), cash C_t^(b)
    - Outside mid M_t^(b), spread O_t^(b)
    - Bid/ask quotes
    - Mark-to-mid equity E_t^(b) = C_t^(b) + M_t^(b) * a_t^(b) * S

    Plan 022 extensions (Phase 3 - inventory timeseries):
    - max_capacity, capacity_pct for utilization tracking
    - is_at_zero, hit_vbt_this_step for VBT routing analysis
    - total_system_face, dealer_share_pct for system-level context
    """

    day: int
    bucket: str
    inventory: int  # a_t^(b) - number of tickets
    cash: Decimal
    bid: Decimal
    ask: Decimal
    midline: Decimal
    vbt_mid: Decimal  # M_t^(b)
    vbt_spread: Decimal  # O_t^(b)
    ticket_size: Decimal  # S

    # Plan 022 - Phase 3: Inventory timeseries fields
    run_id: str = ""  # Run identifier for tracking
    regime: str = ""  # "passive" or "active"
    max_capacity: int = 0  # K* - max dealer inventory capacity
    is_at_zero: bool = False  # inventory == 0?
    hit_vbt_this_step: bool = False  # Did we route to VBT this step?
    total_system_face: Decimal = Decimal(0)  # Total face value in system for this bucket
    dealer_share_pct: Decimal = Decimal(0)  # % of bucket's face held by dealer

    @property
    def mark_to_mid_equity(self) -> Decimal:
        """
        Mark-to-mid equity: E_t^(b) = C_t^(b) + M_t^(b) * a_t^(b) * S

        This is the equity valuation using VBT mid price.
        """
        return self.cash + self.vbt_mid * self.inventory * self.ticket_size

    @property
    def capacity_pct(self) -> Decimal:
        """
        Capacity utilization as percentage (Plan 022).

        inventory / max_capacity * 100, or 0 if max_capacity is 0.
        """
        if self.max_capacity == 0:
            return Decimal(0)
        return (Decimal(self.inventory) / Decimal(self.max_capacity)) * 100

    @property
    def spread(self) -> Decimal:
        """Current bid-ask spread."""
        return self.ask - self.bid

    @property
    def dealer_premium_vs_face(self) -> Decimal:
        """
        Dealer midline premium/discount vs face value (par = 1).

        Returns (midline - 1).
        Positive = premium, Negative = discount.
        """
        return self.midline - Decimal(1)

    @property
    def dealer_premium_pct(self) -> Decimal:
        """
        Dealer premium/discount as percentage.

        ((midline - 1) / 1) * 100
        """
        return (self.midline - Decimal(1)) * 100

    @property
    def vbt_premium_vs_face(self) -> Decimal:
        """
        VBT mid premium/discount vs face value (par = 1).

        Returns (M - 1).
        """
        return self.vbt_mid - Decimal(1)

    @property
    def vbt_premium_pct(self) -> Decimal:
        """
        VBT premium/discount as percentage.

        ((M - 1) / 1) * 100
        """
        return (self.vbt_mid - Decimal(1)) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with serialized Decimals."""
        return {
            # Run context (Plan 022)
            "run_id": self.run_id,
            "regime": self.regime,
            # Core fields
            "day": self.day,
            "bucket": self.bucket,
            "inventory": self.inventory,
            "cash": str(self.cash),
            "bid": str(self.bid),
            "ask": str(self.ask),
            "midline": str(self.midline),
            "vbt_mid": str(self.vbt_mid),
            "vbt_spread": str(self.vbt_spread),
            "ticket_size": str(self.ticket_size),
            "mark_to_mid_equity": str(self.mark_to_mid_equity),
            "spread": str(self.spread),
            "dealer_premium_pct": str(self.dealer_premium_pct),
            "vbt_premium_pct": str(self.vbt_premium_pct),
            # Plan 022 - Phase 3: Inventory timeseries
            "max_capacity": self.max_capacity,
            "capacity_pct": str(self.capacity_pct),
            "is_at_zero": self.is_at_zero,
            "hit_vbt_this_step": self.hit_vbt_this_step,
            "total_system_face": str(self.total_system_face),
            "dealer_share_pct": str(self.dealer_share_pct),
        }


@dataclass
class TraderSnapshot:
    """
    Snapshot of trader state at a point in time.

    Implements Section 8.1 trader state requirements:
    - Cash C_i(t)
    - Set of tickets held
    - Schedule of future obligations
    - Safety margin m_i(t) (Section 8.4)
    """

    day: int
    trader_id: str
    cash: Decimal
    tickets_held_count: int
    tickets_held_ids: list[str]
    total_face_held: Decimal
    obligations_remaining: Decimal  # D_i(t) - future obligations
    saleable_value: Decimal  # Value of tickets at dealer bid prices
    safety_margin: Decimal  # m_i(t) = A_i(t) - D_i(t)
    defaulted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with serialized Decimals."""
        return {
            "day": self.day,
            "trader_id": self.trader_id,
            "cash": str(self.cash),
            "tickets_held_count": self.tickets_held_count,
            "tickets_held_ids": self.tickets_held_ids,
            "total_face_held": str(self.total_face_held),
            "obligations_remaining": str(self.obligations_remaining),
            "saleable_value": str(self.saleable_value),
            "safety_margin": str(self.safety_margin),
            "defaulted": self.defaulted,
        }


@dataclass
class TicketOutcome:
    """
    Final outcome of a ticket for return calculation.

    Implements Section 8.3 ticket-level realized return:
    R_τ = (X_τ - p_buy) / p_buy

    Where X_τ is:
    - Settlement amount at maturity (if held)
    - Resale price if sold to dealer
    - Coupon payments (future extension)
    """

    ticket_id: str
    issuer_id: str
    maturity_day: int
    face_value: Decimal

    # Purchase from dealer (if applicable)
    purchased_from_dealer: bool = False
    purchase_day: int | None = None
    purchase_price: Decimal | None = None
    purchaser_id: str | None = None

    # Sale to dealer (if applicable)
    sold_to_dealer: bool = False
    sale_day: int | None = None
    sale_price: Decimal | None = None
    seller_id: str | None = None

    # Settlement outcome
    settled: bool = False
    settlement_day: int | None = None
    recovery_rate: Decimal | None = None
    settlement_amount: Decimal | None = None

    def realized_return(self) -> Decimal | None:
        """
        Calculate realized return R_τ for ticket purchased from dealer.

        Returns:
            R_τ = (X_τ - p_buy) / p_buy, or None if not purchased from dealer
        """
        if not self.purchased_from_dealer or self.purchase_price is None:
            return None

        p_buy = self.purchase_price

        # Determine total payoff X_τ
        x_tau = Decimal(0)

        if self.sold_to_dealer and self.sale_price is not None:
            # Resold to dealer
            x_tau = self.sale_price
        elif self.settled and self.settlement_amount is not None:
            # Held to maturity
            x_tau = self.settlement_amount
        else:
            # Still held or defaulted with no recovery
            return None

        if p_buy == 0:
            return None

        return (x_tau - p_buy) / p_buy

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with serialized Decimals."""
        return {
            "ticket_id": self.ticket_id,
            "issuer_id": self.issuer_id,
            "maturity_day": self.maturity_day,
            "face_value": str(self.face_value),
            "purchased_from_dealer": self.purchased_from_dealer,
            "purchase_day": self.purchase_day,
            "purchase_price": str(self.purchase_price) if self.purchase_price else None,
            "purchaser_id": self.purchaser_id,
            "sold_to_dealer": self.sold_to_dealer,
            "sale_day": self.sale_day,
            "sale_price": str(self.sale_price) if self.sale_price else None,
            "seller_id": self.seller_id,
            "settled": self.settled,
            "settlement_day": self.settlement_day,
            "recovery_rate": str(self.recovery_rate) if self.recovery_rate else None,
            "settlement_amount": str(self.settlement_amount) if self.settlement_amount else None,
            "realized_return": str(self.realized_return())
            if self.realized_return() is not None
            else None,
        }


@dataclass
class SystemStateSnapshot:
    """
    System-level state at a point in time (Plan 022 - Phase 4).

    Tracks how total debt/money evolves over time to understand
    the "winding down" dynamics of the system.

    Fields:
    - run_id, regime: Run context for experiment tracking
    - day: Simulation day
    - total_face_value: Sum of all outstanding ticket face values
    - face_by_bucket: Face value breakdown by maturity bucket
    - total_cash: Sum of all agent cash holdings
    - debt_to_money: Current debt/money ratio
    """

    run_id: str
    regime: str
    day: int

    # Total face value outstanding
    total_face_value: Decimal  # All maturities
    face_bucket_short: Decimal = Decimal(0)
    face_bucket_mid: Decimal = Decimal(0)
    face_bucket_long: Decimal = Decimal(0)

    # Total cash in system
    total_cash: Decimal = Decimal(0)

    @property
    def debt_to_money(self) -> Decimal:
        """Current debt-to-money ratio."""
        if self.total_cash > 0:
            return self.total_face_value / self.total_cash
        return Decimal(0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with serialized Decimals."""
        return {
            "run_id": self.run_id,
            "regime": self.regime,
            "day": self.day,
            "total_face_value": str(self.total_face_value),
            "face_bucket_short": str(self.face_bucket_short),
            "face_bucket_mid": str(self.face_bucket_mid),
            "face_bucket_long": str(self.face_bucket_long),
            "total_cash": str(self.total_cash),
            "debt_to_money": str(self.debt_to_money),
        }


@dataclass
class RepaymentEvent:
    """
    Track how a trader managed a specific liability (Plan 022 - Phase 2).

    For each (trader, liability) pair, records:
    - Outcome: did they repay or default?
    - Trading activity: buys/sells before maturity
    - Strategy classification: no_trade, hold_to_maturity, sell_before, round_trip

    This enables analysis of whether trading helped traders avoid defaults.
    """

    run_id: str
    regime: str
    trader_id: str
    liability_id: str  # The contract/ticket ID representing their debt
    maturity_day: int
    face_value: Decimal

    # Outcome
    outcome: str  # "repaid" or "defaulted"

    # Trading activity BEFORE this maturity
    buy_count: int = 0
    sell_count: int = 0
    net_cash_pnl: Decimal = Decimal(0)  # Net cash from trading (sells - buys)

    # Strategy classification
    strategy: str = ""  # "no_trade", "hold_to_maturity", "sell_before", "round_trip"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with serialized Decimals."""
        return {
            "run_id": self.run_id,
            "regime": self.regime,
            "trader_id": self.trader_id,
            "liability_id": self.liability_id,
            "maturity_day": self.maturity_day,
            "face_value": str(self.face_value),
            "outcome": self.outcome,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "net_cash_pnl": str(self.net_cash_pnl),
            "strategy": self.strategy,
        }


def classify_trading_strategy(buy_count: int, sell_count: int) -> str:
    """
    Classify a trader's strategy based on their trading activity.

    Returns:
        - "no_trade": Never traded with the dealer
        - "hold_to_maturity": Bought but never sold (holding for maturity payment)
        - "sell_before": Sold tickets to get cash (liquidity-seeking)
        - "round_trip": Both bought and sold (active trading)
    """
    if buy_count == 0 and sell_count == 0:
        return "no_trade"
    elif buy_count > 0 and sell_count > 0:
        return "round_trip"
    elif sell_count > 0:
        return "sell_before"
    else:  # buy_count > 0, sell_count == 0
        return "hold_to_maturity"


@dataclass
class RunMetrics:
    """
    All metrics for a single simulation run.

    This is the main container for Section 8 metrics, aggregating:
    - Trade log (8.1)
    - Daily dealer/trader snapshots (8.1)
    - Ticket outcomes for return calculation (8.3)

    Provides computed metrics for:
    - Dealer P&L and profitability (8.2)
    - Trader investment returns (8.3)
    - Repayment-priority diagnostics (8.4)
    """

    # Trade log (Section 8.1)
    trades: list[TradeRecord] = field(default_factory=list)

    # Daily snapshots (Section 8.1)
    dealer_snapshots: list[DealerSnapshot] = field(default_factory=list)
    trader_snapshots: list[TraderSnapshot] = field(default_factory=list)

    # System state timeseries (Plan 022 - Phase 4)
    system_state_snapshots: list[SystemStateSnapshot] = field(default_factory=list)

    # Repayment events (Plan 022 - Phase 2)
    repayment_events: list[RepaymentEvent] = field(default_factory=list)

    # Ticket outcomes (Section 8.3)
    ticket_outcomes: dict[str, TicketOutcome] = field(default_factory=dict)

    # Initial equity by bucket (for P&L calculation)
    initial_equity_by_bucket: dict[str, Decimal] = field(default_factory=dict)

    # System-level initial state (debt-to-money ratio control variable)
    initial_total_debt: Decimal = Decimal(0)  # Sum of all payable amounts at t=0
    initial_total_money: Decimal = Decimal(0)  # Sum of all cash holdings at t=0

    # Run context (Plan 022 - for tracking)
    run_id: str = ""
    regime: str = ""

    @property
    def debt_to_money_ratio(self) -> Decimal:
        """
        Debt-to-money ratio at simulation start.

        This is a key control variable: results only make sense
        given the same ratio. More money = fewer expected defaults.

        Returns:
            total_debt / total_money (or 0 if no money)
        """
        if self.initial_total_money > 0:
            return self.initial_total_debt / self.initial_total_money
        return Decimal(0)

    # =========================================================================
    # Section 8.2: Dealer and VBT Profitability
    # =========================================================================

    def dealer_pnl_by_bucket(self) -> dict[str, Decimal]:
        """
        Compute dealer P&L by bucket: Π_r^(b) = E_T^(b) - E_0^(b)

        Returns:
            Dictionary mapping bucket_id to P&L in currency units
        """
        pnl = {}
        for bucket_id, initial_equity in self.initial_equity_by_bucket.items():
            # Find final equity for this bucket
            final_snapshots = [s for s in self.dealer_snapshots if s.bucket == bucket_id]
            if final_snapshots:
                # Take the last snapshot
                final_equity = max(final_snapshots, key=lambda s: s.day).mark_to_mid_equity
                pnl[bucket_id] = final_equity - initial_equity
            else:
                pnl[bucket_id] = Decimal(0)
        return pnl

    def dealer_return_by_bucket(self) -> dict[str, Decimal]:
        """
        Compute dealer return by bucket: π_r^(b) = Π_r^(b) / E_0^(b)

        Returns:
            Dictionary mapping bucket_id to return (as fraction)
        """
        pnl = self.dealer_pnl_by_bucket()
        returns = {}
        for bucket_id, profit in pnl.items():
            initial = self.initial_equity_by_bucket.get(bucket_id, Decimal(0))
            if initial > 0:
                returns[bucket_id] = profit / initial
            else:
                returns[bucket_id] = Decimal(0)
        return returns

    def total_dealer_pnl(self) -> Decimal:
        """Total P&L across all buckets."""
        return sum(self.dealer_pnl_by_bucket().values(), Decimal(0))

    def total_dealer_return(self) -> Decimal:
        """
        Total dealer return weighted by initial equity.

        Returns:
            Π_total / E_0_total
        """
        total_pnl = self.total_dealer_pnl()
        total_initial = sum(self.initial_equity_by_bucket.values())
        if total_initial > 0:
            return total_pnl / total_initial
        return Decimal(0)

    def is_dealer_profitable(self) -> bool:
        """Check if dealer is profitable (Π >= 0)."""
        return self.total_dealer_pnl() >= 0

    def spread_income_total(self) -> Decimal:
        """
        Total spread income from interior trades.

        For SELL: income = dealer_ask - dealer_bid (customer gets bid, dealer marks at mid)
        For BUY: income = dealer_ask - dealer_bid (customer pays ask)

        Simplified: count trades where dealer made the spread.
        """
        income = Decimal(0)
        for trade in self.trades:
            if not trade.is_passthrough:
                # Interior trade - dealer earns spread
                if trade.side == "SELL":
                    # Customer sold to dealer at bid
                    # Spread income = midline - bid
                    midline = (trade.dealer_bid_before + trade.dealer_ask_before) / 2
                    income += (midline - trade.dealer_bid_before) * trade.face_value
                else:  # BUY
                    # Customer bought from dealer at ask
                    # Spread income = ask - midline
                    midline = (trade.dealer_bid_before + trade.dealer_ask_before) / 2
                    income += (trade.dealer_ask_before - midline) * trade.face_value
        return income

    def passthrough_count(self) -> int:
        """Count of passthrough trades (dealer to VBT)."""
        return sum(1 for t in self.trades if t.is_passthrough)

    def interior_count(self) -> int:
        """Count of interior trades (dealer only)."""
        return sum(1 for t in self.trades if not t.is_passthrough)

    # =========================================================================
    # Section 8.3: Trader Investment Returns and Liquidity Use
    # =========================================================================

    def trader_returns(self) -> dict[str, Decimal]:
        """
        Compute investment return R_i for each trader.

        R_i = (1/|T_i|) * Σ R_τ for all tickets τ purchased by trader i

        Returns:
            Dictionary mapping trader_id to mean return
        """
        # Group outcomes by purchaser
        returns_by_trader: dict[str, list[Decimal]] = {}

        for outcome in self.ticket_outcomes.values():
            if outcome.purchased_from_dealer and outcome.purchaser_id:
                r_tau = outcome.realized_return()
                if r_tau is not None:
                    if outcome.purchaser_id not in returns_by_trader:
                        returns_by_trader[outcome.purchaser_id] = []
                    returns_by_trader[outcome.purchaser_id].append(r_tau)

        # Compute mean return per trader
        trader_returns: dict[str, Decimal] = {}
        for trader_id, returns in returns_by_trader.items():
            if returns:
                trader_returns[trader_id] = sum(returns, Decimal(0)) / len(returns)
            else:
                trader_returns[trader_id] = Decimal(0)

        return trader_returns

    def mean_trader_return(self) -> Decimal:
        """Mean return across all traders who bought from dealer."""
        returns = list(self.trader_returns().values())
        if returns:
            return sum(returns, Decimal(0)) / len(returns)
        return Decimal(0)

    def fraction_profitable_traders(self) -> Decimal:
        """Fraction of traders with positive return (R_i > 0)."""
        returns = self.trader_returns()
        if not returns:
            return Decimal(0)
        profitable = sum(1 for r in returns.values() if r > 0)
        return Decimal(profitable) / len(returns)

    def liquidity_driven_sales(self) -> int:
        """
        Count of liquidity-driven sales.

        A sale is liquidity-driven when seller had shortfall in obligations.
        """
        return sum(1 for t in self.trades if t.side == "SELL" and t.is_liquidity_driven)

    def rescue_events(self) -> int:
        """
        Count of rescue events.

        A rescue event occurs when a liquidity-driven sale enables the seller
        to meet an obligation that would otherwise have defaulted.

        Note: This is approximated by counting liquidity-driven sales where
        the trader's safety margin improved to positive after the trade.
        """
        rescues = 0
        for trade in self.trades:
            if trade.side == "SELL" and trade.is_liquidity_driven:
                # Check if trade moved margin from negative to positive
                if trade.trader_safety_margin_before < 0 and trade.trader_safety_margin_after >= 0:
                    rescues += 1
        return rescues

    # =========================================================================
    # Section 8.4: Repayment-Priority Diagnostics
    # =========================================================================

    def unsafe_buy_count(self) -> int:
        """Count of BUY trades that reduced safety margin below zero."""
        return sum(1 for t in self.trades if t.reduces_margin_below_zero)

    def fraction_unsafe_buys(self) -> Decimal:
        """Fraction of BUY trades that reduced margin below zero."""
        buys = [t for t in self.trades if t.side == "BUY"]
        if not buys:
            return Decimal(0)
        unsafe = sum(1 for t in buys if t.reduces_margin_below_zero)
        return Decimal(unsafe) / len(buys)

    def margin_at_default_distribution(self) -> list[Decimal]:
        """
        Distribution of safety margins at time of default.

        Returns list of m_i(t) values for agents that defaulted.
        """
        margins = []
        for snapshot in self.trader_snapshots:
            if snapshot.defaulted:
                margins.append(snapshot.safety_margin)
        return margins

    def mean_margin_at_default(self) -> Decimal | None:
        """Mean safety margin at default time."""
        margins = self.margin_at_default_distribution()
        if margins:
            return sum(margins, Decimal(0)) / len(margins)
        return None

    # =========================================================================
    # Mid Price Time Series (Plan 020)
    # =========================================================================

    def dealer_mid_timeseries(self) -> dict[int, dict[str, Decimal]]:
        """
        Extract dealer midline time series by bucket.

        Returns:
            {day: {bucket_id: midline_value}}
        """
        result: dict[int, dict[str, Decimal]] = {}
        for snap in self.dealer_snapshots:
            if snap.day not in result:
                result[snap.day] = {}
            result[snap.day][snap.bucket] = snap.midline
        return result

    def vbt_mid_timeseries(self) -> dict[int, dict[str, Decimal]]:
        """
        Extract VBT mid (M) time series by bucket.

        Returns:
            {day: {bucket_id: M_value}}
        """
        result: dict[int, dict[str, Decimal]] = {}
        for snap in self.dealer_snapshots:
            if snap.day not in result:
                result[snap.day] = {}
            result[snap.day][snap.bucket] = snap.vbt_mid
        return result

    def dealer_premium_timeseries(self) -> dict[int, dict[str, Decimal]]:
        """
        Extract dealer premium/discount vs face time series.

        Returns:
            {day: {bucket_id: premium_pct}}
        """
        result: dict[int, dict[str, Decimal]] = {}
        for snap in self.dealer_snapshots:
            if snap.day not in result:
                result[snap.day] = {}
            result[snap.day][snap.bucket] = snap.dealer_premium_pct
        return result

    def vbt_premium_timeseries(self) -> dict[int, dict[str, Decimal]]:
        """
        Extract VBT premium/discount vs face time series.

        Returns:
            {day: {bucket_id: premium_pct}}
        """
        result: dict[int, dict[str, Decimal]] = {}
        for snap in self.dealer_snapshots:
            if snap.day not in result:
                result[snap.day] = {}
            result[snap.day][snap.bucket] = snap.vbt_premium_pct
        return result

    def _final_dealer_mids(self) -> dict[str, float]:
        """Get final dealer midline by bucket."""
        result = {}
        for bucket_id in {s.bucket for s in self.dealer_snapshots}:
            bucket_snaps = [s for s in self.dealer_snapshots if s.bucket == bucket_id]
            if bucket_snaps:
                final = max(bucket_snaps, key=lambda s: s.day)
                result[bucket_id] = float(final.midline)
        return result

    def _final_vbt_mids(self) -> dict[str, float]:
        """Get final VBT mid by bucket."""
        result = {}
        for bucket_id in {s.bucket for s in self.dealer_snapshots}:
            bucket_snaps = [s for s in self.dealer_snapshots if s.bucket == bucket_id]
            if bucket_snaps:
                final = max(bucket_snaps, key=lambda s: s.day)
                result[bucket_id] = float(final.vbt_mid)
        return result

    def _final_dealer_premiums(self) -> dict[str, float]:
        """Get final dealer premium % by bucket."""
        result = {}
        for bucket_id in {s.bucket for s in self.dealer_snapshots}:
            bucket_snaps = [s for s in self.dealer_snapshots if s.bucket == bucket_id]
            if bucket_snaps:
                final = max(bucket_snaps, key=lambda s: s.day)
                result[bucket_id] = float(final.dealer_premium_pct)
        return result

    def _final_vbt_premiums(self) -> dict[str, float]:
        """Get final VBT premium % by bucket."""
        result = {}
        for bucket_id in {s.bucket for s in self.dealer_snapshots}:
            bucket_snaps = [s for s in self.dealer_snapshots if s.bucket == bucket_id]
            if bucket_snaps:
                final = max(bucket_snaps, key=lambda s: s.day)
                result[bucket_id] = float(final.vbt_premium_pct)
        return result

    # =========================================================================
    # Section 8.5: Experiment-Level Summary Statistics
    # =========================================================================

    def summary(self) -> dict[str, Any]:
        """
        Generate summary statistics for experiment-level aggregation.

        Returns dictionary suitable for comparison across runs.
        """
        return {
            # Dealer profitability (8.2)
            "dealer_total_pnl": float(self.total_dealer_pnl()),
            "dealer_total_return": float(self.total_dealer_return()),
            "dealer_profitable": self.is_dealer_profitable(),
            "dealer_pnl_by_bucket": {k: float(v) for k, v in self.dealer_pnl_by_bucket().items()},
            "dealer_return_by_bucket": {
                k: float(v) for k, v in self.dealer_return_by_bucket().items()
            },
            "spread_income_total": float(self.spread_income_total()),
            "interior_trades": self.interior_count(),
            "passthrough_trades": self.passthrough_count(),
            # Trader returns (8.3)
            "mean_trader_return": float(self.mean_trader_return()),
            "fraction_profitable_traders": float(self.fraction_profitable_traders()),
            "liquidity_driven_sales": self.liquidity_driven_sales(),
            "rescue_events": self.rescue_events(),
            # Repayment priority (8.4)
            "unsafe_buy_count": self.unsafe_buy_count(),
            "fraction_unsafe_buys": float(self.fraction_unsafe_buys()),
            "mean_margin_at_default": (lambda v: float(v) if v is not None else None)(
                self.mean_margin_at_default()
            ),
            # Trade counts
            "total_trades": len(self.trades),
            "total_sell_trades": sum(1 for t in self.trades if t.side == "SELL"),
            "total_buy_trades": sum(1 for t in self.trades if t.side == "BUY"),
            # System-level control variable (Plan 020 - Phase B)
            "initial_total_debt": float(self.initial_total_debt),
            "initial_total_money": float(self.initial_total_money),
            "debt_to_money_ratio": float(self.debt_to_money_ratio),
            # Mid price summary stats (Plan 020 - Phase A.2, A.3)
            "dealer_mid_final": self._final_dealer_mids(),
            "vbt_mid_final": self._final_vbt_mids(),
            "dealer_premium_final_pct": self._final_dealer_premiums(),
            "vbt_premium_final_pct": self._final_vbt_premiums(),
        }

    # =========================================================================
    # Export Methods
    # =========================================================================

    def to_trade_log_csv(self, path: str) -> None:
        """Export trade log to CSV file."""
        import csv

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if not self.trades:
            return

        with open(path_obj, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.trades[0].to_dict().keys())
            writer.writeheader()
            for trade in self.trades:
                writer.writerow(trade.to_dict())

    def to_dealer_snapshots_csv(self, path: str) -> None:
        """Export dealer snapshots to CSV file."""
        import csv

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if not self.dealer_snapshots:
            return

        with open(path_obj, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.dealer_snapshots[0].to_dict().keys())
            writer.writeheader()
            for snapshot in self.dealer_snapshots:
                writer.writerow(snapshot.to_dict())

    def to_trader_snapshots_csv(self, path: str) -> None:
        """Export trader snapshots to CSV file."""
        import csv

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if not self.trader_snapshots:
            return

        # Filter out tickets_held_ids for CSV (too verbose)
        fieldnames = [
            k for k in self.trader_snapshots[0].to_dict().keys() if k != "tickets_held_ids"
        ]

        with open(path_obj, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for snapshot in self.trader_snapshots:
                row = {k: v for k, v in snapshot.to_dict().items() if k != "tickets_held_ids"}
                writer.writerow(row)

    def to_ticket_outcomes_csv(self, path: str) -> None:
        """Export ticket outcomes to CSV file."""
        import csv

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if not self.ticket_outcomes:
            return

        outcomes = list(self.ticket_outcomes.values())
        with open(path_obj, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=outcomes[0].to_dict().keys())
            writer.writeheader()
            for outcome in outcomes:
                writer.writerow(outcome.to_dict())

    def to_summary_json(self, path: str) -> None:
        """Export summary statistics to JSON file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w") as f:
            json.dump(self.summary(), f, indent=2)

    def to_mid_timeseries_csv(self, path: str) -> None:
        """
        Export mid price time series to CSV.

        Columns: day, bucket, dealer_midline, vbt_mid, dealer_premium_pct, vbt_premium_pct
        """
        import csv

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if not self.dealer_snapshots:
            return

        fieldnames = [
            "day",
            "bucket",
            "dealer_midline",
            "vbt_mid",
            "dealer_premium_pct",
            "vbt_premium_pct",
        ]

        with open(path_obj, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for snap in sorted(self.dealer_snapshots, key=lambda s: (s.day, s.bucket)):
                writer.writerow(
                    {
                        "day": snap.day,
                        "bucket": snap.bucket,
                        "dealer_midline": str(snap.midline),
                        "vbt_mid": str(snap.vbt_mid),
                        "dealer_premium_pct": str(snap.dealer_premium_pct),
                        "vbt_premium_pct": str(snap.vbt_premium_pct),
                    }
                )

    def to_inventory_timeseries_csv(self, path: str) -> None:
        """
        Export inventory timeseries to CSV (Plan 022 - Phase 3).

        Tracks dealer capacity utilization over time.

        Columns: run_id, regime, day, bucket, dealer_inventory, max_capacity,
                 capacity_pct, is_at_zero, hit_vbt_this_step, total_system_face,
                 dealer_share_pct
        """
        import csv

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if not self.dealer_snapshots:
            return

        fieldnames = [
            "run_id",
            "regime",
            "day",
            "bucket",
            "dealer_inventory",
            "max_capacity",
            "capacity_pct",
            "is_at_zero",
            "hit_vbt_this_step",
            "total_system_face",
            "dealer_share_pct",
        ]

        with open(path_obj, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for snap in sorted(self.dealer_snapshots, key=lambda s: (s.day, s.bucket)):
                writer.writerow(
                    {
                        "run_id": snap.run_id,
                        "regime": snap.regime,
                        "day": snap.day,
                        "bucket": snap.bucket,
                        "dealer_inventory": snap.inventory,
                        "max_capacity": snap.max_capacity,
                        "capacity_pct": str(snap.capacity_pct),
                        "is_at_zero": snap.is_at_zero,
                        "hit_vbt_this_step": snap.hit_vbt_this_step,
                        "total_system_face": str(snap.total_system_face),
                        "dealer_share_pct": str(snap.dealer_share_pct),
                    }
                )

    def to_system_state_csv(self, path: str) -> None:
        """
        Export system state timeseries to CSV (Plan 022 - Phase 4).

        Tracks how total debt/money evolves over time.

        Columns: run_id, regime, day, total_face_value, face_bucket_short,
                 face_bucket_mid, face_bucket_long, total_cash, debt_to_money
        """
        import csv

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if not self.system_state_snapshots:
            return

        with open(path_obj, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.system_state_snapshots[0].to_dict().keys())
            writer.writeheader()
            for snapshot in sorted(self.system_state_snapshots, key=lambda s: s.day):
                writer.writerow(snapshot.to_dict())

    def to_repayment_events_csv(self, path: str) -> None:
        """
        Export repayment events to CSV (Plan 022 - Phase 2).

        Tracks how each trader managed each liability - did they repay or default,
        and what trading strategy did they use?

        Columns: run_id, regime, trader_id, liability_id, maturity_day, face_value,
                 outcome, buy_count, sell_count, net_cash_pnl, strategy
        """
        import csv

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if not self.repayment_events:
            return

        with open(path_obj, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.repayment_events[0].to_dict().keys())
            writer.writeheader()
            for event in sorted(self.repayment_events, key=lambda e: (e.maturity_day, e.trader_id)):
                writer.writerow(event.to_dict())


# =============================================================================
# Helper Functions for Safety Margin Computation
# =============================================================================


def compute_safety_margin(
    cash: Decimal,
    tickets_held: list[Any],  # List of Ticket objects
    obligations: list[Any],  # List of Ticket objects (where trader is issuer)
    dealers: dict[str, Any],  # bucket_id -> DealerState
    ticket_size: Decimal,
) -> Decimal:
    """
    Compute deterministic safety margin m_i(t) for a trader.

    m_i(t) = A_i(t) - D_i(t)

    Where:
    - A_i(t) = cash + Σ (bid_price * face) for held tickets
    - D_i(t) = Σ face for remaining obligations

    Args:
        cash: Trader's cash holdings
        tickets_held: Tickets owned by trader
        obligations: Trader's outstanding obligations
        dealers: Dealer states by bucket (for bid prices)
        ticket_size: Standard ticket size S

    Returns:
        Safety margin m_i(t)
    """
    # A_i(t): Cash plus saleable value of tickets at dealer bid
    a_i = cash
    for ticket in tickets_held:
        bucket_id = ticket.bucket_id
        dealer = dealers.get(bucket_id)
        if dealer:
            # Value at dealer bid price
            a_i += dealer.bid * ticket.face
        else:
            # Fallback: use face value
            a_i += ticket.face

    # D_i(t): Sum of remaining obligations
    d_i: Decimal = sum((ticket.face for ticket in obligations), Decimal(0))

    return a_i - d_i


def compute_saleable_value(
    tickets_held: list[Any],
    dealers: dict[str, Any],
) -> Decimal:
    """
    Compute total saleable value of tickets at dealer bid prices.

    Args:
        tickets_held: Tickets owned by trader
        dealers: Dealer states by bucket

    Returns:
        Total value if all tickets sold at bid
    """
    total = Decimal(0)
    for ticket in tickets_held:
        bucket_id = ticket.bucket_id
        dealer = dealers.get(bucket_id)
        if dealer:
            total += dealer.bid * ticket.face
        else:
            total += ticket.face
    return total


# =============================================================================
# Liability Tracking (Plan 023 - Default-Aware Instrumentation)
# =============================================================================


@dataclass
class LiabilityInfo:
    """
    Information about a single liability from events.jsonl.

    Used by build_liability_map() to track all created payables
    and their settlement status.
    """

    trader_id: str  # Debtor agent ID
    liability_id: str  # Payable ID (e.g., "PAY_xyz789")
    maturity_day: int  # Due day of the liability
    face_value: Decimal  # Amount owed
    settled: bool = False  # Whether it was settled
    settled_day: int | None = None  # Day it was settled (if settled)


def build_liability_map(events: list[dict[str, Any]]) -> tuple[dict[str, LiabilityInfo], int]:
    """
    Build a map of all liabilities and their settlement status.

    Scans events for:
    - PayableCreated: Register liability with debtor, maturity, face value
    - PayableSettled: Mark liability as settled

    This approach captures ALL liabilities, including those that default
    without an explicit ObligationDefaulted event.

    Args:
        events: List of event dictionaries from system.state.events or events.jsonl

    Returns:
        (liabilities_map, final_day) where:
        - liabilities_map: {liability_id -> LiabilityInfo}
        - final_day: Last day seen in events
    """
    liabilities: dict[str, LiabilityInfo] = {}
    final_day = 0

    for event in events:
        day = event.get("day", 0)
        final_day = max(final_day, day)
        kind = event.get("kind", "") or event.get("event", "")

        if kind == "PayableCreated":
            lid = event.get("payable_id", "")
            if lid:
                liabilities[lid] = LiabilityInfo(
                    trader_id=event.get("debtor", ""),
                    liability_id=lid,
                    maturity_day=event.get("due_day", 0),
                    face_value=Decimal(str(event.get("amount", 0))),
                )
        elif kind == "PayableSettled":
            # Try multiple field names for the payable ID
            lid = event.get("pid") or event.get("contract_id") or event.get("payable_id", "")
            if lid and lid in liabilities:
                liabilities[lid].settled = True
                liabilities[lid].settled_day = day

    return liabilities, final_day


def compute_trading_stats_by_trader(trades: list[TradeRecord]) -> dict[str, list[dict[str, Any]]]:
    """
    Group trades by trader_id with trade details.

    Args:
        trades: List of TradeRecord objects

    Returns:
        {trader_id: [{"day": d, "side": s, "price": p}, ...]}
    """
    by_trader: dict[str, list[dict[str, Any]]] = {}

    for trade in trades:
        trader_id = trade.trader_id
        if trader_id not in by_trader:
            by_trader[trader_id] = []
        by_trader[trader_id].append(
            {
                "day": trade.day,
                "side": trade.side,
                "price": trade.price,
            }
        )

    return by_trader


def get_trades_before_day(trader_trades: list[dict[str, Any]], maturity_day: int) -> dict[str, Any]:
    """
    Compute trading stats for trades BEFORE a given maturity day.

    Args:
        trader_trades: List of trade dicts from compute_trading_stats_by_trader()
        maturity_day: The maturity day to filter against

    Returns:
        {"buy_count": N, "sell_count": N, "net_cash_pnl": Decimal}
    """
    buy_count = 0
    sell_count = 0
    net_cash_pnl = Decimal(0)

    for t in trader_trades:
        if t["day"] < maturity_day:
            if t["side"] == "BUY":
                buy_count += 1
                net_cash_pnl -= Decimal(str(t["price"]))
            elif t["side"] == "SELL":
                sell_count += 1
                net_cash_pnl += Decimal(str(t["price"]))

    return {
        "buy_count": buy_count,
        "sell_count": sell_count,
        "net_cash_pnl": net_cash_pnl,
    }


# =============================================================================
# Repayment Event Builder (Plan 022 - Phase 2, enhanced by Plan 023)
# =============================================================================


def build_repayment_events(
    event_log: list[dict[str, Any]],
    trades: list[TradeRecord],
    run_id: str = "",
    regime: str = "",
) -> list[RepaymentEvent]:
    """
    Build RepaymentEvent objects for ALL liabilities that matured.

    Plan 023 enhancement: Uses liability map approach to capture BOTH
    repaid AND defaulted liabilities, even when no explicit default event exists.

    Algorithm:
    1. Build map of all PayableCreated liabilities
    2. Mark settled liabilities from PayableSettled events
    3. For each liability that matured by final_day:
       - Compute trading stats from trades
       - Classify strategy
       - Set outcome based on settled status

    Args:
        event_log: List of event dictionaries from system.events or events.jsonl
        trades: List of TradeRecord objects from metrics.trades
        run_id: Run identifier for tracking
        regime: "passive" or "active"

    Returns:
        List of RepaymentEvent objects (one per matured liability)
    """
    # Build liability map from events
    liabilities, final_day = build_liability_map(event_log)

    # Build trading stats indexed by trader
    trading_stats = compute_trading_stats_by_trader(trades)

    repayment_events: list[RepaymentEvent] = []

    for lid, info in liabilities.items():
        # Only include liabilities that have matured by the final simulation day
        if info.maturity_day > final_day:
            continue

        # Skip if we don't have a valid trader_id
        if not info.trader_id:
            continue

        # Get trading stats for this trader BEFORE this maturity
        trader_trades = trading_stats.get(info.trader_id, [])
        stats = get_trades_before_day(trader_trades, info.maturity_day)

        buy_count = stats["buy_count"]
        sell_count = stats["sell_count"]
        net_cash_pnl = stats["net_cash_pnl"]

        # Classify strategy
        strategy = classify_trading_strategy(buy_count, sell_count)

        # Determine outcome: settled = repaid, not settled = defaulted
        outcome = "repaid" if info.settled else "defaulted"

        # Create RepaymentEvent
        repayment_event = RepaymentEvent(
            run_id=run_id,
            regime=regime,
            trader_id=info.trader_id,
            liability_id=lid,
            maturity_day=info.maturity_day,
            face_value=info.face_value,
            outcome=outcome,
            buy_count=buy_count,
            sell_count=sell_count,
            net_cash_pnl=net_cash_pnl,
            strategy=strategy,
        )
        repayment_events.append(repayment_event)

    return repayment_events
