"""
Data models for the dealer ring simulation.

This module defines the core data structures used in the dealer ring model:
- Tickets: Tradable debt instruments with face value and maturity
- Buckets: Maturity-based groupings of tickets
- DealerState: Per-bucket dealer market-making state
- VBTState: Per-bucket value-based trader state
- TraderState: Ring trader state with single-issuer constraint
"""

from dataclasses import dataclass, field
from decimal import Decimal

from bilancio.core.ids import AgentId

# Type alias for ticket identifiers
TicketId = str


@dataclass
class BucketConfig:
    """
    Configuration for a maturity bucket.

    Buckets partition tickets by remaining maturity τ (tau).
    Each bucket defines an inclusive range [tau_min, tau_max].
    tau_max=None means unbounded (τ ≥ tau_min).

    Attributes:
        name: Human-readable bucket identifier (e.g., "short", "mid", "long")
        tau_min: Minimum remaining maturity (inclusive)
        tau_max: Maximum remaining maturity (inclusive), None for unbounded
    """

    name: str
    tau_min: int
    tau_max: int | None


# Default three-bucket configuration
DEFAULT_BUCKETS = [
    BucketConfig("short", 1, 3),  # τ ∈ {1, 2, 3}
    BucketConfig("mid", 4, 8),  # τ ∈ {4, ..., 8}
    BucketConfig("long", 9, None),  # τ ≥ 9
]


@dataclass
class Ticket:
    """
    A tradable debt instrument (ticket).

    Represents a promise by issuer_id to pay face value on maturity_day.
    Currently held by owner_id (the creditor).

    Attributes:
        id: Unique ticket identifier
        issuer_id: Agent who owes this debt (the debtor)
        owner_id: Current holder of the ticket (the creditor)
        face: Face value S (typically 1)
        maturity_day: Absolute day when ticket matures
        remaining_tau: Remaining days to maturity (updated each day)
        bucket_id: Maturity bucket assignment ("short" | "mid" | "long")
        serial: Serial number for deterministic tie-breaking
    """

    id: TicketId
    issuer_id: AgentId
    owner_id: AgentId
    face: Decimal
    maturity_day: int
    remaining_tau: int = 0
    bucket_id: str = ""
    serial: int = 0


@dataclass
class DealerState:
    """
    Per-bucket dealer market-making state.

    Tracks inventory, cash, and derived quoting parameters for a single
    maturity bucket. The dealer provides two-sided liquidity using a
    value-based pricing function.

    Attributes:
        bucket_id: Maturity bucket identifier
        agent_id: Dealer's agent ID
        inventory: List of tickets currently held
        cash: Cash holdings

    Derived quantities (recomputed after each trade):
        a: Number of tickets held = len(inventory)
        x: Face inventory = a * S
        V: Mid-valued inventory = M*a + C
        K_star: Maximum buy tickets fundable
        X_star: One-sided capacity in face units
        N: Number of ladder rungs
        lambda_: Layoff probability
        I: Inside width (spread parameter)

    Current quotes:
        bid: Current bid price b_c(x)
        ask: Current ask price a_c(x)
        midline: Current midline p(x)

    Pin flags:
        is_pinned_bid: True if bid == B (at outside bid)
        is_pinned_ask: True if ask == A (at outside ask)
    """

    bucket_id: str
    agent_id: AgentId = ""
    inventory: list[Ticket] = field(default_factory=list)
    cash: Decimal = Decimal(0)

    # Derived quantities
    a: int = 0
    x: Decimal = Decimal(0)
    V: Decimal = Decimal(0)
    K_star: int = 0
    X_star: Decimal = Decimal(0)
    N: int = 1
    lambda_: Decimal = Decimal(1)
    I: Decimal = Decimal(0)  # noqa: E741

    # Quotes
    bid: Decimal = Decimal(0)
    ask: Decimal = Decimal(0)
    midline: Decimal = Decimal(0)

    # Pin flags
    is_pinned_bid: bool = False
    is_pinned_ask: bool = False

    def ticket_ids_by_issuer(self) -> dict[AgentId, list[TicketId]]:
        """
        Group inventory tickets by issuer.

        Returns:
            Dictionary mapping issuer_id to list of ticket IDs
        """
        result: dict[AgentId, list[TicketId]] = {}
        for ticket in self.inventory:
            if ticket.issuer_id not in result:
                result[ticket.issuer_id] = []
            result[ticket.issuer_id].append(ticket.id)
        return result


@dataclass
class VBTState:
    """
    Per-bucket value-based trader state.

    VBTs provide outside liquidity with adaptive anchors based on
    realized losses from defaults. Each bucket has independent anchors.

    Attributes:
        bucket_id: Maturity bucket identifier
        agent_id: VBT agent ID

    Anchors:
        M: Outside mid price
        O: Outside spread

    Derived outside quotes:
        A: Ask price = M + O/2
        B: Bid price = M - O/2

    Sensitivity parameters:
        phi_M: Mid sensitivity to loss (default 1.0)
        phi_O: Spread sensitivity to loss (default 0.6)

    Optional clipping:
        O_min: Minimum spread (default 0)
        clip_nonneg_B: Clip bid to be non-negative (default True)

    Balance sheet:
        inventory: List of tickets held
        cash: Cash holdings
    """

    bucket_id: str
    agent_id: AgentId = ""

    # Anchors
    M: Decimal = Decimal(1)
    O: Decimal = Decimal("0.30")  # noqa: E741

    # Derived outside quotes
    A: Decimal = Decimal(0)
    B: Decimal = Decimal(0)

    # Sensitivity parameters
    phi_M: Decimal = Decimal(1)
    phi_O: Decimal = Decimal("0.6")

    # Optional clipping
    O_min: Decimal = Decimal(0)
    clip_nonneg_B: bool = True

    # Balance sheet
    inventory: list[Ticket] = field(default_factory=list)
    cash: Decimal = Decimal(0)

    def recompute_quotes(self) -> None:
        """
        Update A and B from M and O with optional clipping.

        Computes:
            A = M + O/2
            B = M - O/2

        Applies clipping:
            - O >= O_min (minimum spread)
            - B >= 0 if clip_nonneg_B is True
        """
        # Ensure minimum spread
        O_effective = max(self.O, self.O_min)

        # Compute quotes
        half_spread = O_effective / 2
        self.A = self.M + half_spread
        self.B = self.M - half_spread

        # Cap ask at par: zero-coupon claims pay at most face at maturity
        self.A = min(self.A, Decimal(1))

        # Apply non-negative bid clipping if enabled
        if self.clip_nonneg_B:
            self.B = max(self.B, Decimal(0))

    def update_from_loss(self, loss_rate: Decimal) -> None:
        """
        Apply loss-based anchor update rule.

        Updates anchors based on realized loss rate:
            M_{t+1} = M_t - phi_M * l_t
            O_{t+1} = max(O_min, O_t + phi_O * l_t)

        Then recomputes quotes A and B.

        Args:
            loss_rate: Realized loss rate l_t (fraction of face value lost)
        """
        # Update mid (decreases with loss)
        self.M = self.M - self.phi_M * loss_rate

        # Update spread (increases with loss, subject to minimum)
        self.O = max(self.O_min, self.O + self.phi_O * loss_rate)

        # Recompute derived quotes
        self.recompute_quotes()


@dataclass
class TraderState:
    """
    State for a ring trader.

    Ring traders can hold tickets from multiple issuers. The asset_issuer_id
    tracks the issuer assigned via ring topology (payables / rollover) but does
    NOT constrain secondary-market purchases.

    Attributes:
        agent_id: Trader's agent ID
        cash: Cash holdings
        tickets_owned: List of tickets owned (assets)
        obligations: List of tickets this agent issued (liabilities)
        asset_issuer_id: Issuer assigned via ring topology (payables / rollover only)
        defaulted: True if agent has defaulted
    """

    agent_id: AgentId
    cash: Decimal = Decimal(0)
    tickets_owned: list[Ticket] = field(default_factory=list)
    obligations: list[Ticket] = field(default_factory=list)
    asset_issuer_id: AgentId | None = None
    defaulted: bool = False

    def payment_due(self, day: int) -> Decimal:
        """
        Calculate total payment obligations due on a given day.

        Args:
            day: Day to check for maturity obligations

        Returns:
            Total face value of tickets maturing on this day
        """
        return sum(
            (ticket.face for ticket in self.obligations if ticket.maturity_day == day),
            Decimal(0),
        )

    def shortfall(self, day: int) -> Decimal:
        """
        Calculate payment shortfall on a given day.

        Args:
            day: Day to check for shortfall

        Returns:
            max(0, payment_due - cash)
        """
        due = self.payment_due(day)
        return max(Decimal(0), due - self.cash)

    def upcoming_shortfall(self, start_day: int, horizon: int) -> Decimal:
        """Maximum shortfall across upcoming days (for urgency computation).

        Used by sell risk assessment to determine trader urgency. A seller
        with non-zero upcoming shortfall will have reduced risk thresholds,
        making them more willing to accept dealer bids below EV.
        """
        max_sf = Decimal(0)
        for day_offset in range(horizon + 1):
            max_sf = max(max_sf, self.shortfall(start_day + day_offset))
        return max_sf

    def earliest_liability_day(self, after_day: int) -> int | None:
        """
        Find the earliest liability date after the given day.

        Args:
            after_day: Find liabilities strictly after this day

        Returns:
            Earliest maturity day > after_day, or None if no future liabilities
        """
        future_days = [
            ticket.maturity_day for ticket in self.obligations if ticket.maturity_day > after_day
        ]
        return min(future_days) if future_days else None
